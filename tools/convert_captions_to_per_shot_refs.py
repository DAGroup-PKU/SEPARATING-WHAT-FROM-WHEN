#!/usr/bin/env python3
"""把训练 JSON 里的 MTSS caption 从顶层 references[] 转成 per-shot references[]。

转换规则 (写法 1):
  - 顶层 references[] → 拆到每个 shot 内的 references[]
  - ref 首次出现 → 完整 object {ref_id, type, id_features, attributes}
  - ref 复用 → 字符串 "REF_ID"
  - 删 semantic_description、appearance_anchor 嵌套、references_in_shot
  - 顶层 global_context 包装 → 拍平到顶层 (scene_description/global_style/global_audio)
  - attributes list → join 成 string

幂等: 已经是新 schema 的 caption (顶层无 references[],shot 内有 references[])
跳过不再转。

输入: 训练 JSON,JSON 数组,每个 item 形如 {"caption": "<MTSS JSON 字符串>",
"media_path": "..."}。

用法:
  # 单个文件,原地覆盖 (自动 .bak 备份)
  python tools/convert_captions_to_per_shot_refs.py dataset/train_mtss.json

  # 多个文件
  python tools/convert_captions_to_per_shot_refs.py dataset/train_mtss.json dataset/train_drama02.json

  # glob (注意 shell 展开)
  python tools/convert_captions_to_per_shot_refs.py dataset/train_*.json

  # 干跑,只统计不写盘
  python tools/convert_captions_to_per_shot_refs.py --dry-run dataset/train_mtss.json

  # 不备份 (默认会写 .bak)
  python tools/convert_captions_to_per_shot_refs.py --no-backup dataset/train_mtss.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import traceback
from pathlib import Path


NA_PLACEHOLDERS = {"N/A", "N.A.", "NONE", ""}

# 匹配 visual_description 里的 inline 时间标记: [0.0s] / [12.5s] / [7s] 等。
# caption_mtss.py 的 schema 明令禁止 ("Do NOT include inline time markers"),
# 但 Gemini 生成时常不遵守。这些标记纯粹是 token 浪费 (LLM 对纯数字 token
# 推理弱;真正的时间锚靠 RoTE K-rotation),还和模型的 RoTE 信号冗余/打架。
_INLINE_TIME_MARKER = re.compile(r"\s*\[\d+(?:\.\d+)?s\]\s*")


def strip_inline_time_markers_in_dict(mtss: dict) -> bool:
    """对一个 MTSS caption dict in-place 删 visual_description 里的 [X.Xs] 标记。

    返回是否真的发生了改动 (用于幂等检测)。
    """
    changed = False
    for shot in mtss.get("shots", []) or []:
        if not isinstance(shot, dict):
            continue
        desc = shot.get("visual_description")
        if not isinstance(desc, str):
            continue
        new_desc = _INLINE_TIME_MARKER.sub(" ", desc)
        # 收尾: 连续空格归一, 首尾 strip
        new_desc = re.sub(r"\s+", " ", new_desc).strip()
        if new_desc != desc:
            shot["visual_description"] = new_desc
            changed = True
    return changed


def localize_references(mtss: dict) -> dict:
    """把一个 MTSS caption dict 转成 per-shot references schema。

    返回新 dict,不修改输入。

    转换语义:
      - 顶层 references[] 提取出来,按 references_in_shot 顺序分发到每个 shot
      - shot 内 [REF_ID] 占位符但 references_in_shot 漏列的 ref,也补进 shot.references
      - ref 首次出现走 object 形态,后续 shot 走字符串复用
      - 删字段: semantic_description, appearance_anchor 嵌套层, references_in_shot
      - global_context 包装拍平
    """
    import re

    refs_by_id = {r["ref_id"]: r for r in mtss.get("references", []) or [] if isinstance(r, dict) and r.get("ref_id")}
    introduced: set[str] = set()

    new_shots: list[dict] = []
    for shot in mtss.get("shots", []) or []:
        if not isinstance(shot, dict):
            new_shots.append(shot)
            continue

        desc = shot.get("visual_description", "") or ""
        ris = shot.get("references_in_shot") or []
        in_desc = re.findall(r"\[([A-Z]+_\d+)\]", desc)

        # 顺序: references_in_shot 优先,desc 中漏列的补在后
        ordered: list[str] = []
        seen_local: set[str] = set()
        for rid in list(ris) + in_desc:
            if rid in seen_local:
                continue
            seen_local.add(rid)
            ordered.append(rid)

        new_refs: list = []
        for rid in ordered:
            if rid not in refs_by_id:
                continue
            if rid in introduced:
                new_refs.append(rid)
            else:
                ref = refs_by_id[rid]
                anchor = ref.get("appearance_anchor") or {}
                id_feat_obj = anchor.get("id_features") or {}
                # id_features 兼容 dict ({"detail_description": "..."}) 或 string
                if isinstance(id_feat_obj, dict):
                    id_feat = id_feat_obj.get("detail_description", "") or id_feat_obj.get("description", "") or ""
                elif isinstance(id_feat_obj, str):
                    id_feat = id_feat_obj
                else:
                    id_feat = ""

                attrs = anchor.get("attributes")
                if isinstance(attrs, list):
                    attrs = ", ".join(str(x) for x in attrs)
                elif isinstance(attrs, dict):
                    attrs = ", ".join(f"{k}: {v}" for k, v in attrs.items() if v)
                elif not isinstance(attrs, str):
                    attrs = ""
                if attrs.strip().upper() in NA_PLACEHOLDERS:
                    attrs = ""

                obj: dict = {"ref_id": rid, "type": ref.get("type")}
                if id_feat:
                    obj["id_features"] = str(id_feat).rstrip(".")
                if attrs:
                    obj["attributes"] = str(attrs).rstrip(".")
                new_refs.append(obj)
                introduced.add(rid)

        new_shot = {
            "shot_id": shot.get("shot_id"),
            "time_range": shot.get("time_range"),
            "references": new_refs,
            "visual_description": desc,
        }
        # 保留其他可选字段
        for k in ("camera", "active_events"):
            if k in shot:
                new_shot[k] = shot[k]
        new_shots.append(new_shot)

    new_data: dict = {"shots": new_shots}

    for k in ("events", "subtitles"):
        if k in mtss:
            new_data[k] = mtss[k]

    # global_context 包装拍平
    gc = mtss.get("global_context")
    if isinstance(gc, dict):
        for k in ("scene_description", "global_style", "global_audio"):
            if k in gc:
                new_data[k] = gc[k]
    for k in ("scene_description", "global_style", "global_audio"):
        if k in mtss and k not in new_data:
            new_data[k] = mtss[k]

    # 顺带清掉 visual_description 里的 [X.Xs] inline time markers
    # (caption_mtss.py schema 明令禁止, 模型理解不了纯数字 token,
    # 时间信号已通过 RoTE K-rotation 编码)
    strip_inline_time_markers_in_dict(new_data)

    return new_data


def is_already_new_schema(mtss: dict) -> bool:
    """判断 caption 是不是已经是 per-shot references schema (幂等检测)。

    条件: 顶层无 references[] 字段, 且 shots 不为空 且 至少一个 shot 内有
    references 字段。
    """
    if "references" in mtss:
        return False
    shots = mtss.get("shots", []) or []
    if not shots:
        return False
    for s in shots:
        if isinstance(s, dict) and "references" in s:
            return True
    return False


def convert_one_item(item: dict) -> tuple[bool, str | None]:
    """转一条 item。返回 (是否实际转换, 错误信息或 None)。

    分支:
      - 旧 schema → localize_references (含 inline time marker 清洗)
      - 新 schema 但还含 inline time marker → 单独再清洗
      - 新 schema 且无 inline time marker → 跳过 (幂等)
    """
    cap_str = item.get("caption")
    if not isinstance(cap_str, str):
        return False, "caption 字段不是字符串"
    try:
        mtss = json.loads(cap_str)
    except json.JSONDecodeError as e:
        return False, f"caption JSON 解析失败: {e}"
    if not isinstance(mtss, dict):
        return False, "caption 解析后不是 dict"

    if is_already_new_schema(mtss):
        # 不重做 schema 转换, 但补一次 inline time marker 清洗 (用户可能已经
        # 转过 schema, 这次想再清干净 markers)
        if strip_inline_time_markers_in_dict(mtss):
            item["caption"] = json.dumps(mtss, ensure_ascii=False)
            return True, None
        return False, None  # 已干净,真正跳过

    new_mtss = localize_references(mtss)
    item["caption"] = json.dumps(new_mtss, ensure_ascii=False)
    return True, None


def convert_file(path: Path, dry_run: bool, backup: bool, remove_broken: bool = False) -> dict:
    """转一个文件。返回 stats dict。

    remove_broken=True 时,把 caption 无法 json.loads 的 item 从输出 list 里
    剔除掉 (而不是保留原样)。避免下游 process_captions.py 在这些坏数据上
    反复挂掉。被剔除的 item 数量记在 stats['removed']。
    """
    stats = {
        "path": str(path),
        "items": 0,
        "converted": 0,
        "already_new": 0,
        "errors": 0,
        "removed": 0,
        "size_before": 0,
        "size_after": 0,
        "error_samples": [],
    }
    if not path.is_file():
        stats["errors"] = 1
        stats["error_samples"].append(f"file not found: {path}")
        return stats

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        stats["errors"] = 1
        stats["error_samples"].append(f"load failed: {e}")
        return stats

    if not isinstance(data, list):
        stats["errors"] = 1
        stats["error_samples"].append("top-level 不是 list,跳过")
        return stats

    stats["items"] = len(data)
    stats["size_before"] = path.stat().st_size

    # 用 None 标记要剔除的 item, 处理完再过滤
    kept: list = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            stats["errors"] += 1
            if len(stats["error_samples"]) < 5:
                stats["error_samples"].append(f"item[{idx}] 不是 dict")
            kept.append(item)
            continue
        try:
            changed, err = convert_one_item(item)
            if err:
                stats["errors"] += 1
                if len(stats["error_samples"]) < 5:
                    stats["error_samples"].append(f"item[{idx}] {err}")
                if remove_broken:
                    stats["removed"] += 1
                    continue  # 跳过 kept.append
            elif changed:
                stats["converted"] += 1
            else:
                stats["already_new"] += 1
        except Exception as e:
            stats["errors"] += 1
            if len(stats["error_samples"]) < 5:
                stats["error_samples"].append(f"item[{idx}] crash: {type(e).__name__}: {e}")
            if remove_broken:
                stats["removed"] += 1
                continue
        kept.append(item)
    data = kept

    if dry_run:
        # 估算转换后的体积
        out_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        stats["size_after"] = len(out_bytes)
        return stats

    if stats["converted"] == 0 and stats["removed"] == 0:
        # 没有任何条目实际转换、也没剔除,不写盘
        stats["size_after"] = stats["size_before"]
        return stats

    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    stats["size_after"] = path.stat().st_size
    return stats


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="一个或多个训练 JSON 文件 (top-level list of {caption, media_path})")
    ap.add_argument("--dry-run", action="store_true", help="只统计不写盘")
    ap.add_argument("--no-backup", action="store_true", help="不写 .bak 备份")
    ap.add_argument("--remove-broken", action="store_true",
                    help="把 caption JSON 损坏的 item 从输出里剔除 (默认: 保留原样)。"
                         "避免下游 process_captions.py 在坏数据上反复挂掉。")
    args = ap.parse_args(argv)

    total = {"items": 0, "converted": 0, "already_new": 0, "errors": 0, "removed": 0,
             "size_before": 0, "size_after": 0}

    for fp in args.files:
        path = Path(fp)
        print(f"\n=== {path} ===", flush=True)
        stats = convert_file(path, dry_run=args.dry_run, backup=not args.no_backup,
                             remove_broken=args.remove_broken)

        print(f"  items:        {stats['items']}")
        print(f"  converted:    {stats['converted']}")
        print(f"  already-new:  {stats['already_new']}  (幂等跳过)")
        print(f"  errors:       {stats['errors']}")
        if args.remove_broken and stats["removed"]:
            print(f"  removed:      {stats['removed']}  (坏 item 已剔除)")
        if stats["error_samples"]:
            print(f"  errors 样本 (前 5 条):")
            for e in stats["error_samples"]:
                print(f"    - {e}")
        if stats["size_before"]:
            delta = stats["size_after"] - stats["size_before"]
            pct = (1 - stats["size_after"] / stats["size_before"]) * 100 if stats["size_before"] else 0
            print(f"  bytes:        {stats['size_before']:,} → {stats['size_after']:,} "
                  f"({'-' if delta < 0 else '+'}{abs(pct):.1f}%)")

        for k in ("items", "converted", "already_new", "errors", "removed", "size_before", "size_after"):
            total[k] += stats[k]

    if len(args.files) > 1:
        print(f"\n=== 合计 ===")
        print(f"  items: {total['items']}, converted: {total['converted']}, "
              f"already-new: {total['already_new']}, errors: {total['errors']}, "
              f"removed: {total['removed']}")
        if total["size_before"]:
            pct = (1 - total["size_after"] / total["size_before"]) * 100
            print(f"  total bytes: {total['size_before']:,} → {total['size_after']:,} ({pct:+.1f}%)")

    if args.dry_run:
        print("\n(--dry-run, 未写盘)")

    unhandled_errors = total["errors"] - total["removed"] if args.remove_broken else total["errors"]
    return 0 if unhandled_errors <= 0 else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
