"""Annotate text tokens with (t_start, t_end) intervals (seconds) for TCR.

Given an MTSS JSON caption and a Hugging Face fast tokenizer, produce a
per-token (t_s, t_e) tensor that Temporal Context Routing consumes. The same
packed intervals are also what the rotary (RoTE) baseline reads.

Per-token rules:
  - shots[i] tokens          : the shot's time_range
  - events[i] tokens         : the event's time_range
  - subtitles[i] tokens      : the subtitle's time_range
  - references[i] tokens     : priority chain
                                 0. ref's own time_range field if present (v2
                                    captures it into timing_map["references"])
                                 1. time_range of the shot whose interval
                                    contains appearance_anchor.id_features.timestamp
                                    — center matches the labeled "best observation
                                    moment", radius matches that appearance's actual
                                    duration, sinc envelope concentrates K within
                                    the window (TIE-faithful)
                                 2. ts set but no shot contains it → (ts, ts)
                                 3. ts missing → time_range of the first shot
                                    whose references_in_shot lists this ref_id
  - global_style / global_audio / scene_description / structural chars
                             : (0, T_total) — full clip
  - padding / special (BOS/EOS) tokens
                             : (-1, -1) sentinel — RoPE replaced with identity

Times are kept in seconds throughout; the cross-attn RoTE consumes them as-is.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

import torch


NO_ROPE_SENTINEL = -1.0


class TokenizerLike(Protocol):
    """HF fast tokenizer (or LTXVGemmaTokenizer wrapping one)."""

    def __call__(self, text: str, **kwargs: Any) -> Any: ...


def _resolve_hf_tokenizer(tokenizer: Any) -> Any:
    """Accept either a raw HF tokenizer or an LTXVGemmaTokenizer wrapper."""
    inner = getattr(tokenizer, "tokenizer", None)
    return inner if inner is not None else tokenizer


def _timestamp_to_seconds(ts: Any) -> float | None:
    """Parse 'MM:SS', 'M:SS', 'H:MM:SS', or numeric to seconds. None on failure."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if not isinstance(ts, str):
        return None
    s = ts.strip()
    if not s:
        return None
    if ":" in s:
        parts = s.split(":")
        try:
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def annotate_time_ranges(
    caption_json: str,
    tokenizer: TokenizerLike,
    seq_len: int,
    clip_duration: float | None = None,
) -> torch.Tensor:
    """Build per-token (t_s, t_e) seconds annotations for an MTSS caption.

    Args:
        caption_json: MTSS caption JSON string.
        tokenizer: HF fast tokenizer (or LTXVGemmaTokenizer wrapper).
        seq_len: Padded token sequence length.
        clip_duration: Total clip duration in seconds; inferred from JSON if None.

    Returns:
        Tensor of shape (seq_len, 2). Padding / special tokens are
        (NO_ROPE_SENTINEL, NO_ROPE_SENTINEL).
    """
    hf_tok = _resolve_hf_tokenizer(tokenizer)

    # Strip to match LTXVGemmaTokenizer.tokenize_with_weights, which calls text.strip()
    # before encoding. Without this, char offsets would be shifted by the leading
    # whitespace and per-token (t_s, t_e) would be misaligned against the embeddings.
    caption_json = caption_json.strip()

    # ---- Parse JSON; fall back to all-no-RoPE on failure ----
    try:
        data = json.loads(caption_json)
        if not isinstance(data, dict):
            raise ValueError("MTSS JSON must be an object at the top level")
    except (json.JSONDecodeError, TypeError, ValueError):
        return torch.full((seq_len, 2), NO_ROPE_SENTINEL, dtype=torch.float32)

    if clip_duration is None:
        clip_duration = _infer_duration(data)

    char_anno = _build_char_annotations(caption_json, data, clip_duration)

    # ---- Tokenize with offsets ----
    enc = hf_tok(
        caption_json,
        return_offsets_mapping=True,
        return_attention_mask=True,
        padding="max_length",
        padding_side="left",
        max_length=seq_len,
        truncation=True,
    )
    offsets = enc["offset_mapping"]
    attn = enc["attention_mask"]

    n = len(caption_json)
    positions = torch.full((seq_len, 2), NO_ROPE_SENTINEL, dtype=torch.float32)
    for i in range(min(seq_len, len(offsets))):
        if attn[i] == 0:
            continue  # padding
        cs, ce = offsets[i]
        if cs == ce:
            continue  # special tokens (BOS/EOS) carry an empty char span
        mid = min((cs + ce) // 2, n - 1)
        positions[i] = char_anno[mid]
    return positions


def build_context_positions_from_captions(
    captions: list[str],
    tokenizer: TokenizerLike,
    seq_len: int,
    clip_durations: list[float] | None = None,
) -> torch.Tensor:
    """Batch helper. Returns (B, seq_len, 2)."""
    return torch.stack(
        [
            annotate_time_ranges(
                cap,
                tokenizer,
                seq_len,
                clip_durations[i] if clip_durations is not None else None,
            )
            for i, cap in enumerate(captions)
        ],
        dim=0,
    )


def repack_to_connector_layout(
    positions: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Repack (t_s, t_e) to mirror Embeddings1DConnector's left-aligned output.

    The Gemma tokenizer left-pads, so ``annotate_time_ranges`` returns positions
    in the layout ``[sentinel..., real_0, ..., real_{n-1}]``. The text connector
    (``Embeddings1DConnector._replace_padded_with_learnable_registers``) physically
    reorders the K embeddings to ``[real_0, ..., real_{n-1}, register, ...]``.
    To keep RoTE rotations aligned with the actual K tokens, ``context_positions``
    must undergo the same permutation: real (t_s, t_e) packed at the front,
    NO_ROPE_SENTINEL filling the register slots so registers skip RoPE.

    Args:
        positions: (T, 2) or (B, T, 2). Per-token (t_s, t_e) before repack.
        attention_mask: (T,) or (B, T). 1 = real token (incl. BOS/EOS), 0 = pad.

    Returns:
        Same shape as ``positions``, repacked.
    """
    if positions.dim() == 2:
        return _repack_single(positions, attention_mask)
    if positions.dim() == 3:
        out = torch.full_like(positions, NO_ROPE_SENTINEL)
        for b in range(positions.shape[0]):
            out[b] = _repack_single(positions[b], attention_mask[b])
        return out
    raise ValueError(f"positions must be 2D or 3D, got shape {tuple(positions.shape)}")


def _repack_single(positions: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(positions, NO_ROPE_SENTINEL)
    m = attention_mask.bool()
    n = int(m.sum().item())
    if n:
        out[:n] = positions[m]
    return out


def annotate_and_pack_for_rote(
    caption_json: str,
    tokenizer: TokenizerLike,
    seq_len: int,
    clip_duration: float | None = None,
) -> torch.Tensor:
    """``annotate_time_ranges`` + ``repack_to_connector_layout`` in one call.

    Returns positions in the layout the text connector produces — real-token
    (t_s, t_e) at indices [0, real_len), NO_ROPE_SENTINEL after. This is the
    form the K-side RoTE expects, since ``Embeddings1DConnector`` repacks real
    tokens to the front before cross-attention sees them.
    """
    hf_tok = _resolve_hf_tokenizer(tokenizer)
    enc = hf_tok(
        caption_json.strip(),
        return_attention_mask=True,
        padding="max_length",
        padding_side="left",
        max_length=seq_len,
        truncation=True,
    )
    attn = torch.as_tensor(enc["attention_mask"], dtype=torch.long)
    positions = annotate_time_ranges(caption_json, tokenizer, seq_len, clip_duration)
    return repack_to_connector_layout(positions, attn)


def strip_time_ranges(prompt_json: str) -> tuple[str, dict[str, list[tuple[float, float] | None]]]:
    """Strip ``time_range`` fields from MTSS JSON prompt, return stripped string + timing_map.

    Why: ``"time_range": [0.0, 2.0]`` chars are tokenized into ~8 Gemma tokens per
    shot/event/subtitle and convey no signal the model can actually use (LLMs are
    weak at numeric reasoning). The real time info goes via RoTE K-rotation. So
    we strip the literal field from the text fed to the tokenizer, but keep the
    values in a side-channel ``timing_map`` for the annotator.

    Args:
        prompt_json: Original MTSS JSON string (with ``time_range`` fields).

    Returns:
        (stripped_json, timing_map) where:
        - ``stripped_json`` is the JSON string with all ``time_range`` fields removed
        - ``timing_map`` is ``{"shots": [(ts, te), ...], "events": [...], "subtitles": [...]}``,
          index-aligned with the corresponding section's items. Missing time_range → None.
    """
    # Tolerate non-JSON input (e.g. plain-text negative prompts like
    # "worst quality, blurry, ...") — return input unchanged with empty timing.
    try:
        data = json.loads(prompt_json)
    except (json.JSONDecodeError, TypeError, ValueError):
        return prompt_json, {"shots": [], "events": [], "subtitles": []}
    if not isinstance(data, dict):
        return prompt_json, {"shots": [], "events": [], "subtitles": []}

    timing_map: dict[str, list[tuple[float, float] | None]] = {}
    for section in ("shots", "events", "subtitles"):
        items = data.get(section, []) or []
        section_timing: list[tuple[float, float] | None] = []
        for item in items:
            tr = item.pop("time_range", None) if isinstance(item, dict) else None
            if tr and len(tr) >= 2:
                section_timing.append((float(tr[0]), float(tr[1])))
            else:
                section_timing.append(None)
        timing_map[section] = section_timing
    # If some caption tools emit a top-level "time_range" on references[]
    # entries, capture it into timing_map["references"] (keyed by ref_id) so
    # the v2 annotator can use it as the highest-priority time anchor for that
    # ref's K tokens, ahead of the timestamp→containing-shot path. Strip the
    # literal from the text — the model would just see "0.0, 2.0" digits that
    # carry no signal it can leverage; the real time info goes via RoTE.
    ref_timing: dict[str, tuple[float, float]] = {}
    for ref in data.get("references", []) or []:
        if not isinstance(ref, dict):
            continue
        tr = ref.pop("time_range", None)
        rid = ref.get("ref_id")
        if rid and tr and len(tr) >= 2:
            ref_timing[rid] = (float(tr[0]), float(tr[1]))
    if ref_timing:
        timing_map["references"] = ref_timing
    stripped = json.dumps(data, ensure_ascii=False)
    return stripped, timing_map


def _restore_time_ranges_into_data(
    data: dict[str, Any],
    timing_map: dict[str, Any],
) -> None:
    """In-place inject timing_map back into data dict (NOT the JSON string).

    Used by v2 annotator: the JSON STRING fed to the annotator is stripped
    (for char-span alignment with tokenizer), but the PARSED dict needs
    time_range values for the time-anchor computation logic.

    timing_map shape:
      - "shots"/"events"/"subtitles": index-aligned list of (t_s, t_e) | None
      - "references" (optional): {ref_id: (t_s, t_e)} for refs that have an
        explicit time_range field; restored so _compute_reference_intervals
        can prefer it over the timestamp→containing-shot path.
    """
    for section in ("shots", "events", "subtitles"):
        tr_list = timing_map.get(section, [])
        items = data.get(section, []) or []
        for i, tr in enumerate(tr_list):
            if i >= len(items) or not isinstance(items[i], dict):
                continue
            if tr is not None:
                items[i]["time_range"] = [tr[0], tr[1]]

    ref_timing = timing_map.get("references") or {}
    if ref_timing:
        for ref in data.get("references", []) or []:
            if not isinstance(ref, dict):
                continue
            tr = ref_timing.get(ref.get("ref_id"))
            if tr is not None:
                ref["time_range"] = [tr[0], tr[1]]


def annotate_time_ranges_v2(
    stripped_json: str,
    tokenizer: TokenizerLike,
    seq_len: int,
    timing_map: dict[str, list[tuple[float, float] | None]],
    clip_duration: float | None = None,
) -> torch.Tensor:
    """Same semantics as :func:`annotate_time_ranges` but consumes stripped JSON + side-channel timing_map.

    The stripped JSON has no ``"time_range": [...]`` text, so the tokenizer sees fewer tokens
    and the char-span layout differs from the original. We re-parse the stripped JSON, inject
    the timing back into the data dict (NOT the string), then run the standard annotation logic.
    Char spans for ``_mark_timed_items`` come from the stripped string, matching what the
    tokenizer sees.
    """
    hf_tok = _resolve_hf_tokenizer(tokenizer)
    stripped_json = stripped_json.strip()

    try:
        data = json.loads(stripped_json)
        if not isinstance(data, dict):
            raise ValueError("MTSS JSON must be an object at the top level")
    except (json.JSONDecodeError, TypeError, ValueError):
        return torch.full((seq_len, 2), NO_ROPE_SENTINEL, dtype=torch.float32)

    # Restore time_range values into the parsed dict (string stays stripped).
    _restore_time_ranges_into_data(data, timing_map)

    if clip_duration is None:
        clip_duration = _infer_duration(data)

    char_anno = _build_char_annotations(stripped_json, data, clip_duration)

    enc = hf_tok(
        stripped_json,
        return_offsets_mapping=True,
        return_attention_mask=True,
        padding="max_length",
        padding_side="left",
        max_length=seq_len,
        truncation=True,
    )
    offsets = enc["offset_mapping"]
    attn = enc["attention_mask"]

    n = len(stripped_json)
    positions = torch.full((seq_len, 2), NO_ROPE_SENTINEL, dtype=torch.float32)
    for i in range(min(seq_len, len(offsets))):
        if attn[i] == 0:
            continue
        cs, ce = offsets[i]
        if cs == ce:
            continue
        mid = min((cs + ce) // 2, n - 1)
        positions[i] = char_anno[mid]
    return positions


def annotate_and_pack_intervals(
    stripped_json: str,
    tokenizer: TokenizerLike,
    seq_len: int,
    timing_map: dict[str, list[tuple[float, float] | None]],
    clip_duration: float | None = None,
) -> torch.Tensor:
    """Pack per-token intervals from stripped JSON + the side-channel timing map.

    Use this together with :func:`strip_time_ranges` so the text encoder never
    sees ``"time_range": [...]`` while TCR still receives index-aligned intervals.
    """
    hf_tok = _resolve_hf_tokenizer(tokenizer)
    enc = hf_tok(
        stripped_json.strip(),
        return_attention_mask=True,
        padding="max_length",
        padding_side="left",
        max_length=seq_len,
        truncation=True,
    )
    attn = torch.as_tensor(enc["attention_mask"], dtype=torch.long)
    positions = annotate_time_ranges_v2(
        stripped_json, tokenizer, seq_len, timing_map, clip_duration
    )
    return repack_to_connector_layout(positions, attn)


# Backward-compatible alias used by older call sites.
annotate_and_pack_for_rote_v2 = annotate_and_pack_intervals


def make_no_rope_positions(seq_len: int, batch_size: int = 1) -> torch.Tensor:
    """All-sentinel context_positions: every K token bypasses RoTE.

    Use this for the CFG negative pass and for prompt-dropout / null-prompt
    samples — there is no semantic time anchor, so K should not get any RoPE
    rotation. Shape matches what ``annotate_and_pack_for_rote`` returns
    (with the leading batch axis): ``(batch_size, seq_len, 2)``.
    """
    return torch.full((batch_size, seq_len, 2), NO_ROPE_SENTINEL, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _infer_duration(data: dict[str, Any]) -> float:
    """Maximum end-time across all time_range fields in shots/events/subtitles."""
    max_t = 0.0
    for section in ("shots", "events", "subtitles"):
        for item in data.get(section, []) or []:
            tr = item.get("time_range")
            if tr and len(tr) >= 2:
                max_t = max(max_t, float(tr[0]), float(tr[1]))
    return max(max_t, 1.0)


def _build_char_annotations(
    json_str: str,
    data: dict[str, Any],
    clip_duration: float,
) -> torch.Tensor:
    """(N_chars, 2) per-char (t_s, t_e). Default (0, T_total)."""
    n = len(json_str)
    anno = torch.zeros(n, 2, dtype=torch.float32)
    anno[:, 1] = clip_duration  # default = full clip

    # Shots / events / subtitles: each item gets its own time_range.
    for section in ("shots", "events", "subtitles"):
        items = data.get(section, []) or []
        if items:
            _mark_timed_items(json_str, section, items, anno, _item_time_range)

    # References: use shots[].references_in_shot union; fallback to timestamp anchor.
    refs = data.get("references", []) or []
    if refs:
        ref_time_map = _compute_reference_intervals(refs, data.get("shots", []) or [])
        _mark_timed_items(
            json_str,
            "references",
            refs,
            anno,
            lambda item: ref_time_map.get(item.get("ref_id")),
        )

    return anno


def _item_time_range(item: dict[str, Any]) -> tuple[float, float] | None:
    tr = item.get("time_range")
    if tr and len(tr) >= 2:
        return float(tr[0]), float(tr[1])
    return None


def _find_shot_containing_ts(
    shots: list[dict[str, Any]], ts: float
) -> list[float] | None:
    """Pick the shot whose interval contains ``ts``.

    Adjacent shots in MTSS share boundaries (e.g., SHOT_4 ends at 7, SHOT_5
    starts at 7). At a shared boundary, ``ts`` semantically belongs to the
    shot starting there, not the one ending there. Resolution rule:

      1. Prefer a shot where ``t_s <= ts < t_e`` (half-open interval).
      2. If none matches (only happens when ts == clip end == last shot's
         t_e), fall back to ``t_s <= ts <= t_e``.

    Returns the time_range list, or ``None`` if no shot contains ``ts``.
    """
    boundary_match: list[float] | None = None
    for s in shots:
        tr = s.get("time_range")
        if not tr:
            continue
        t_s, t_e = tr[0], tr[1]
        if t_s <= ts < t_e:
            return tr
        if t_s <= ts <= t_e:
            boundary_match = tr
    return boundary_match


def _compute_reference_intervals(
    refs: list[dict[str, Any]],
    shots: list[dict[str, Any]],
) -> dict[str, tuple[float, float]]:
    """For each ref_id: the time_range of the shot whose interval contains
    `appearance_anchor.id_features.timestamp`.

    Rationale: TIE-faithful interval encoding requires both a correct center
    AND a meaningful radius — sinc envelope is what makes K attention
    concentrate within a window. Using the containing shot:

      * center = (t_s + t_e) / 2 ≈ ts (exact when ts is at the shot midpoint;
        off by ≤ shot/2 when ts is at the boundary).
      * radius = (t_e − t_s) / 2 = the actual duration of this appearance,
        so sinc decays smoothly outside the shot. Q frames inside the shot
        get strong agreement; Q frames in unrelated shots fall off naturally.

    This is strictly better than:
      * shot-union (min, max): center collapses to the union midpoint, which
        often lies between appearances where the entity is *not* present.
      * point anchor (ts, ts): radius=0 → sinc≡1 → loses TIE's segmental
        attention concentration; degenerates to plain point RoPE.

    Priority chain (highest to lowest):
      0. ref has explicit ``time_range`` field → use it directly (no shot lookup).
         For v2 path this comes via timing_map["references"] which is restored
         into the ref dict before this function runs.
      1. ``appearance_anchor.id_features.timestamp`` is set and falls within
         some shot → use that shot's time_range.
      2. ts set but no shot contains it → point anchor (ts, ts).
      3. ts missing/unparseable → time_range of the first shot (in document
         order) whose ``references_in_shot`` lists this ref_id.
      4. none of the above → ref_id absent from map (chars get default (0, T_total)).

    Multi-appearance refs (e.g., PERSON_1 in SHOT_1 and SHOT_3) are handled
    separately: each ``[PERSON_1]`` placeholder in a shot's visual_description
    is a distinct K-token inheriting that shot's time_range, so per-shot
    attention is carried by those placeholders. The references[] block's
    K-tokens provide the stable identity anchor at the labeled best moment.
    """
    out: dict[str, tuple[float, float]] = {}
    for ref in refs:
        ref_id = ref.get("ref_id")
        if not ref_id:
            continue
        # Highest-priority: an explicit time_range on the ref itself. Caption
        # tools that emit this convey "this entity exists / is most clearly
        # described over [t_s, t_e]" directly, no shot lookup needed.
        explicit_tr = ref.get("time_range") if isinstance(ref, dict) else None
        if explicit_tr and len(explicit_tr) >= 2:
            out[ref_id] = (float(explicit_tr[0]), float(explicit_tr[1]))
            continue
        # `.get(k, {})` only returns the default when k is MISSING; if the
        # caption JSON has `"appearance_anchor": null` (or null one level
        # deeper) the .get returns None and the next .get crashes. Coerce
        # None at every level with `or {}`.
        ts_raw = (
            ((ref.get("appearance_anchor") or {})
             .get("id_features") or {})
            .get("timestamp")
        )
        ts = _timestamp_to_seconds(ts_raw)
        if ts is not None:
            anchor = _find_shot_containing_ts(shots, ts)
            if anchor is not None:
                out[ref_id] = (float(anchor[0]), float(anchor[1]))
                continue
            # ts set but no shot contains it — degenerate fallback.
            out[ref_id] = (ts, ts)
            continue
        # ts missing/unparseable: fall back to the first shot (in document
        # order) whose references_in_shot lists this ref_id.
        for s in shots:
            if ref_id in (s.get("references_in_shot") or []):
                tr = s.get("time_range")
                if tr and len(tr) >= 2:
                    out[ref_id] = (float(tr[0]), float(tr[1]))
                    break
    return out


def _mark_timed_items(
    json_str: str,
    section_name: str,
    items: list[dict[str, Any]],
    anno: torch.Tensor,
    time_range_fn: Any,
) -> None:
    """Walk the array section's char span and mark each {…} item with its time."""
    pattern = rf'"{section_name}"\s*:\s*\['
    match = re.search(pattern, json_str)
    if not match:
        return

    pos = match.end()  # past '['
    item_idx = 0
    while item_idx < len(items) and pos < len(json_str):
        while pos < len(json_str) and json_str[pos] in " \t\n\r,":
            pos += 1
        if pos >= len(json_str) or json_str[pos] == "]":
            break
        if json_str[pos] != "{":
            pos += 1
            continue

        obj_end = _find_matching_brace(json_str, pos)
        if obj_end is None:
            break

        tr = time_range_fn(items[item_idx])
        if tr is not None:
            anno[pos : obj_end + 1, 0] = tr[0]
            anno[pos : obj_end + 1, 1] = tr[1]

        pos = obj_end + 1
        item_idx += 1


def _find_matching_brace(s: str, pos: int) -> int | None:
    """Find the closing `}` that matches the `{` at `pos`."""
    if pos >= len(s) or s[pos] != "{":
        return None
    depth = 1
    i = pos + 1
    in_string = False
    escape = False
    while i < len(s) and depth > 0:
        c = s[i]
        if escape:
            escape = False
        elif c == "\\":
            escape = True
        elif c == '"':
            in_string = not in_string
        elif not in_string:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
        i += 1
    return i - 1 if depth == 0 else None
