# Known Issues — ltx-pipelines

未修复的已知问题。打算用这个包跑 production inference 前先看一眼。

## 1. Production pipelines 完全没用 RoTE（严重）

**影响范围**：所有用 `PromptEncoder` + `modality_from_latent_state` 的 pipeline
- `TI2VidOneStagePipeline`
- `TI2VidTwoStagesPipeline` / `TI2VidTwoStagesHQPipeline`
- `A2VidPipelineTwoStage`
- `KeyframeInterpolationPipeline`
- `DistilledPipeline`
- `ICLoraPipeline`
- `RetakePipeline`

**症状**：构造的 `Modality` 对象 `context_positions=None`（在 `utils/helpers.py:267` `modality_from_latent_state`），导致 `transformer_args.py:195` 的判断 `if modality.context_positions is not None` 走 false 分支，**完全跳过 RoTE K-side rotation**。

text cross-attention 退化成无位置编码的内容匹配。

**与训练的 mismatch**：LoRA 训练时（`packages/ltx-trainer/`）context_positions 是经过 `annotate_and_pack_for_rote_v2` 算出来的真实 (t_s, t_e) 张量；推理时 ltx-pipelines 给的是 None。两者分布完全不同。

**用户期望**：production inference 应该用 RoTE，跟训练分布对齐。

**修复路径**：
1. `PromptEncoder.__call__` 接受 / 返回 timing_map（或直接在 PromptEncoder 内部 strip）
2. `modality_from_latent_state` 增加 `context_positions` 参数
3. 每个 pipeline 在拼接 prompts → context 后，需要：
   - `stripped, timing = strip_time_ranges(prompt)`
   - `seq_len = pos_mask.shape[1]`
   - `ctx_pos = annotate_and_pack_for_rote_v2(stripped, tokenizer, seq_len, timing).unsqueeze(0)`
   - 把 `ctx_pos` 传给 `modality_from_latent_state`
4. 负向 prompt 用 `make_no_rope_positions(seq_len)`

## 2. Production pipelines 没 strip time_range（次要）

**影响范围**：同上，所有 production pipeline 调 `PromptEncoder.__call__`

**症状**：`utils/blocks.py:376` 里 `raw_outputs = [text_encoder.encode(p) for p in prompts]` 直接喂原 prompt 给 Gemma，prompt 里的 `"time_range": [...]` 字符串原样进 tokenizer，产生 ~8 token/section 的无用内容。

**与训练的 mismatch**：训练侧 `process_captions.py` 走的是 `strip_time_ranges(prompt)` 后再 encode，stripped prompt 没有 `"time_range":` 字符串。embedding 来自的 token 序列分布不一致。

**修复路径**：在调 `text_encoder.encode(p)` 之前先 `stripped, _ = strip_time_ranges(p)`。注意要和 issue #1 的修复一起做，timing_map 不能丢。

## 3. 修复优先级

如果你只是要用 `scripts/inference.py` (`ltx-trainer/validation_sampler.py`)，**当前不受这两个 issue 影响**——trainer 那边的链路已修。

如果切到 ltx-pipelines 的任何 production pipeline，必须同时修 #1 + #2 才能保证训练/推理分布对齐。

## 历史

- 这两个 issue 在 `feat/strip-time-range` 分支引入 `strip_time_ranges` + `annotate_and_pack_for_rote_v2` 时浮现
- 当时优先修了 ltx-trainer 侧（用户实际使用路径）
- ltx-pipelines 侧的修复因影响面大（要改 8 个 pipeline 的调用契约）暂缓
