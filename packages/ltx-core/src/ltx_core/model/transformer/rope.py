import functools
import math
from enum import Enum
from typing import Callable, Tuple

import numpy as np
import torch
from einops import rearrange


class LTXRopeType(Enum):
    INTERLEAVED = "interleaved"
    SPLIT = "split"


def apply_rotary_emb(
    input_tensor: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
) -> torch.Tensor:
    if rope_type == LTXRopeType.INTERLEAVED:
        return apply_interleaved_rotary_emb(input_tensor, *freqs_cis)
    elif rope_type == LTXRopeType.SPLIT:
        return apply_split_rotary_emb(input_tensor, *freqs_cis)
    else:
        raise ValueError(f"Invalid rope type: {rope_type}")


def apply_interleaved_rotary_emb(
    input_tensor: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor
) -> torch.Tensor:
    t_dup = rearrange(input_tensor, "... (d r) -> ... d r", r=2)
    t1, t2 = t_dup.unbind(dim=-1)
    t_dup = torch.stack((-t2, t1), dim=-1)
    input_tensor_rot = rearrange(t_dup, "... d r -> ... (d r)")

    out = input_tensor * cos_freqs + input_tensor_rot * sin_freqs

    return out


def apply_split_rotary_emb(
    input_tensor: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor
) -> torch.Tensor:
    needs_reshape = False
    if input_tensor.ndim != 4 and cos_freqs.ndim == 4:
        b, h, t, _ = cos_freqs.shape
        input_tensor = input_tensor.reshape(b, t, h, -1).swapaxes(1, 2)
        needs_reshape = True

    split_input = rearrange(input_tensor, "... (d r) -> ... d r", d=2)
    first_half_input = split_input[..., :1, :]
    second_half_input = split_input[..., 1:, :]

    output = split_input * cos_freqs.unsqueeze(-2)
    first_half_output = output[..., :1, :]
    second_half_output = output[..., 1:, :]

    first_half_output.addcmul_(-sin_freqs.unsqueeze(-2), second_half_input)
    second_half_output.addcmul_(sin_freqs.unsqueeze(-2), first_half_input)

    output = rearrange(output, "... d r -> ... (d r)")
    if needs_reshape:
        output = output.swapaxes(1, 2).reshape(b, t, -1)

    return output


@functools.lru_cache(maxsize=5)
def generate_freq_grid_np(
    positional_embedding_theta: float, positional_embedding_max_pos_count: int, inner_dim: int
) -> torch.Tensor:
    theta = positional_embedding_theta
    start = 1
    end = theta

    n_elem = 2 * positional_embedding_max_pos_count
    pow_indices = np.power(
        theta,
        np.linspace(
            np.log(start) / np.log(theta),
            np.log(end) / np.log(theta),
            inner_dim // n_elem,
            dtype=np.float64,
        ),
    )
    return torch.tensor(pow_indices * math.pi / 2, dtype=torch.float32)


@functools.lru_cache(maxsize=5)
def generate_freq_grid_pytorch(
    positional_embedding_theta: float, positional_embedding_max_pos_count: int, inner_dim: int
) -> torch.Tensor:
    theta = positional_embedding_theta
    start = 1
    end = theta
    n_elem = 2 * positional_embedding_max_pos_count

    indices = theta ** (
        torch.linspace(
            math.log(start, theta),
            math.log(end, theta),
            inner_dim // n_elem,
            dtype=torch.float32,
        )
    )
    indices = indices.to(dtype=torch.float32)

    indices = indices * math.pi / 2

    return indices


def get_fractional_positions(indices_grid: torch.Tensor, max_pos: list[int]) -> torch.Tensor:
    n_pos_dims = indices_grid.shape[1]
    assert n_pos_dims == len(max_pos), (
        f"Number of position dimensions ({n_pos_dims}) must match max_pos length ({len(max_pos)})"
    )
    fractional_positions = torch.stack(
        [indices_grid[:, i] / max_pos[i] for i in range(n_pos_dims)],
        dim=-1,
    )
    return fractional_positions


def generate_freqs(
    indices: torch.Tensor, indices_grid: torch.Tensor, max_pos: list[int], use_middle_indices_grid: bool
) -> torch.Tensor:
    if use_middle_indices_grid:
        assert len(indices_grid.shape) == 4
        assert indices_grid.shape[-1] == 2
        indices_grid_start, indices_grid_end = indices_grid[..., 0], indices_grid[..., 1]
        indices_grid = (indices_grid_start + indices_grid_end) / 2.0
    elif len(indices_grid.shape) == 4:
        indices_grid = indices_grid[..., 0]

    fractional_positions = get_fractional_positions(indices_grid, max_pos)
    indices = indices.to(device=fractional_positions.device)

    freqs = (indices * (fractional_positions.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)
    return freqs


def split_freqs_cis(freqs: torch.Tensor, pad_size: int, num_attention_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos_freq = freqs.cos()
    sin_freq = freqs.sin()

    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])

        cos_freq = torch.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = torch.concatenate([sin_padding, sin_freq], axis=-1)

    # Reshape freqs to be compatible with multi-head attention
    b = cos_freq.shape[0]
    t = cos_freq.shape[1]

    cos_freq = cos_freq.reshape(b, t, num_attention_heads, -1)
    sin_freq = sin_freq.reshape(b, t, num_attention_heads, -1)

    cos_freq = torch.swapaxes(cos_freq, 1, 2)  # (B,H,T,D//2)
    sin_freq = torch.swapaxes(sin_freq, 1, 2)  # (B,H,T,D//2)
    return cos_freq, sin_freq


def interleaved_freqs_cis(freqs: torch.Tensor, pad_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos_freq = freqs.cos().repeat_interleave(2, dim=-1)
    sin_freq = freqs.sin().repeat_interleave(2, dim=-1)
    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(cos_freq[:, :, :pad_size])
        cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
        sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)
    return cos_freq, sin_freq


def compute_tcr_bias(
    q_time: torch.Tensor,
    k_intervals: torch.Tensor,
    beta: float = 5.0,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Temporal Context Routing (TCR) bias for text cross-attention.

    Implements the paper operator (Eq. 1), with an optional width multiplier
    ``alpha`` that the paper fixes at 1:

        B_{ij} = -β α (t_i - c_j)^2 / (2 r_j^2)

    where ``c_j`` is the interval center and ``r_j = max((e_j - s_j) / 2, ε)``
    with ``ε = 1e-4`` seconds. Sentinel tokens (``t_s < 0`` or ``t_e < 0``)
    receive ``B = 0`` and are not routed. The term is added to pre-softmax
    text cross-attention logits; semantic scores are left intact.

    Args:
        q_time: (B, T_q) query wall-clock times in seconds.
        k_intervals: (B, T_k, 2) key intervals ``(t_s, t_e)`` in seconds.
        beta: overall scale β. Paper default is 5.0 (endpoint retention −β/2).
        alpha: optional width multiplier. Paper default is 1.0.

    Returns:
        (B, 1, T_q, T_k) TCR attention bias.
    """
    k_ts = k_intervals[..., 0]
    k_te = k_intervals[..., 1]
    sentinel_mask = (k_ts < 0) | (k_te < 0)

    center = (k_ts + k_te) * 0.5
    radius = ((k_te - k_ts) * 0.5).clamp(min=1e-4)

    dist = q_time.unsqueeze(-1) - center.unsqueeze(-2)
    bias = -beta * alpha * dist.pow(2) / (2.0 * radius.unsqueeze(-2).pow(2))

    if sentinel_mask.any():
        bias = bias.masked_fill(sentinel_mask.unsqueeze(-2), 0.0)

    return bias.unsqueeze(1)


def precompute_rote_freqs_cis(
    q_time: torch.Tensor,
    k_intervals: torch.Tensor,
    dim: int,
    out_dtype: torch.dtype,
    time_max_pos: float,
    theta: float = 10000.0,
    gamma: float = 1.0,
    alpha: float = 1.0,
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
    num_attention_heads: int = 1,
    freq_grid_generator: Callable[[float, int, int], torch.Tensor] = generate_freq_grid_pytorch,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """RoTE for text cross-attention, time-aligned with V↔A cross-attn.

    Q-side reuses LTX's standard 1D point RoPE: same ``indices`` grid
    (``θ^linspace · π/2`` from :func:`generate_freq_grid_pytorch`) and the same
    ``pos / time_max_pos · 2 − 1`` mapping into ``[-1, 1]`` that
    :func:`precompute_freqs_cis` uses for the V↔A cross-attn time axis. So a
    video/audio token at the same wall-clock second produces the same
    Q-rotation in V↔A and in V/A↔Text — text events become a third "axis"
    on the same time scale.

    K-side (text events) shares the indices grid + normalization, then layers
    the TIE [t_s, t_e] interval with sinc-modulated decay (Appendix A.2):

        t̂_s, t̂_e := normalize(t_s), normalize(t_e)        # in [-1, 1]
        center    = (t̂_s + t̂_e) / 2
        radius    = (t̂_e − t̂_s) / 2
        θ_c       = indices · center                        # (B, T_k, half)
        φ         = indices · radius
        sincs     = sinc(α·φ/π) / mean(sincs)               # Duration Invariance
        K cos/sin = (θ_c.cos, θ_c.sin) · sincs
        Q cos/sin = (indices · q̂_time).cos / sin

    Note on degenerate cases: when ``radius → 0`` (point-interval text token,
    e.g. a single timestamp), ``sincs → 1`` and K collapses to a standard
    point RoPE at ``center``. Because Q and K share the same ``indices``, K
    at center then matches a Q at the same normalized time exactly — TIE's
    symmetric dot-product property holds.

    Args:
        q_time: (B, T_q) Q time positions in seconds.
        k_intervals: (B, T_k, 2) K (t_s, t_e) in seconds. Tokens with t_s < 0
            or t_e < 0 are no-RoPE (identity rotation, used for padding,
            JSON-structural tokens, and CFG negative prompts).
        dim: full inner_dim of the attention's Q/K projection.
        out_dtype: output dtype.
        time_max_pos: time-axis normalizer (seconds). Should match the
            ``cross_pe_max_pos`` (V↔A) or ``positional_embedding_max_pos[0]``
            (single-modality) the model uses for self-/cross-attn time PE,
            so all attention layers interpret a given second identically.
        theta: RoPE base frequency (default 10000, matches LTX self-attn).
        gamma: multiplicative scale on the LTX freq grid (default 1.0 → Q-side
            is *literally* identical to V↔A's time RoPE). Bumping γ above 1
            stretches the freq spectrum and packs more cycles into the clip.
        alpha: sinc bandwidth multiplier (TIE default 1.0). Controls how
            quickly K decays away from the interval center.
        rope_type: INTERLEAVED or SPLIT — must match the attention layer.
        num_attention_heads: needed only for SPLIT layout.
        freq_grid_generator: generator for the ``indices`` grid; defaults to
            the pytorch variant. Pass :func:`generate_freq_grid_np` for
            ``double_precision_rope`` parity with self-attn.

    Returns:
        ((q_cos, q_sin), (k_cos, k_sin)) compatible with apply_rotary_emb.
    """
    device = q_time.device
    half = dim // 2

    # LTX standard freq grid (n_pos_dims=1 → returns (half,)). Same grid used by
    # precompute_freqs_cis for V↔A cross-attn time axis. γ scales the spectrum;
    # at γ=1.0 Q rotation equals V↔A's time rotation channel-for-channel.
    indices = freq_grid_generator(theta, 1, dim).to(device=device, dtype=torch.float32)
    if gamma != 1.0:
        indices = indices * gamma

    inv_max = 2.0 / float(time_max_pos)  # `pos · inv_max − 1` ≡ `pos / time_max_pos · 2 − 1`

    # ----- Q side: point RoPE at normalized time (matches V↔A cross-attn) -----
    q_norm = q_time * inv_max - 1.0  # (B, T_q), in [-1, 1] when 0 ≤ q_time ≤ time_max_pos
    q_phase = torch.einsum("bt,d->btd", q_norm, indices)  # (B, T_q, half)
    q_cos = q_phase.cos()
    q_sin = q_phase.sin()

    # ----- K side: interval RoTE on normalized [t_s, t_e] -----
    k_ts = k_intervals[..., 0]
    k_te = k_intervals[..., 1]
    no_rope_mask = (k_ts < 0) | (k_te < 0)  # (B, T_k)

    t_s_norm = k_ts * inv_max - 1.0
    t_e_norm = k_te * inv_max - 1.0
    center = (t_s_norm + t_e_norm) * 0.5
    radius = (t_e_norm - t_s_norm) * 0.5

    theta_c = torch.einsum("bt,d->btd", center, indices)  # (B, T_k, half)
    phi = torch.einsum("bt,d->btd", radius, indices)
    # torch.sinc(x) = sin(πx)/(πx); we want sin(α·φ)/(α·φ) → arg α·φ/π.
    sincs = torch.sinc(alpha * phi / math.pi)
    # Duration Invariance — divide by mean. Sign-preserving guard against
    # vanishing means (point-interval edge case is handled by the no-rope
    # check below for true sentinels; legitimate radius=0 yields mean=1, safe).
    sincs_mean = sincs.mean(dim=-1, keepdim=True)
    safe_mean = sincs_mean.sign().clamp(min=1.0) * sincs_mean.abs().clamp(min=1e-8)
    sincs = sincs / safe_mean

    k_cos = theta_c.cos() * sincs
    k_sin = theta_c.sin() * sincs

    if no_rope_mask.any():
        m = no_rope_mask[:, :, None]
        k_cos = torch.where(m, torch.ones_like(k_cos), k_cos)
        k_sin = torch.where(m, torch.zeros_like(k_sin), k_sin)

    # ----- Layout match with apply_rotary_emb -----
    if rope_type == LTXRopeType.INTERLEAVED:
        # (B, T, half) → (B, T, dim): pairs (2l, 2l+1) share freq l, matching
        # apply_interleaved_rotary_emb's "(d r)" reshape.
        q_cos = q_cos.repeat_interleave(2, dim=-1)
        q_sin = q_sin.repeat_interleave(2, dim=-1)
        k_cos = k_cos.repeat_interleave(2, dim=-1)
        k_sin = k_sin.repeat_interleave(2, dim=-1)
    elif rope_type == LTXRopeType.SPLIT:
        if half % num_attention_heads != 0:
            raise ValueError(
                f"RoTE SPLIT requires inner_dim/2 ({half}) divisible by "
                f"num_attention_heads ({num_attention_heads})."
            )
        head_half = half // num_attention_heads

        def _to_split(x: torch.Tensor) -> torch.Tensor:
            b, t = x.shape[:2]
            return x.reshape(b, t, num_attention_heads, head_half).swapaxes(1, 2)

        q_cos, q_sin = _to_split(q_cos), _to_split(q_sin)
        k_cos, k_sin = _to_split(k_cos), _to_split(k_sin)
    else:
        raise NotImplementedError(f"RoTE: unsupported rope_type {rope_type}")

    return (
        (q_cos.to(out_dtype), q_sin.to(out_dtype)),
        (k_cos.to(out_dtype), k_sin.to(out_dtype)),
    )


def precompute_freqs_cis(
    indices_grid: torch.Tensor,
    dim: int,
    out_dtype: torch.dtype,
    theta: float = 10000.0,
    max_pos: list[int] | None = None,
    use_middle_indices_grid: bool = False,
    num_attention_heads: int = 32,
    rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
    freq_grid_generator: Callable[[float, int, int, torch.device], torch.Tensor] = generate_freq_grid_pytorch,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_pos is None:
        max_pos = [20, 2048, 2048]

    indices = freq_grid_generator(theta, indices_grid.shape[1], dim)
    freqs = generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid)

    if rope_type == LTXRopeType.SPLIT:
        expected_freqs = dim // 2
        current_freqs = freqs.shape[-1]
        pad_size = expected_freqs - current_freqs
        cos_freq, sin_freq = split_freqs_cis(freqs, pad_size, num_attention_heads)
    else:
        # 2 because of cos and sin by 3 for (t, x, y), 1 for temporal only
        n_elem = 2 * indices_grid.shape[1]
        cos_freq, sin_freq = interleaved_freqs_cis(freqs, dim % n_elem)
    return cos_freq.to(out_dtype), sin_freq.to(out_dtype)
