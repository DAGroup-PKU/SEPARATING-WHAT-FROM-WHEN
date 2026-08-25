from dataclasses import dataclass, replace

import torch

from ltx_core.model.transformer.adaln import AdaLayerNormSingle
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import (
    LTXRopeType,
    compute_tcr_bias,
    generate_freq_grid_np,
    generate_freq_grid_pytorch,
    precompute_freqs_cis,
    precompute_rote_freqs_cis,
)


@dataclass(frozen=True)
class TransformerArgs:
    x: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor
    timesteps: torch.Tensor
    embedded_timestep: torch.Tensor
    positional_embeddings: torch.Tensor
    cross_positional_embeddings: torch.Tensor | None
    cross_scale_shift_timestep: torch.Tensor | None
    cross_gate_timestep: torch.Tensor | None
    enabled: bool
    prompt_timestep: torch.Tensor | None = None
    self_attention_mask: torch.Tensor | None = (
        None  # Additive log-space self-attention bias (B, 1, T, T), None = full attention
    )
    text_cross_pe: tuple[torch.Tensor, torch.Tensor] | None = None
    text_cross_k_pe: tuple[torch.Tensor, torch.Tensor] | None = None
    text_cross_temporal_bias: torch.Tensor | None = None


class TransformerArgsPreprocessor:
    def __init__(  # noqa: PLR0913
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        use_middle_indices_grid: bool,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
        caption_projection: torch.nn.Module | None = None,
        prompt_adaln: AdaLayerNormSingle | None = None,
        rote_gamma: float = 1.0,
        rote_alpha: float = 1.0,
        time_max_pos: float | None = None,
        temporal_mode: str = "tcr",
        tcr_beta: float = 5.0,
        tcr_alpha: float = 1.0,
    ) -> None:
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.num_attention_heads = num_attention_heads
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.double_precision_rope = double_precision_rope
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type
        self.caption_projection = caption_projection
        self.prompt_adaln = prompt_adaln
        # Temporal operator for text cross-attention.
        #   temporal_mode="tcr": Temporal Context Routing (paper default).
        #   temporal_mode="rotary": RoTE baseline (Q point RoPE + K sinc interval).
        # rote_gamma / rote_alpha only apply to the rotary baseline.
        self.rote_gamma = rote_gamma
        self.rote_alpha = rote_alpha
        if temporal_mode in ("additive", "tab"):
            temporal_mode = "tcr"
        self.temporal_mode = temporal_mode
        self.tcr_beta = tcr_beta
        self.tcr_alpha = tcr_alpha
        # Time-axis normalizer in seconds. Multimodal models override this
        # with cross_pe_max_pos so V→T and A→T share the same time scale.
        # Single-modality falls back to max_pos[0].
        self.time_max_pos = float(time_max_pos) if time_max_pos is not None else float(max_pos[0])

    def _prepare_timestep(
        self, timestep: torch.Tensor, adaln: AdaLayerNormSingle, batch_size: int, hidden_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare timestep embeddings."""
        timestep_scaled = timestep * self.timestep_scale_multiplier
        timestep, embedded_timestep = adaln(
            timestep_scaled.flatten(),
            hidden_dtype=hidden_dtype,
        )
        # Second dimension is 1 or number of tokens (if timestep_per_token)
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])

        return timestep, embedded_timestep

    def _prepare_context(
        self,
        context: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Prepare context for transformer blocks."""
        if self.caption_projection is not None:
            context = self.caption_projection(context)
        batch_size = x.shape[0]
        return context.view(batch_size, -1, x.shape[-1])

    def _prepare_attention_mask(self, attention_mask: torch.Tensor | None, x_dtype: torch.dtype) -> torch.Tensor | None:
        """Prepare attention mask."""
        if attention_mask is None or torch.is_floating_point(attention_mask):
            return attention_mask

        return (attention_mask - 1).to(x_dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(x_dtype).max

    def _prepare_self_attention_mask(
        self, attention_mask: torch.Tensor | None, x_dtype: torch.dtype
    ) -> torch.Tensor | None:
        """Prepare self-attention mask by converting [0,1] values to additive log-space bias.
        Input shape: (B, T, T) with values in [0, 1].
        Output shape: (B, 1, T, T) with 0.0 for full attention and a large negative value
        for masked positions.
        Positions with attention_mask <= 0 are fully masked (mapped to the dtype's minimum
        representable value). Strictly positive entries are converted via log-space for
        smooth attenuation, with small values clamped for numerical stability.
        Returns None if input is None (no masking).
        """
        if attention_mask is None:
            return None

        # Convert [0, 1] attention mask to additive log-space bias:
        #   1.0 -> log(1.0) = 0.0  (no bias, full attention)
        #   0.0 -> finfo.min        (fully masked)
        finfo = torch.finfo(x_dtype)
        eps = finfo.tiny

        bias = torch.full_like(attention_mask, finfo.min, dtype=x_dtype)
        positive = attention_mask > 0
        if positive.any():
            bias[positive] = torch.log(attention_mask[positive].clamp(min=eps)).to(x_dtype)

        return bias.unsqueeze(1)  # (B, 1, T, T) for head broadcast

    def _prepare_positional_embeddings(
        self,
        positions: torch.Tensor,
        inner_dim: int,
        max_pos: list[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
        x_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare positional embeddings."""
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        pe = precompute_freqs_cis(
            positions,
            dim=inner_dim,
            out_dtype=x_dtype,
            theta=self.positional_embedding_theta,
            max_pos=max_pos,
            use_middle_indices_grid=use_middle_indices_grid,
            num_attention_heads=num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
        )
        return pe

    def prepare(
        self,
        modality: Modality,
        cross_modality: Modality | None = None,  # noqa: ARG002
    ) -> TransformerArgs:
        x = self.patchify_proj(modality.latent)
        batch_size = x.shape[0]
        timestep, embedded_timestep = self._prepare_timestep(
            modality.timesteps, self.adaln, batch_size, modality.latent.dtype
        )
        prompt_timestep = None
        if self.prompt_adaln is not None:
            prompt_timestep, _ = self._prepare_timestep(
                modality.sigma, self.prompt_adaln, batch_size, modality.latent.dtype
            )
        context = self._prepare_context(modality.context, x)
        attention_mask = self._prepare_attention_mask(modality.context_mask, modality.latent.dtype)
        pe = self._prepare_positional_embeddings(
            positions=modality.positions,
            inner_dim=self.inner_dim,
            max_pos=self.max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
            x_dtype=modality.latent.dtype,
        )
        self_attention_mask = self._prepare_self_attention_mask(modality.attention_mask, modality.latent.dtype)

        text_cross_pe = None
        text_cross_k_pe = None
        text_cross_temporal_bias = None
        if modality.context_positions is not None:
            if self.temporal_mode == "tcr":
                text_cross_temporal_bias = self._prepare_tcr_bias(modality, x.dtype)
            else:
                text_cross_pe, text_cross_k_pe = self._prepare_text_cross_rote(
                    modality, x.dtype
                )

        return TransformerArgs(
            x=x,
            context=context,
            context_mask=attention_mask,
            timesteps=timestep,
            embedded_timestep=embedded_timestep,
            positional_embeddings=pe,
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=modality.enabled,
            prompt_timestep=prompt_timestep,
            self_attention_mask=self_attention_mask,
            text_cross_pe=text_cross_pe,
            text_cross_k_pe=text_cross_k_pe,
            text_cross_temporal_bias=text_cross_temporal_bias,
        )


    def _prepare_text_cross_rote(
        self,
        modality: Modality,
        x_dtype: torch.dtype,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Build Q-side RoPE and K-side RoTE for text cross-attention.

        Single-axis 1D rotation on all inner_dim channels — cross-attn between
        video/audio and text has no spatial (h,w) semantics, so the rotation
        only depends on time. Both sides normalize seconds to [-1, 1] via
        ``time_max_pos`` (= ``cross_pe_max_pos`` in multimodal models),
        matching V<->A cross-attn's time axis. Q sees a point RoPE at
        ``q_time_norm``; K sees a sinc-modulated rotation around the
        ``[t_s_norm, t_e_norm]`` interval midpoint.

        All time inputs are in seconds:
          - Video Q: ``modality.positions[:, 0, :]`` is seconds
            (``base_strategy._get_video_positions`` divides by fps).
          - Audio Q: ``patchifier._compute_audio_timings`` returns seconds
            via ``hop_length / sample_rate``.
          - K (text): ``context_positions`` comes from ``time_range_annotator``
            (MTSS JSON ``time_range`` fields, in seconds).
        """
        positions = modality.positions
        # positions shape: (B, n_axes, T, 2) when use_middle_indices_grid, else (B, n_axes, T)
        if positions.ndim == 4:
            t_start = positions[:, 0, :, 0]
            t_end = positions[:, 0, :, 1]
            q_time = (t_start + t_end) / 2.0
        else:
            q_time = positions[:, 0, :]

        k_intervals = modality.context_positions  # (B, T_k, 2) in seconds

        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        return precompute_rote_freqs_cis(
            q_time=q_time,
            k_intervals=k_intervals,
            dim=self.inner_dim,
            out_dtype=x_dtype,
            time_max_pos=self.time_max_pos,
            theta=self.positional_embedding_theta,
            gamma=self.rote_gamma,
            alpha=self.rote_alpha,
            rope_type=self.rope_type,
            num_attention_heads=self.num_attention_heads,
            freq_grid_generator=freq_grid_generator,
        )

    def _prepare_tcr_bias(
        self,
        modality: Modality,
        x_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build Temporal Context Routing bias for text cross-attention."""
        positions = modality.positions
        if positions.ndim == 4:
            t_start = positions[:, 0, :, 0]
            t_end = positions[:, 0, :, 1]
            q_time = (t_start + t_end) / 2.0
        else:
            q_time = positions[:, 0, :]

        k_intervals = modality.context_positions
        return compute_tcr_bias(
            q_time=q_time,
            k_intervals=k_intervals,
            beta=self.tcr_beta,
            alpha=self.tcr_alpha,
        ).to(dtype=x_dtype)


class MultiModalTransformerArgsPreprocessor:
    def __init__(  # noqa: PLR0913
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        cross_pe_max_pos: int,
        use_middle_indices_grid: bool,
        audio_cross_attention_dim: int,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
        av_ca_timestep_scale_multiplier: int,
        caption_projection: torch.nn.Module | None = None,
        prompt_adaln: AdaLayerNormSingle | None = None,
        rote_gamma: float = 1.0,
        rote_alpha: float = 1.0,
        temporal_mode: str = "tcr",
        tcr_beta: float = 5.0,
        tcr_alpha: float = 1.0,
    ) -> None:
        self.simple_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=patchify_proj,
            adaln=adaln,
            inner_dim=inner_dim,
            max_pos=max_pos,
            num_attention_heads=num_attention_heads,
            use_middle_indices_grid=use_middle_indices_grid,
            timestep_scale_multiplier=timestep_scale_multiplier,
            double_precision_rope=double_precision_rope,
            positional_embedding_theta=positional_embedding_theta,
            rope_type=rope_type,
            caption_projection=caption_projection,
            prompt_adaln=prompt_adaln,
            rote_gamma=rote_gamma,
            rote_alpha=rote_alpha,
            temporal_mode=temporal_mode,
            tcr_beta=tcr_beta,
            tcr_alpha=tcr_alpha,
            # V↔A cross-attn normalizes time by cross_pe_max_pos = max(video, audio).
            # Reuse it so V/A→text TCR (and the rotary baseline) share the same
            # wall-clock second across self-attn / V↔A / V/A↔text.
            time_max_pos=cross_pe_max_pos,
        )
        self.cross_scale_shift_adaln = cross_scale_shift_adaln
        self.cross_gate_adaln = cross_gate_adaln
        self.cross_pe_max_pos = cross_pe_max_pos
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier

    def prepare(
        self,
        modality: Modality,
        cross_modality: Modality | None = None,
    ) -> TransformerArgs:
        transformer_args = self.simple_preprocessor.prepare(modality)
        if cross_modality is None:
            return transformer_args

        if cross_modality.sigma.numel() > 1:
            if cross_modality.sigma.shape[0] != modality.timesteps.shape[0]:
                raise ValueError("Cross modality sigma must have the same batch size as the modality")
            if cross_modality.sigma.ndim != 1:
                raise ValueError("Cross modality sigma must be a 1D tensor")

        cross_timestep = cross_modality.sigma.view(
            modality.timesteps.shape[0], 1, *[1] * len(modality.timesteps.shape[2:])
        )

        cross_pe = self.simple_preprocessor._prepare_positional_embeddings(
            positions=modality.positions[:, 0:1, :],
            inner_dim=self.audio_cross_attention_dim,
            max_pos=[self.cross_pe_max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=self.simple_preprocessor.num_attention_heads,
            x_dtype=modality.latent.dtype,
        )

        cross_scale_shift_timestep, cross_gate_timestep = self._prepare_cross_attention_timestep(
            timestep=cross_timestep,
            timestep_scale_multiplier=self.simple_preprocessor.timestep_scale_multiplier,
            batch_size=transformer_args.x.shape[0],
            hidden_dtype=modality.latent.dtype,
        )

        return replace(
            transformer_args,
            cross_positional_embeddings=cross_pe,
            cross_scale_shift_timestep=cross_scale_shift_timestep,
            cross_gate_timestep=cross_gate_timestep,
        )

    def _prepare_cross_attention_timestep(
        self,
        timestep: torch.Tensor | None,
        timestep_scale_multiplier: int,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare cross attention timestep embeddings."""
        timestep = timestep * timestep_scale_multiplier

        av_ca_factor = self.av_ca_timestep_scale_multiplier / timestep_scale_multiplier

        scale_shift_timestep, _ = self.cross_scale_shift_adaln(
            timestep.flatten(),
            hidden_dtype=hidden_dtype,
        )
        scale_shift_timestep = scale_shift_timestep.view(batch_size, -1, scale_shift_timestep.shape[-1])
        gate_noise_timestep, _ = self.cross_gate_adaln(
            timestep.flatten() * av_ca_factor,
            hidden_dtype=hidden_dtype,
        )
        gate_noise_timestep = gate_noise_timestep.view(batch_size, -1, gate_noise_timestep.shape[-1])

        return scale_shift_timestep, gate_noise_timestep
