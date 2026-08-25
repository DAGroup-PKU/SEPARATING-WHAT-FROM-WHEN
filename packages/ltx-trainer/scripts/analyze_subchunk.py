"""
Sub-chunk temporal precision analysis.

Evidence 1: VAE reconstruction preserves intra-chunk shot transitions.
Evidence 2: Gaussian bias produces distinct attention distributions for
            different sub-chunk annotation positions, while attention masks
            produce identical distributions.

Usage:
    # Evidence 1: VAE reconstruction (needs GPU + model)
    python scripts/analyze_subchunk.py vae-recon \
        --video-path /path/to/video_with_shot_cut.mp4 \
        --model-path /path/to/ltx-2.3.safetensors \
        --cut-time 2.1

    # Evidence 2: Bias sensitivity (no GPU needed)
    python scripts/analyze_subchunk.py bias-sensitivity \
        --output-dir ./analysis_output
"""
import json
import math
from pathlib import Path

import torch
import typer

app = typer.Typer()


def gaussian_bias(q_times: torch.Tensor, center: float, radius: float,
                  scale: float = 5.0, alpha: float = 1.0) -> torch.Tensor:
    """Compute Gaussian temporal bias for a single event."""
    dist = q_times - center
    return -scale * alpha * dist.pow(2) / (2.0 * radius ** 2)


@app.command()
def bias_sensitivity(
    output_dir: str = typer.Option("./analysis_output", help="Output directory"),
    fps: int = typer.Option(24, help="Video FPS"),
    temporal_compression: int = typer.Option(8, help="VAE temporal compression"),
    clip_duration: float = typer.Option(8.0, help="Clip duration in seconds"),
    scale: float = typer.Option(5.0, help="Gaussian bias scale"),
) -> None:
    """Evidence 2: Show Gaussian bias is sensitive to sub-chunk annotation changes."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    chunk_duration = temporal_compression / fps
    n_chunks = int(clip_duration / chunk_duration)
    q_times = torch.tensor([(i + 0.5) * chunk_duration for i in range(n_chunks)])

    print(f"Video: {clip_duration}s, {fps}fps, {temporal_compression}x compression")
    print(f"Chunks: {n_chunks}, each {chunk_duration:.3f}s")
    print(f"Chunk centers: {[f'{t:.3f}' for t in q_times.tolist()]}")
    print()

    boundary_chunk_idx = 6
    chunk_start = boundary_chunk_idx * chunk_duration
    chunk_end = (boundary_chunk_idx + 1) * chunk_duration
    print(f"=== Focus chunk {boundary_chunk_idx}: [{chunk_start:.3f}s, {chunk_end:.3f}s], center={q_times[boundary_chunk_idx]:.3f}s ===")
    print()

    test_boundaries = [
        chunk_start,
        chunk_start + chunk_duration * 0.25,
        chunk_start + chunk_duration * 0.5,
        chunk_start + chunk_duration * 0.75,
        chunk_end,
    ]

    results = {"chunk_info": {
        "idx": boundary_chunk_idx,
        "start": chunk_start,
        "end": chunk_end,
        "center": q_times[boundary_chunk_idx].item(),
        "duration": chunk_duration,
    }, "comparisons": []}

    print(f"{'Boundary':>10} | {'Method':>12} | {'Shot A attn':>12} | {'Shot B attn':>12} | {'Ratio A:B':>10}")
    print("-" * 65)

    for boundary in test_boundaries:
        a_center = boundary / 2
        a_radius = max(boundary / 2, 1e-4)
        b_center = (boundary + clip_duration) / 2
        b_radius = (clip_duration - boundary) / 2

        t_q = q_times[boundary_chunk_idx]

        bias_a = gaussian_bias(t_q.unsqueeze(0), a_center, a_radius, scale=scale).item()
        bias_b = gaussian_bias(t_q.unsqueeze(0), b_center, b_radius, scale=scale).item()

        content = 3.0
        score_a = content + bias_a
        score_b = content + bias_b

        max_s = max(score_a, score_b)
        exp_a = math.exp(score_a - max_s)
        exp_b = math.exp(score_b - max_s)
        total = exp_a + exp_b
        gauss_attn_a = exp_a / total
        gauss_attn_b = exp_b / total

        chunk_center = t_q.item()
        if chunk_center < boundary:
            mask_attn_a, mask_attn_b = 1.0, 0.0
        else:
            mask_attn_a, mask_attn_b = 0.0, 1.0

        print(f"{boundary:10.3f} | {'Gaussian':>12} | {gauss_attn_a:12.4f} | {gauss_attn_b:12.4f} | {gauss_attn_a:.2f}:{gauss_attn_b:.2f}")
        print(f"{'':>10} | {'Mask':>12} | {mask_attn_a:12.4f} | {mask_attn_b:12.4f} | {mask_attn_a:.0f}:{mask_attn_b:.0f}")
        print()

        results["comparisons"].append({
            "boundary": boundary,
            "gaussian": {"shot_a": gauss_attn_a, "shot_b": gauss_attn_b},
            "mask": {"shot_a": mask_attn_a, "shot_b": mask_attn_b},
        })

    gauss_variations = [r["gaussian"]["shot_a"] for r in results["comparisons"]]
    mask_variations = [r["mask"]["shot_a"] for r in results["comparisons"]]

    gauss_range = max(gauss_variations) - min(gauss_variations)
    mask_range = max(mask_variations) - min(mask_variations)

    print("=== Summary ===")
    print(f"Gaussian: Shot A attention ranges from {min(gauss_variations):.4f} to {max(gauss_variations):.4f} (range={gauss_range:.4f})")
    print(f"Mask:     Shot A attention ranges from {min(mask_variations):.4f} to {max(mask_variations):.4f} (range={mask_range:.4f})")
    print()
    if mask_range < gauss_range:
        print(f"→ Gaussian bias produces {len(set(f'{v:.4f}' for v in gauss_variations))} distinct distributions")
        print(f"→ Attention mask produces {len(set(f'{v:.4f}' for v in mask_variations))} distinct distributions")
        print("→ Gaussian bias is sensitive to sub-chunk boundary positions; mask is not.")

    results["summary"] = {"gaussian_range": gauss_range, "mask_range": mask_range}

    result_file = output_path / "subchunk_sensitivity.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_file}")

    # Full attention curves for plotting
    curve_file = output_path / "attention_curves.json"
    curves = {"q_times": q_times.tolist(), "boundaries": {}}

    for boundary in test_boundaries:
        a_center = boundary / 2
        a_radius = max(boundary / 2, 1e-4)
        b_center = (boundary + clip_duration) / 2
        b_radius = (clip_duration - boundary) / 2

        gauss_a_list, gauss_b_list, mask_a_list = [], [], []

        for t_q in q_times:
            bias_a = gaussian_bias(t_q.unsqueeze(0), a_center, a_radius, scale=scale).item()
            bias_b = gaussian_bias(t_q.unsqueeze(0), b_center, b_radius, scale=scale).item()

            content = 3.0
            score_a = content + bias_a
            score_b = content + bias_b
            max_s = max(score_a, score_b)
            exp_a = math.exp(score_a - max_s)
            exp_b = math.exp(score_b - max_s)
            total = exp_a + exp_b

            gauss_a_list.append(exp_a / total)
            gauss_b_list.append(exp_b / total)
            mask_a_list.append(1.0 if t_q.item() < boundary else 0.0)

        curves["boundaries"][f"{boundary:.3f}"] = {
            "gaussian_shot_a": gauss_a_list,
            "gaussian_shot_b": gauss_b_list,
            "mask_shot_a": mask_a_list,
        }

    with open(curve_file, "w") as f:
        json.dump(curves, f, indent=2)
    print(f"Curves saved to {curve_file}")


@app.command()
def vae_recon(
    video_path: str = typer.Option(..., help="Video with a known shot cut"),
    model_path: str = typer.Option(..., help="LTX-2 checkpoint path"),
    cut_time: float = typer.Option(..., help="Known shot cut time in seconds"),
    output_dir: str = typer.Option("./analysis_output", help="Output directory"),
    device: str = typer.Option("cuda", help="Device"),
) -> None:
    """Evidence 1: VAE encode->decode preserves intra-chunk shot transitions."""
    from ltx_trainer.model_loader import load_video_vae_encoder, load_video_vae_decoder
    from ltx_trainer.video_utils import read_video
    from ltx_trainer.utils import save_image

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading video: {video_path}")
    print(f"Known shot cut at: {cut_time}s")

    video, fps = read_video(video_path)
    print(f"Video shape: {video.shape}, fps: {fps}")

    cut_frame = int(cut_time * fps)
    temporal_compression = 8
    chunk_idx = cut_frame // temporal_compression
    chunk_start_frame = chunk_idx * temporal_compression
    chunk_end_frame = (chunk_idx + 1) * temporal_compression
    cut_position_in_chunk = (cut_frame - chunk_start_frame) / temporal_compression

    print(f"Cut at frame {cut_frame}")
    print(f"Falls in chunk {chunk_idx}: frames [{chunk_start_frame}, {chunk_end_frame})")
    print(f"Position within chunk: {cut_position_in_chunk:.2f} ({cut_position_in_chunk*100:.0f}%)")
    print()

    if cut_position_in_chunk <= 0 or cut_position_in_chunk >= 1:
        print("WARNING: Cut is at chunk boundary, not intra-chunk. Choose a different cut_time.")
        return

    print("Loading VAE encoder/decoder...")
    vae_encoder = load_video_vae_encoder(model_path, device=device)
    vae_decoder = load_video_vae_decoder(model_path, device=device)

    print("Encoding video...")
    video_tensor = video.unsqueeze(0).to(device)
    with torch.inference_mode():
        latent = vae_encoder(video_tensor)
        print(f"Latent shape: {latent.shape}")

        print("Decoding latent...")
        reconstructed = vae_decoder(latent)

    reconstructed = reconstructed.squeeze(0).cpu()

    for label, v in [("original", video), ("reconstructed", reconstructed)]:
        for offset in range(-2, 3):
            frame_idx = cut_frame + offset
            if 0 <= frame_idx < v.shape[0]:
                frame = v[frame_idx]
                frame_path = output_path / f"{label}_frame{frame_idx}_offset{offset:+d}.png"
                save_image(frame, str(frame_path))

    print("\nPer-frame reconstruction MSE around cut:")
    for offset in range(-3, 4):
        frame_idx = cut_frame + offset
        if 0 <= frame_idx < min(video.shape[0], reconstructed.shape[0]):
            mse = (video[frame_idx].float() - reconstructed[frame_idx].float()).pow(2).mean().item()
            marker = " <-- CUT" if offset == 0 else ""
            in_chunk = "▓" if chunk_start_frame <= frame_idx < chunk_end_frame else "░"
            print(f"  frame {frame_idx:3d} (offset {offset:+d}) {in_chunk} MSE={mse:.6f}{marker}")

    print("\nFrame-to-frame difference (should spike at cut):")
    for offset in range(-3, 4):
        frame_idx = cut_frame + offset
        if 1 <= frame_idx < reconstructed.shape[0]:
            diff = (reconstructed[frame_idx].float() - reconstructed[frame_idx-1].float()).pow(2).mean().item()
            marker = " <-- CUT" if offset == 0 else ""
            in_chunk = "▓" if chunk_start_frame <= frame_idx < chunk_end_frame else "░"
            print(f"  frame {frame_idx-1:3d}->{frame_idx:3d} {in_chunk} delta={diff:.6f}{marker}")

    print(f"\nFrames saved to {output_path}/")
    print("If frame-to-frame delta spikes at the cut frame, the VAE preserves intra-chunk transitions.")


if __name__ == "__main__":
    app()
