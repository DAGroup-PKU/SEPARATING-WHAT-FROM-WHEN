#!/usr/bin/env python3

"""
Compute latent representations for video generation training.
This module provides functionality for processing video and image files, including:
- Loading videos/images from various file formats (CSV, JSON, JSONL)
- Resizing, cropping, and transforming media
- MediaDataset for video-only preprocessing workflows
- BucketSampler for grouping videos by resolution
Can be used as a standalone script:
    python scripts/process_videos.py dataset.csv --resolution-buckets 768x768x25 \
        --output-dir /path/to/output --model-source /path/to/ltx2.safetensors
"""

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import typer
from pillow_heif import register_heif_opener
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import crop, resize, to_tensor
from transformers.utils.logging import disable_progress_bar

from ltx_core.model.audio_vae import AudioProcessor
from ltx_core.types import Audio
from ltx_trainer import logger
from ltx_trainer.model_loader import load_audio_vae_encoder, load_video_vae_encoder
from ltx_trainer.utils import open_image_as_srgb
from ltx_trainer.video_utils import get_video_frame_count, read_video

disable_progress_bar()

# Register HEIF/HEIC support
register_heif_opener()

# Constants for validation
VAE_SPATIAL_FACTOR = 32
VAE_TEMPORAL_FACTOR = 8

# Audio constants
AUDIO_LATENT_CHANNELS = 8
AUDIO_FREQUENCY_BINS = 16

DEFAULT_TILE_SIZE = 512  # Spatial tile size in pixels (must be ≥64 and divisible by 32)
DEFAULT_TILE_OVERLAP = 128  # Spatial tile overlap in pixels (must be divisible by 32)

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Process videos/images and save latent representations for video generation training.",
)


class MediaDataset(Dataset):
    """
    Dataset for processing video and image files.
    This dataset is designed for media preprocessing workflows where you need to:
    - Load and preprocess videos/images
    - Apply resizing and cropping transformations
    - Handle different resolution buckets
    - Filter out invalid media files
    - Optionally extract audio from video files
    """

    def __init__(
        self,
        dataset_file: str | Path,
        main_media_column: str,
        video_column: str,
        resolution_buckets: list[tuple[int, int, int]],
        reshape_mode: str = "center",
        with_audio: bool = False,
    ) -> None:
        """
        Initialize the media dataset.
        Args:
            dataset_file: Path to CSV/JSON/JSONL metadata file
            video_column: Column name for video paths in the metadata file
            resolution_buckets: List of (frames, height, width) tuples
            reshape_mode: How to crop videos ("center", "random")
            with_audio: Whether to extract audio from video files
        """
        super().__init__()

        self.dataset_file = Path(dataset_file)
        self.main_media_column = main_media_column
        self.resolution_buckets = resolution_buckets
        self.reshape_mode = reshape_mode
        self.with_audio = with_audio

        # First load main media paths
        self.main_media_paths = self._load_video_paths(main_media_column)

        # Then load reference video paths
        self.video_paths = self._load_video_paths(video_column)

        # Filter out videos with insufficient frames
        self._filter_valid_videos()

        # Temporal length is no longer capped by bucket frame count; clips are padded dynamically.
        self.max_target_frames = None

        # Set up video transforms.
        # NOTE: do NOT use a lambda here. With DataLoader(multiprocessing_context="forkserver"),
        # the dataset is pickled to each worker; lambdas defined in __init__ are not picklable
        # (AttributeError: Can't get local object '<locals>.<lambda>'). Use a module-level fn.
        self.transforms = transforms.Compose(
            [
                transforms.Lambda(_clamp01_inplace),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single video/image with metadata, and optionally audio."""
        if isinstance(index, list):
            # Special case for BucketSampler - return cached data
            return index

        video_path: Path = self.video_paths[index]

        # Compute relative path of the video
        data_root = self.dataset_file.parent
        relative_path = str(video_path.relative_to(data_root))
        media_relative_path = str(self.main_media_paths[index].relative_to(data_root))

        if video_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            media_tensor = self._preprocess_image(video_path)
            fps = 1.0
            audio_data = None  # Images don't have audio
        else:
            media_tensor, fps = self._preprocess_video(video_path)

            # Extract audio if enabled
            if self.with_audio:
                # Calculate target duration from the processed video frames
                # This ensures audio is trimmed to match the exact video duration
                # media_tensor is [C, F, H, W] so shape[1] is num_frames
                target_duration = media_tensor.shape[1] / fps
                audio_data = self._extract_audio(video_path, target_duration)
            else:
                audio_data = None

        # media_tensor is [C, F, H, W] format for VAE compatibility
        _, num_frames, height, width = media_tensor.shape

        result = {
            "video": media_tensor,
            "relative_path": relative_path,
            "main_media_relative_path": media_relative_path,
            "video_metadata": {
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "fps": fps,
            },
        }

        # Add audio data if available
        if audio_data is not None:
            result["audio"] = audio_data

        return result

    @staticmethod
    def _extract_audio(video_path: Path, target_duration: float) -> dict[str, torch.Tensor | int] | None:
        """Extract audio track from a video file, trimmed to match video duration.

        Uses ffmpeg subprocess (via imageio-ffmpeg) to extract audio as WAV,
        then loads the WAV data directly. This is the most portable approach
        since it doesn't depend on torchaudio backends or PyAV fork-safety.
        """
        try:
            waveform, sample_rate = MediaDataset._extract_audio_ffmpeg(video_path)
        except Exception as e:
            logger.warning(f"Could not extract audio from {video_path}: {e}")
            return None

        target_samples = int(target_duration * sample_rate)
        current_samples = waveform.shape[-1]

        if current_samples > target_samples:
            waveform = waveform[..., :target_samples]
        elif current_samples < target_samples:
            padding = target_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return {"waveform": waveform, "sample_rate": sample_rate}

    @staticmethod
    def _extract_audio_ffmpeg(video_path: Path, sample_rate: int = 44100) -> tuple[torch.Tensor, int]:
        """Extract audio from a video file using ffmpeg subprocess.

        Outputs raw PCM f32le to stdout, parsed directly into a tensor.
        Uses imageio-ffmpeg's bundled ffmpeg binary for portability.
        """

        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg_exe = "ffmpeg"

        cmd = [
            ffmpeg_exe,
            "-i", str(video_path),
            "-vn",                    # no video
            "-acodec", "pcm_f32le",   # 32-bit float little-endian
            "-ar", str(sample_rate),  # resample to target rate
            "-ac", "2",               # stereo
            "-f", "f32le",            # raw PCM output (no WAV header)
            "pipe:1",
        ]
        result = subprocess.run(
            cmd, capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[-500:]
            raise RuntimeError(f"ffmpeg exited with code {result.returncode}: {stderr}")

        raw = result.stdout
        if len(raw) == 0:
            raise ValueError("ffmpeg produced no audio output")

        audio_np = np.frombuffer(raw, dtype=np.float32)
        # Reshape to [channels=2, samples]
        audio_np = audio_np.reshape(-1, 2).T
        waveform = torch.from_numpy(audio_np.copy())
        return waveform, sample_rate

    def _load_video_paths(self, column: str) -> list[Path]:
        """Load video paths from the specified data source."""
        if self.dataset_file.suffix == ".csv":
            return self._load_video_paths_from_csv(column)
        elif self.dataset_file.suffix == ".json":
            return self._load_video_paths_from_json(column)
        elif self.dataset_file.suffix == ".jsonl":
            return self._load_video_paths_from_jsonl(column)
        else:
            raise ValueError("Expected `dataset_file` to be a path to a CSV, JSON, or JSONL file.")

    def _load_video_paths_from_csv(self, column: str) -> list[Path]:
        """Load video paths from a CSV file."""
        df = pd.read_csv(self.dataset_file)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in CSV file")

        data_root = self.dataset_file.parent
        video_paths = [data_root / Path(line.strip()) for line in df[column].tolist()]

        # Validate that all paths exist
        invalid_paths = [path for path in video_paths if not path.is_file()]
        if invalid_paths:
            raise ValueError(f"Found {len(invalid_paths)} invalid video paths. First few: {invalid_paths[:5]}")

        return video_paths

    def _load_video_paths_from_json(self, column: str) -> list[Path]:
        """Load video paths from a JSON file."""
        with open(self.dataset_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")

        data_root = self.dataset_file.parent
        video_paths = []
        for entry in data:
            if column not in entry:
                raise ValueError(f"Key '{column}' not found in JSON entry")
            video_paths.append(data_root / Path(entry[column].strip()))

        # Validate that all paths exist
        invalid_paths = [path for path in video_paths if not path.is_file()]
        if invalid_paths:
            raise ValueError(f"Found {len(invalid_paths)} invalid video paths. First few: {invalid_paths[:5]}")

        return video_paths

    def _load_video_paths_from_jsonl(self, column: str) -> list[Path]:
        """Load video paths from a JSONL file."""
        data_root = self.dataset_file.parent
        video_paths = []
        with open(self.dataset_file, "r", encoding="utf-8") as file:
            for line in file:
                entry = json.loads(line)
                if column not in entry:
                    raise ValueError(f"Key '{column}' not found in JSONL entry")
                video_paths.append(data_root / Path(entry[column].strip()))

        # Validate that all paths exist
        invalid_paths = [path for path in video_paths if not path.is_file()]
        if invalid_paths:
            raise ValueError(f"Found {len(invalid_paths)} invalid video paths. First few: {invalid_paths[:5]}")

        return video_paths

    def _filter_valid_videos(self) -> None:
        """Filter out videos with insufficient frames."""
        original_length = len(self.video_paths)
        valid_video_paths = []
        valid_main_media_paths = []
        # Minimum valid frame count is 1 (%8==1 after alignment); no bucket-based filter needed.
        min_frames_required = 1

        for i, video_path in enumerate(self.video_paths):
            if video_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                valid_video_paths.append(video_path)
                valid_main_media_paths.append(self.main_media_paths[i])
                continue

            try:
                frame_count = get_video_frame_count(video_path)

                if frame_count >= min_frames_required:
                    valid_video_paths.append(video_path)
                    valid_main_media_paths.append(self.main_media_paths[i])
                else:
                    logger.warning(
                        f"Skipping video at {video_path} - has {frame_count} frames, "
                        f"which is less than the minimum required frames ({min_frames_required})"
                    )
            except Exception as e:
                logger.warning(f"Failed to read video at {video_path}: {e!s}")

        # Update both path lists to maintain synchronization
        self.video_paths = valid_video_paths
        self.main_media_paths = valid_main_media_paths

        if len(self.video_paths) < original_length:
            logger.warning(
                f"Filtered out {original_length - len(self.video_paths)} videos with insufficient frames. "
                f"Proceeding with {len(self.video_paths)} valid videos."
            )

    def _preprocess_image(self, path: Path) -> torch.Tensor:
        """Preprocess a single image by resizing and applying transforms."""
        image = open_image_as_srgb(path)
        image = to_tensor(image)
        image = image.unsqueeze(0)  # Add frame dimension [1, C, H, W] for bucket selection

        # Find nearest resolution bucket and resize
        nearest_bucket = self._get_resolution_bucket_for_item(image)
        _, target_height, target_width = nearest_bucket
        image_resized = self._resize_and_crop(image, target_height, target_width)
        # _resize_and_crop returns [C, H, W] for single-frame input (squeeze removes dim 0)

        # Apply transforms
        image = self.transforms(image_resized)  # [C, H, W] -> [C, H, W]

        # Add frame dimension in VAE format: [C, H, W] -> [C, 1, H, W]
        image = image.unsqueeze(1)
        return image

    @staticmethod
    def _align_frame_count(num_frames: int, frame_mod: int = 8, min_frames: int = 1) -> int:
        """Pad frame count up to the nearest value satisfying num % frame_mod == 1.

        This avoids truncation: clips of any length are padded to the next valid
        frame count rather than snapped down to a predefined bucket value.
        """
        target = max(num_frames, min_frames)
        rem = target % frame_mod
        if rem != 1:
            target += (frame_mod + 1 - rem) % frame_mod
        return target

    def _preprocess_video(self, path: Path) -> tuple[torch.Tensor, float]:
        """Preprocess a video by loading, resizing, and applying transforms.
        Returns:
            Tuple of (video tensor in [C, F, H, W] format, fps)
        """
        # Load all frames (no frame-count limit — temporal length is preserved as-is)
        video, fps = read_video(path, max_frames=None)

        # Spatial resolution: pick the nearest bucket by aspect ratio
        _, target_height, target_width = self._get_spatial_bucket_for_item(video)
        frames_resized = self._resize_and_crop(video, target_height, target_width)

        # Temporal length: pad up to next valid frame count (%8 == 1), never truncate
        num_frames = frames_resized.shape[0]
        target_num_frames = self._align_frame_count(num_frames)
        if target_num_frames > num_frames:
            # Repeat the last frame to pad
            last_frame = frames_resized[-1].unsqueeze(0)
            pad = last_frame.expand(target_num_frames - num_frames, *last_frame.shape[1:])
            frames_resized = torch.cat([frames_resized, pad], dim=0)

        # Apply transforms to each frame and stack
        video = torch.stack([self.transforms(frame) for frame in frames_resized], dim=0)

        # Permute [F,C,H,W] -> [C,F,H,W] for VAE compatibility
        video = video.permute(1, 0, 2, 3).contiguous()

        return video, fps

    def _get_spatial_bucket_for_item(self, media_tensor: torch.Tensor) -> tuple[int, int, int]:
        """Pick the nearest bucket by spatial aspect ratio only; frame count is ignored."""
        num_frames, _, height, width = media_tensor.shape

        def distance(bucket: tuple[int, int, int]) -> float:
            _, bucket_height, bucket_width = bucket
            return abs(math.log(width / height) - math.log(bucket_width / bucket_height))

        nearest_bucket = min(self.resolution_buckets, key=distance)
        return nearest_bucket

    def _get_resolution_bucket_for_item(self, media_tensor: torch.Tensor) -> tuple[int, int, int]:
        """Legacy: kept for compatibility. Delegates to spatial bucket selection."""
        return self._get_spatial_bucket_for_item(media_tensor)

    def _resize_and_crop(self, media_tensor: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """Resize and crop tensor to target size."""
        # Get current dimensions
        current_height, current_width = media_tensor.shape[2], media_tensor.shape[3]

        # Calculate aspect ratios to determine which dimension to resize first
        current_aspect = current_width / current_height
        target_aspect = target_width / target_height

        # Resize while maintaining aspect ratio - scale to make the smaller dimension fit
        if current_aspect > target_aspect:
            # Current is wider than target, so scale by height
            new_width = int(current_width * target_height / current_height)
            media_tensor = resize(
                media_tensor,
                size=[target_height, new_width],  # type: ignore
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            # Current is taller than target, so scale by width
            new_height = int(current_height * target_width / current_width)
            media_tensor = resize(
                media_tensor,
                size=[new_height, target_width],
                interpolation=InterpolationMode.BICUBIC,
            )

        # Update dimensions after resize
        current_height, current_width = media_tensor.shape[2], media_tensor.shape[3]
        media_tensor = media_tensor.squeeze(0)

        # Calculate how much we need to crop from each dimension
        delta_h = current_height - target_height
        delta_w = current_width - target_width

        # Determine crop position based on reshape mode
        if self.reshape_mode == "random":
            # Random crop position
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif self.reshape_mode == "center":
            # Center crop
            top, left = delta_h // 2, delta_w // 2
        else:
            raise ValueError(f"Unsupported reshape mode: {self.reshape_mode}")

        # Perform the final crop to exact target dimensions
        media_tensor = crop(media_tensor, top=top, left=left, height=target_height, width=target_width)
        return media_tensor


def _clamp01_inplace(x: torch.Tensor) -> torch.Tensor:
    """In-place clamp to [0, 1]. Module-level so it is picklable for forkserver/spawn workers."""
    return x.clamp_(0, 1)


def compute_latents(  # noqa: PLR0913, PLR0915
    dataset_file: str | Path,
    video_column: str,
    resolution_buckets: list[tuple[int, int, int]],
    output_dir: str,
    model_path: str,
    main_media_column: str | None = None,
    reshape_mode: str = "center",
    batch_size: int = 1,
    device: str = "cuda",
    vae_tiling: bool = False,
    with_audio: bool = False,
    audio_output_dir: str | None = None,
) -> None:
    """
    Process videos and save latent representations.
    Args:
        dataset_file: Path to metadata file (CSV/JSON/JSONL) containing video paths
        video_column: Column name for video paths in the metadata file
        resolution_buckets: List of (frames, height, width) tuples
        output_dir: Directory to save video latents
        model_path: Path to LTX-2 checkpoint (.safetensors)
        reshape_mode: How to crop videos ("center", "random")
        main_media_column: Column name for main media paths (if different from video_column)
        batch_size: Batch size for processing
        device: Device to use for computation
        vae_tiling: Whether to enable VAE tiling
        with_audio: Whether to extract and encode audio from videos
        audio_output_dir: Directory to save audio latents (required if with_audio=True)
    """
    # Validate audio parameters
    if with_audio and audio_output_dir is None:
        raise ValueError("audio_output_dir must be provided when with_audio=True")

    console = Console()
    torch_device = torch.device(device)

    # Create dataset
    dataset = MediaDataset(
        dataset_file=dataset_file,
        main_media_column=main_media_column or video_column,
        video_column=video_column,
        resolution_buckets=resolution_buckets,
        reshape_mode=reshape_mode,
        with_audio=with_audio,
    )
    logger.info(f"Loaded {len(dataset)} valid media files")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up audio output directory if needed
    audio_output_path = None
    if with_audio:
        audio_output_path = Path(audio_output_dir)
        audio_output_path.mkdir(parents=True, exist_ok=True)

    # Load video VAE encoder
    with console.status(f"[bold]Loading video VAE encoder from [cyan]{model_path}[/]...", spinner="dots"):
        vae = load_video_vae_encoder(model_path, device=torch_device, dtype=torch.bfloat16)

    # Load audio VAE encoder and audio processor if needed
    audio_vae_encoder = None
    audio_processor = None
    if with_audio:
        # Audio VAE dtype: default float32; set PREPROCESS_AUDIO_DTYPE=bf16 to halve GPU compute.
        # WARNING: bf16 may slightly change audio latent quality; test end-to-end before adopting.
        _audio_dtype_env = os.environ.get("PREPROCESS_AUDIO_DTYPE", "fp32").lower()
        _audio_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(
            _audio_dtype_env, torch.float32
        )
        with console.status(f"[bold]Loading audio VAE encoder from [cyan]{model_path}[/]...", spinner="dots"):
            audio_vae_encoder = load_audio_vae_encoder(
                checkpoint_path=model_path,
                device=torch_device,
                dtype=_audio_dtype,
            )
            # Create audio processor for waveform-to-spectrogram conversion
            audio_processor = AudioProcessor(
                target_sample_rate=audio_vae_encoder.sample_rate,
                mel_bins=audio_vae_encoder.mel_bins,
                mel_hop_length=audio_vae_encoder.mel_hop_length,
                n_fft=audio_vae_encoder.n_fft,
            ).to(torch_device)

    # Create dataloader
    # Note: batch_size=1 required when with_audio because audio extraction can fail for some videos,
    # and the default collate function can't handle mixed None/dict values across a batch.
    # PyAV is not fork-safe, so when num_workers > 0 we use "forkserver" context, which re-execs a
    # clean Python interpreter per worker and avoids libav state being shared across processes.
    # Audio is extracted via an ffmpeg subprocess (not PyAV), so it is safe with worker processes.
    num_workers = int(os.environ.get("PREPROCESS_NUM_WORKERS", "8"))
    if with_audio:
        if batch_size > 1:
            logger.warning("Audio processing requires batch_size=1. Overriding batch_size to 1.")
            batch_size = 1

    loader_kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs["multiprocessing_context"] = "forkserver"
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(os.environ.get("PREPROCESS_PREFETCH_FACTOR", "4"))
    logger.info(
        f"DataLoader: num_workers={num_workers}, batch_size={batch_size}, "
        f"pin_memory=True, prefetch_factor={loader_kwargs.get('prefetch_factor', 'n/a')}"
    )
    dataloader = DataLoader(dataset, **loader_kwargs)

    # Track audio statistics
    audio_success_count = 0
    audio_skip_count = 0

    # Async save pool: torch.save (pickle + fsync) is the single biggest main-thread stall.
    # I/O releases the GIL, so a small thread pool gives real overlap with GPU encode.
    import concurrent.futures as _cf
    _save_workers = int(os.environ.get("PREPROCESS_SAVE_WORKERS", "4"))
    save_executor = _cf.ThreadPoolExecutor(max_workers=max(1, _save_workers), thread_name_prefix="pt-save")
    save_futures: list = []

    def _flush_save_futures(final: bool = False) -> None:
        """Drop completed futures; surface errors early. When final=True, wait for everything."""
        nonlocal save_futures
        if final:
            _cf.wait(save_futures)
            for fut in save_futures:
                fut.result()
            save_futures = []
            return
        # Opportunistic drain; cap queue depth to bound RAM usage.
        max_inflight = _save_workers * 4
        still = []
        for fut in save_futures:
            if fut.done():
                fut.result()  # raise if save failed
            else:
                still.append(fut)
        save_futures = still
        while len(save_futures) > max_inflight:
            # Oldest first; simple back-pressure.
            save_futures.pop(0).result()

    # Dedicated CUDA stream so audio VAE encode overlaps with next iteration's video VAE encode.
    audio_stream = (
        torch.cuda.Stream(device=torch_device)
        if (with_audio and torch.cuda.is_available())
        else None
    )

    # Process batches
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing videos", total=len(dataloader))

        for batch in dataloader:
            # Get video tensor - shape is [B, F, C, H, W] from DataLoader
            video = batch["video"]
            # Non-blocking H2D transfer so the next batch can be prepared by workers in parallel.
            if torch.cuda.is_available() and video.device.type == "cpu":
                video = video.to(torch_device, non_blocking=True)

            # Encode video
            with torch.inference_mode():
                video_latent_data = encode_video(vae=vae, video=video, use_tiling=vae_tiling)

            # Save latents for each item in batch
            for i in range(len(batch["relative_path"])):
                output_rel_path = Path(batch["main_media_relative_path"][i]).with_suffix(".pt")
                output_file = output_path / output_rel_path

                # Create output directory maintaining structure
                output_file.parent.mkdir(parents=True, exist_ok=True)

                # Index into batch to get this item's latents
                latent_data = {
                    "latents": video_latent_data["latents"][i].cpu().contiguous(),  # [C, F', H', W']
                    "num_frames": video_latent_data["num_frames"],
                    "height": video_latent_data["height"],
                    "width": video_latent_data["width"],
                    "fps": batch["video_metadata"]["fps"][i].item(),
                }

                # Async save: hand off pickle + disk write to a thread so GPU can start next encode.
                save_futures.append(save_executor.submit(torch.save, latent_data, output_file))

                # Process audio if enabled (audio is already extracted by the dataset)
                if with_audio:
                    audio_batch = batch.get("audio")
                    if audio_batch is not None:
                        # Extract the i-th item from batched audio data
                        # DataLoader collates [channels, samples] -> [batch, channels, samples]
                        audio_data = Audio(
                            waveform=audio_batch["waveform"][i],
                            sampling_rate=audio_batch["sample_rate"][i].item(),
                        )

                        # Encode audio on a separate CUDA stream so it overlaps with the NEXT
                        # iteration's video encode on the default stream.
                        if audio_stream is not None:
                            audio_stream.wait_stream(torch.cuda.current_stream(torch_device))
                            with torch.cuda.stream(audio_stream), torch.inference_mode():
                                audio_latents = encode_audio(audio_vae_encoder, audio_processor, audio_data)
                            # Make sure the d2h below sees the encode result.
                            torch.cuda.current_stream(torch_device).wait_stream(audio_stream)
                        else:
                            with torch.inference_mode():
                                audio_latents = encode_audio(audio_vae_encoder, audio_processor, audio_data)

                        # Save audio latents
                        audio_output_file = audio_output_path / output_rel_path
                        audio_output_file.parent.mkdir(parents=True, exist_ok=True)

                        audio_save_data = {
                            "latents": audio_latents["latents"].cpu().contiguous(),
                            "num_time_steps": audio_latents["num_time_steps"],
                            "frequency_bins": audio_latents["frequency_bins"],
                            "duration": audio_latents["duration"],
                        }

                        save_futures.append(
                            save_executor.submit(torch.save, audio_save_data, audio_output_file)
                        )
                        audio_success_count += 1
                    else:
                        # Video has no audio track
                        audio_skip_count += 1

            # Bound queue depth + surface errors from earlier saves.
            _flush_save_futures(final=False)

            progress.advance(task)

    # Wait for all outstanding saves to finish before we index .pt files below.
    _flush_save_futures(final=True)
    save_executor.shutdown(wait=True)

    # Write frame-count index: {relative_stem: num_latent_frames}
    # Used by BucketBatchSampler to group same-length clips per step.
    import json as _json
    frame_count_index: dict[str, int] = {}
    for pt_file in sorted(output_path.glob("**/*.pt")):
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=True)
            nf = int(data["num_frames"]) if isinstance(data, dict) else int(data.shape[1])
            frame_count_index[pt_file.stem] = nf
        except Exception:
            pass
    index_path = output_path / "frame_counts.json"
    index_path.write_text(_json.dumps(frame_count_index, indent=2))
    logger.info(f"Frame count index written: {index_path} ({len(frame_count_index)} entries)")

    # Log summary
    logger.info(f"Processed {len(dataset)} videos. Latents saved to {output_path}")
    if with_audio:
        logger.info(
            f"Audio processing: {audio_success_count} videos with audio, "
            f"{audio_skip_count} videos without audio (skipped)"
        )
        if audio_success_count == 0 and len(dataset) > 0:
            raise RuntimeError(
                f"Audio extraction failed for ALL {len(dataset)} videos. "
                f"No audio_latents were generated. Check that ffmpeg can extract audio: "
                f"ffmpeg -i '<your_clip.mp4>' -vn -f wav /tmp/test.wav"
            )


def encode_video(
    vae: torch.nn.Module,
    video: torch.Tensor,
    dtype: torch.dtype | None = None,
    use_tiling: bool = False,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
) -> dict[str, torch.Tensor | int]:
    """Encode video into non-patchified latent representation.
    Args:
        vae: Video VAE encoder model
        video: Input tensor of shape [B, C, F, H, W] (batch, channels, frames, height, width)
               This is the format expected by the VAE encoder.
        dtype: Target dtype for output latents
        use_tiling: Whether to use spatial tiling for memory efficiency
        tile_size: Tile size in pixels (must be divisible by 32)
        tile_overlap: Overlap between tiles in pixels (must be divisible by 32)
    Returns:
        Dict containing non-patchified latents and shape information:
        {
            "latents": Tensor[B, C, F', H', W'],  # Non-patchified format with batch dim
            "num_frames": int,  # Latent frame count
            "height": int,  # Latent height
            "width": int,  # Latent width
        }
    """
    device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype

    # Add batch dimension if needed
    if video.ndim == 4:
        video = video.unsqueeze(0)  # [C, F, H, W] -> [B, C, F, H, W]

    video = video.to(device=device, dtype=vae_dtype)

    # Choose encoding method based on tiling flag
    if use_tiling:
        latents = tiled_encode_video(
            vae=vae,
            video=video,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
    else:
        # Encode video - VAE expects [B, C, F, H, W], returns [B, C, F', H', W']
        latents = vae(video)

    if dtype is not None:
        latents = latents.to(dtype=dtype)

    _, _, num_frames, height, width = latents.shape

    return {
        "latents": latents,  # [B, C, F', H', W']
        "num_frames": num_frames,
        "height": height,
        "width": width,
    }


def tiled_encode_video(  # noqa: PLR0912, PLR0915
    vae: torch.nn.Module,
    video: torch.Tensor,
    tile_size: int = DEFAULT_TILE_SIZE,
    tile_overlap: int = DEFAULT_TILE_OVERLAP,
) -> torch.Tensor:
    """Encode video using spatial tiling for memory efficiency.
    Splits the video into overlapping spatial tiles, encodes each tile separately,
    and blends the results using linear feathering in the overlap regions.
    Args:
        vae: Video VAE encoder model
        video: Input tensor of shape [B, C, F, H, W]
        tile_size: Tile size in pixels (must be divisible by 32)
        tile_overlap: Overlap between tiles in pixels (must be divisible by 32)
    Returns:
        Encoded latent tensor [B, C_latent, F_latent, H_latent, W_latent]
    """
    batch, _channels, frames, height, width = video.shape
    device = video.device
    dtype = video.dtype

    # Validate tile parameters
    if tile_size % VAE_SPATIAL_FACTOR != 0:
        raise ValueError(f"tile_size must be divisible by {VAE_SPATIAL_FACTOR}, got {tile_size}")
    if tile_overlap % VAE_SPATIAL_FACTOR != 0:
        raise ValueError(f"tile_overlap must be divisible by {VAE_SPATIAL_FACTOR}, got {tile_overlap}")
    if tile_overlap >= tile_size:
        raise ValueError(f"tile_overlap ({tile_overlap}) must be less than tile_size ({tile_size})")

    # If video fits in a single tile, use regular encoding
    if height <= tile_size and width <= tile_size:
        return vae(video)

    # Calculate output dimensions
    # VAE compresses: H -> H/32, W -> W/32, F -> 1 + (F-1)/8
    output_height = height // VAE_SPATIAL_FACTOR
    output_width = width // VAE_SPATIAL_FACTOR
    output_frames = 1 + (frames - 1) // VAE_TEMPORAL_FACTOR

    # Latent channels (128 for LTX-2)
    # Get from a small test encode or assume 128
    latent_channels = 128

    # Initialize output and weight tensors
    output = torch.zeros(
        (batch, latent_channels, output_frames, output_height, output_width),
        device=device,
        dtype=dtype,
    )
    weights = torch.zeros(
        (batch, 1, output_frames, output_height, output_width),
        device=device,
        dtype=dtype,
    )

    # Calculate tile positions with overlap
    # Step size is tile_size - tile_overlap
    step_h = tile_size - tile_overlap
    step_w = tile_size - tile_overlap

    h_positions = list(range(0, max(1, height - tile_overlap), step_h))
    w_positions = list(range(0, max(1, width - tile_overlap), step_w))

    # Ensure last tile covers the edge
    if h_positions[-1] + tile_size < height:
        h_positions.append(height - tile_size)
    if w_positions[-1] + tile_size < width:
        w_positions.append(width - tile_size)

    # Remove duplicates and sort
    h_positions = sorted(set(h_positions))
    w_positions = sorted(set(w_positions))

    # Overlap in latent space
    overlap_out_h = tile_overlap // VAE_SPATIAL_FACTOR
    overlap_out_w = tile_overlap // VAE_SPATIAL_FACTOR

    # Process each tile
    for h_pos in h_positions:
        for w_pos in w_positions:
            # Calculate tile boundaries in input space
            h_start = max(0, h_pos)
            w_start = max(0, w_pos)
            h_end = min(h_start + tile_size, height)
            w_end = min(w_start + tile_size, width)

            # Ensure tile dimensions are divisible by VAE_SPATIAL_FACTOR
            tile_h = ((h_end - h_start) // VAE_SPATIAL_FACTOR) * VAE_SPATIAL_FACTOR
            tile_w = ((w_end - w_start) // VAE_SPATIAL_FACTOR) * VAE_SPATIAL_FACTOR

            if tile_h < VAE_SPATIAL_FACTOR or tile_w < VAE_SPATIAL_FACTOR:
                continue

            # Adjust end positions
            h_end = h_start + tile_h
            w_end = w_start + tile_w

            # Extract tile
            tile = video[:, :, :, h_start:h_end, w_start:w_end]

            # Encode tile
            encoded_tile = vae(tile)

            # Get actual encoded dimensions
            _, _, tile_out_frames, tile_out_height, tile_out_width = encoded_tile.shape

            # Calculate output positions
            out_h_start = h_start // VAE_SPATIAL_FACTOR
            out_w_start = w_start // VAE_SPATIAL_FACTOR
            out_h_end = min(out_h_start + tile_out_height, output_height)
            out_w_end = min(out_w_start + tile_out_width, output_width)

            # Trim encoded tile if necessary
            actual_tile_h = out_h_end - out_h_start
            actual_tile_w = out_w_end - out_w_start
            encoded_tile = encoded_tile[:, :, :, :actual_tile_h, :actual_tile_w]

            # Create blending mask with linear feathering at edges
            mask = torch.ones(
                (1, 1, tile_out_frames, actual_tile_h, actual_tile_w),
                device=device,
                dtype=dtype,
            )

            # Apply feathering at edges (linear blend in overlap regions)
            # Left edge
            if h_pos > 0 and overlap_out_h > 0 and overlap_out_h < actual_tile_h:
                fade_in = torch.linspace(0.0, 1.0, overlap_out_h + 2, device=device, dtype=dtype)[1:-1]
                mask[:, :, :, :overlap_out_h, :] *= fade_in.view(1, 1, 1, -1, 1)

            # Right edge (bottom in height dimension)
            if h_end < height and overlap_out_h > 0 and overlap_out_h < actual_tile_h:
                fade_out = torch.linspace(1.0, 0.0, overlap_out_h + 2, device=device, dtype=dtype)[1:-1]
                mask[:, :, :, -overlap_out_h:, :] *= fade_out.view(1, 1, 1, -1, 1)

            # Top edge (left in width dimension)
            if w_pos > 0 and overlap_out_w > 0 and overlap_out_w < actual_tile_w:
                fade_in = torch.linspace(0.0, 1.0, overlap_out_w + 2, device=device, dtype=dtype)[1:-1]
                mask[:, :, :, :, :overlap_out_w] *= fade_in.view(1, 1, 1, 1, -1)

            # Bottom edge (right in width dimension)
            if w_end < width and overlap_out_w > 0 and overlap_out_w < actual_tile_w:
                fade_out = torch.linspace(1.0, 0.0, overlap_out_w + 2, device=device, dtype=dtype)[1:-1]
                mask[:, :, :, :, -overlap_out_w:] *= fade_out.view(1, 1, 1, 1, -1)

            # Accumulate weighted results
            output[:, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += encoded_tile * mask
            weights[:, :, :, out_h_start:out_h_end, out_w_start:out_w_end] += mask

    # Normalize by weights (avoid division by zero)
    output = output / (weights + 1e-8)

    return output


def encode_audio(
    audio_vae_encoder: torch.nn.Module,
    audio_processor: torch.nn.Module,
    audio: Audio,
) -> dict[str, torch.Tensor | int | float]:
    """Encode audio waveform into latent representation.
    Args:
        audio_vae_encoder: Audio VAE encoder model from ltx-core
        audio_processor: AudioProcessor for waveform-to-spectrogram conversion
        audio: Audio container with waveform tensor and sampling rate.
    Returns:
        Dict containing audio latents and shape information:
        {
            "latents": Tensor[C, T, F],  # Non-patchified format
            "num_time_steps": int,
            "frequency_bins": int,
            "duration": float,
        }
    """
    device = next(audio_vae_encoder.parameters()).device
    dtype = next(audio_vae_encoder.parameters()).dtype

    waveform = audio.waveform.to(device=device, dtype=dtype)

    # Add batch dimension if needed: [channels, samples] -> [batch, channels, samples]
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    # Calculate duration
    duration = waveform.shape[-1] / audio.sampling_rate

    # Convert waveform to mel spectrogram using AudioProcessor
    mel_spectrogram = audio_processor.waveform_to_mel(Audio(waveform=waveform, sampling_rate=audio.sampling_rate))
    mel_spectrogram = mel_spectrogram.to(dtype=dtype)

    # Encode mel spectrogram to latents
    latents = audio_vae_encoder(mel_spectrogram)

    # latents shape: [batch, channels, time, freq] = [1, 8, T, 16]
    _, _channels, time_steps, freq_bins = latents.shape

    return {
        "latents": latents.squeeze(0),  # [C, T, F] - remove batch dim
        "num_time_steps": time_steps,
        "frequency_bins": freq_bins,
        "duration": duration,
    }


def parse_resolution_buckets(resolution_buckets_str: str) -> list[tuple[int, int, int]]:
    """Parse resolution buckets from string format to list of tuples (frames, height, width)"""
    resolution_buckets = []
    for bucket_str in resolution_buckets_str.split(";"):
        w, h, f = map(int, bucket_str.split("x"))

        if w % VAE_SPATIAL_FACTOR != 0 or h % VAE_SPATIAL_FACTOR != 0:
            raise typer.BadParameter(
                f"Width and height must be multiples of {VAE_SPATIAL_FACTOR}, got {w}x{h}",
                param_hint="resolution-buckets",
            )

        if f % VAE_TEMPORAL_FACTOR != 1:
            raise typer.BadParameter(
                f"Number of frames must be a multiple of {VAE_TEMPORAL_FACTOR} plus 1, got {f}",
                param_hint="resolution-buckets",
            )

        resolution_buckets.append((f, h, w))
    return resolution_buckets


def compute_scaled_resolution_buckets(
    resolution_buckets: list[tuple[int, int, int]],
    scale_factor: int,
) -> list[tuple[int, int, int]]:
    """Compute scaled resolution buckets and validate the results."""
    if scale_factor == 1:
        return resolution_buckets

    scaled_buckets = []
    for frames, height, width in resolution_buckets:
        # Validate that scale factor evenly divides the dimensions
        if height % scale_factor != 0:
            raise ValueError(
                f"Height {height} is not evenly divisible by scale factor {scale_factor}. "
                f"Choose a scale factor that divides {height} evenly."
            )
        if width % scale_factor != 0:
            raise ValueError(
                f"Width {width} is not evenly divisible by scale factor {scale_factor}. "
                f"Choose a scale factor that divides {width} evenly."
            )

        scaled_height = height // scale_factor
        scaled_width = width // scale_factor

        # Validate scaled dimensions are divisible by VAE spatial factor
        if scaled_height % VAE_SPATIAL_FACTOR != 0:
            raise ValueError(
                f"Scaled height {scaled_height} (from {height} / {scale_factor}) "
                f"is not divisible by {VAE_SPATIAL_FACTOR}. "
                f"Choose a different scale factor or adjust your resolution buckets."
            )
        if scaled_width % VAE_SPATIAL_FACTOR != 0:
            raise ValueError(
                f"Scaled width {scaled_width} (from {width} / {scale_factor}) "
                f"is not divisible by {VAE_SPATIAL_FACTOR}. "
                f"Choose a different scale factor or adjust your resolution buckets."
            )

        scaled_buckets.append((frames, scaled_height, scaled_width))

    return scaled_buckets


@app.command()
def main(  # noqa: PLR0913
    dataset_file: str = typer.Argument(
        ...,
        help="Path to metadata file (CSV/JSON/JSONL) containing video paths",
    ),
    resolution_buckets: str = typer.Option(
        ...,
        help='Resolution buckets in format "WxHxF;WxHxF;..." (e.g. "768x768x25;512x512x49")',
    ),
    output_dir: str = typer.Option(
        ...,
        help="Output directory to save video latents",
    ),
    model_path: str = typer.Option(
        ...,
        help="Path to LTX-2 checkpoint (.safetensors file)",
    ),
    video_column: str = typer.Option(
        default="media_path",
        help="Column name in the dataset JSON/JSONL/CSV file containing video paths",
    ),
    batch_size: int = typer.Option(
        default=1,
        help="Batch size for processing",
    ),
    device: str = typer.Option(
        default="cuda",
        help="Device to use for computation",
    ),
    vae_tiling: bool = typer.Option(
        default=False,
        help="Enable VAE tiling for larger video resolutions",
    ),
    reshape_mode: str = typer.Option(
        default="center",
        help="How to crop videos: 'center' or 'random'",
    ),
    with_audio: bool = typer.Option(
        default=False,
        help="Extract and encode audio from video files",
    ),
    audio_output_dir: str | None = typer.Option(
        default=None,
        help="Output directory for audio latents (required if --with-audio is set)",
    ),
) -> None:
    """Process videos/images and save latent representations for video generation training.
    This script processes videos and images from metadata files and saves latent representations
    that can be used for training video generation models. The output latents will maintain
    the same folder structure and naming as the corresponding media files.
    Examples:
        # Process videos from a CSV file
        python scripts/process_videos.py dataset.csv --resolution-buckets 768x768x25 \\
            --output-dir ./latents --model-path /path/to/ltx2.safetensors
        # Process videos from a JSON file with custom video column
        python scripts/process_videos.py dataset.json --resolution-buckets 768x768x25 \\
            --output-dir ./latents --model-path /path/to/ltx2.safetensors --video-column "video_path"
        # Enable VAE tiling to save GPU VRAM
        python scripts/process_videos.py dataset.csv --resolution-buckets 1024x1024x25 \\
            --output-dir ./latents --model-path /path/to/ltx2.safetensors --vae-tiling
        # Process videos with audio
        python scripts/process_videos.py dataset.csv --resolution-buckets 768x768x25 \\
            --output-dir ./latents --model-path /path/to/ltx2.safetensors \\
            --with-audio --audio-output-dir ./audio_latents
    """

    # Validate dataset file exists
    if not Path(dataset_file).is_file():
        raise typer.BadParameter(f"Dataset file not found: {dataset_file}")

    # Validate audio parameters
    if with_audio and audio_output_dir is None:
        raise typer.BadParameter("--audio-output-dir is required when --with-audio is set")

    # Parse resolution buckets
    parsed_resolution_buckets = parse_resolution_buckets(resolution_buckets)

    if len(parsed_resolution_buckets) > 1:
        logger.warning(
            "Using multiple resolution buckets. "
            "When training with multiple resolution buckets, you must use a batch size of 1."
        )

    # Process latents
    compute_latents(
        dataset_file=dataset_file,
        video_column=video_column,
        resolution_buckets=parsed_resolution_buckets,
        output_dir=output_dir,
        model_path=model_path,
        reshape_mode=reshape_mode,
        batch_size=batch_size,
        device=device,
        vae_tiling=vae_tiling,
        with_audio=with_audio,
        audio_output_dir=audio_output_dir,
    )


if __name__ == "__main__":
    app()
