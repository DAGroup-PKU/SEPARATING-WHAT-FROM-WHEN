#!/usr/bin/env python

"""
Train LTXV models using configuration from YAML files.
This script provides a command-line interface for training LTXV models using
either LoRA fine-tuning or full model fine-tuning. It loads configuration from
a YAML file and passes it to the trainer.
Basic usage:
    python scripts/train.py CONFIG_PATH [--disable-progress-bars]
Resume is automatic when a training state file exists next to the loaded checkpoint.
To start fresh, set `checkpoints.no_resume: true` in the YAML config.
For multi-GPU/FSDP training, configure and launch via Accelerate:
    accelerate config
    accelerate launch scripts/train.py CONFIG_PATH
"""

from pathlib import Path

import typer
import yaml
from rich.console import Console

from ltx_trainer.config import LtxTrainerConfig
from ltx_trainer.trainer import LtxvTrainer

console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Train LTXV models using configuration from YAML files.",
)


def _apply_path_overrides(
    config_data: dict,
    *,
    model_path: str | None,
    text_encoder_path: str | None,
    output_dir: str | None,
    data_root: str | None,
    load_checkpoint: str | None,
    steps: int | None,
) -> dict:
    """Overlay explicit path / step flags onto the yaml dict before validation."""
    model = config_data.setdefault("model", {})
    data = config_data.setdefault("data", {})
    optimization = config_data.setdefault("optimization", {})
    if model_path:
        model["model_path"] = model_path
    if text_encoder_path:
        model["text_encoder_path"] = text_encoder_path
    if load_checkpoint:
        model["load_checkpoint"] = load_checkpoint
    if output_dir:
        config_data["output_dir"] = output_dir
    if data_root:
        data["preprocessed_data_root"] = data_root
    if steps is not None:
        optimization["steps"] = steps
    return config_data


@app.command()
def main(
    config_path: str = typer.Argument(..., help="Path to YAML configuration file"),
    disable_progress_bars: bool = typer.Option(
        False,
        "--disable-progress-bars",
        help="Disable progress bars (useful for multi-process runs)",
    ),
    model_path: str | None = typer.Option(
        None,
        "--model-path",
        help="Override model.model_path (LTX-2.3 checkpoint .safetensors).",
    ),
    text_encoder_path: str | None = typer.Option(
        None,
        "--text-encoder-path",
        help="Override model.text_encoder_path (Gemma directory).",
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        help="Override output_dir for checkpoints and samples.",
    ),
    data_root: str | None = typer.Option(
        None,
        "--data-root",
        help="Override data.preprocessed_data_root.",
    ),
    load_checkpoint: str | None = typer.Option(
        None,
        "--load-checkpoint",
        help="Override model.load_checkpoint (resume weights).",
    ),
    steps: int | None = typer.Option(
        None,
        "--steps",
        help="Override optimization.steps (useful for a short smoke run).",
    ),
) -> None:
    """Train the model using the provided configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        typer.echo(f"Error: Configuration file {config_path} does not exist.")
        raise typer.Exit(code=1)

    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    config_data = _apply_path_overrides(
        config_data,
        model_path=model_path,
        text_encoder_path=text_encoder_path,
        output_dir=output_dir,
        data_root=data_root,
        load_checkpoint=load_checkpoint,
        steps=steps,
    )

    try:
        trainer_config = LtxTrainerConfig(**config_data)
    except Exception as e:
        typer.echo(f"Error: Invalid configuration data: {e}")
        raise typer.Exit(code=1) from e

    trainer = LtxvTrainer(trainer_config)
    trainer.train(disable_progress_bars=disable_progress_bars)


if __name__ == "__main__":
    app()
