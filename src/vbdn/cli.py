"""Command-line interface and command dispatch."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import (
    ALL_DATASETS,
    DEFAULT_BIG2015_PATH,
    DEFAULT_DATASETS,
    DEFAULT_PRETRAINED_MODELS,
    PAPER_EPOCHS,
    PAPER_IMAGE_SIZE,
    PAPER_LEARNING_RATE,
    PAPER_MOMENTUM,
    PAPER_PRETRAINED_BATCH_SIZE,
    PAPER_TEST_BATCH_SIZE,
    PAPER_TRAIN_BATCH_SIZE,
    RESULTS_DIR,
    RUN_ALL_CONVNET_EPOCHS,
    RUN_ALL_PRETRAINED_EPOCHS,
    SAFE_EPOCHS,
    SAFE_IMAGE_SIZE,
    SAFE_PRETRAINED_BATCH_SIZE,
    SAFE_PRETRAINED_LEARNING_RATE,
    SAFE_TEST_BATCH_SIZE,
    SAFE_TRAIN_BATCH_SIZE,
    SEED,
    ensure_output_dirs,
)


def _add_dataset_options(parser: argparse.ArgumentParser, defaults=DEFAULT_DATASETS) -> None:
    """Attach dataset discovery and BIG2015 conversion flags."""
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(defaults),
        help="Known dataset names or ImageFolder paths",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download known Kaggle datasets when needed",
    )
    parser.add_argument(
        "--big2015-path",
        type=Path,
        default=DEFAULT_BIG2015_PATH,
        help="Folder containing BIG2015 train.7z and trainLabels.csv",
    )
    parser.add_argument(
        "--big2015-images",
        type=Path,
        help="Image output folder; defaults to <big2015-path>/images",
    )
    parser.add_argument("--big2015-workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--big2015-limit",
        type=int,
        help="Convert at most N BIG2015 samples (useful for smoke tests)",
    )
    parser.add_argument(
        "--big2015-samples-per-class",
        type=int,
        help="Stream a balanced BIG2015 subset directly from train.7z",
    )
    parser.add_argument(
        "--rebuild-big2015",
        action="store_true",
        help="Regenerate BIG2015 images that already exist",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        help="Limit each class for resource-safe smoke tests",
    )


def _add_training_options(
    parser: argparse.ArgumentParser,
    epochs_default: int | None = SAFE_EPOCHS,
) -> None:
    """Attach paper-aligned optimization flags."""
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Train fraction for BIG2015 and custom paths; known supplied splits stay fixed",
    )
    parser.add_argument("--batch-size", type=int, default=SAFE_TRAIN_BATCH_SIZE)
    parser.add_argument("--test-batch-size", type=int, default=SAFE_TEST_BATCH_SIZE)
    parser.add_argument(
        "--pretrained-batch-size",
        type=int,
        default=SAFE_PRETRAINED_BATCH_SIZE,
        help="Training batch size for pretrained model comparisons",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--epochs",
        type=int,
        default=epochs_default,
        help=(
            "Epochs for every CNN; on run-all, overrides both model-family defaults"
            if epochs_default is None
            else "Number of training epochs"
        ),
    )
    parser.add_argument("--lr", type=float, default=PAPER_LEARNING_RATE)
    parser.add_argument(
        "--pretrained-lr",
        type=float,
        default=SAFE_PRETRAINED_LEARNING_RATE,
        help="Learning rate used only for pretrained models",
    )
    parser.add_argument("--momentum", type=float, default=PAPER_MOMENTUM)
    parser.add_argument("--image-size", type=int, default=SAFE_IMAGE_SIZE)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument(
        "--save-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist CNN checkpoints after training",
    )
    parser.add_argument(
        "--paper-settings",
        action="store_true",
        help="Override training parameters with the exact paper profile",
    )


def _add_malevis_option(parser: argparse.ArgumentParser) -> None:
    """Attach the paper's optional MaleVis no-Other experiment flag."""
    parser.add_argument(
        "--malevis-without-other",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also train the main ConvNet after removing MaleVis Other",
    )


def _add_pretrained_options(parser: argparse.ArgumentParser) -> None:
    """Attach pretrained model selection flags."""
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_PRETRAINED_MODELS))


def _add_glcm_options(parser: argparse.ArgumentParser) -> None:
    """Attach classical model and GLCM experiment flags."""
    parser.add_argument(
        "--glcm-variants",
        nargs="+",
        choices=("raw", "balanced-augmented"),
        default=["raw", "balanced-augmented"],
        help="Run raw and/or balanced augmented GLCM experiments",
    )
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--rf-estimators", type=int, default=100)
    parser.add_argument("--gbdt-estimators", type=int, default=100)
    parser.add_argument("--xgb-estimators", type=int, default=100)
    parser.add_argument(
        "--glcm-image-size",
        type=int,
        default=128,
        help="Square working size for resource-safe GLCM extraction",
    )
    parser.add_argument(
        "--glcm-levels",
        type=int,
        default=256,
        help="GLCM gray levels; 256 follows the paper's 0-255 intensity range",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the complete CLI without importing ML pipelines."""
    parser = argparse.ArgumentParser(
        prog="malware-cli",
        description="Train and report malware image classifiers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version="%(prog)s 2.0")
    commands = parser.add_subparsers(dest="command", required=True)

    def dataset_command(name: str, help_text: str, defaults=DEFAULT_DATASETS):
        """Create a command sharing only dataset options."""
        command = commands.add_parser(
            name,
            help=help_text,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        _add_dataset_options(command, defaults)
        return command

    convnet = dataset_command("convnet", "Train the custom convolutional network")
    _add_training_options(convnet)
    _add_malevis_option(convnet)

    pretrained = dataset_command("pretrained", "Fine-tune pretrained CNNs")
    _add_training_options(pretrained)
    _add_pretrained_options(pretrained)

    glcm = dataset_command("glcm", "Train classical models on GLCM features")
    glcm.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Train fraction for BIG2015 and custom paths; known supplied splits stay fixed",
    )
    _add_glcm_options(glcm)

    run_all = dataset_command("run-all", "Run every model on every dataset", ALL_DATASETS)
    _add_training_options(run_all, epochs_default=None)
    _add_malevis_option(run_all)
    run_all.add_argument(
        "--convnet-epochs",
        type=int,
        default=RUN_ALL_CONVNET_EPOCHS,
        help="Epochs for the custom ConvNet in run-all",
    )
    run_all.add_argument(
        "--pretrained-epochs",
        type=int,
        default=RUN_ALL_PRETRAINED_EPOCHS,
        help="Epochs for each pretrained deep model in run-all",
    )
    run_all.add_argument("--skip-convnet", action="store_true")
    run_all.add_argument("--skip-pretrained", action="store_true")
    run_all.add_argument("--skip-glcm", action="store_true")
    _add_pretrained_options(run_all)
    _add_glcm_options(run_all)

    # Preserve the original command name.
    all_alias = dataset_command("all", "Alias for run-all", ALL_DATASETS)
    _add_training_options(all_alias, epochs_default=None)
    _add_malevis_option(all_alias)
    all_alias.add_argument(
        "--convnet-epochs",
        type=int,
        default=RUN_ALL_CONVNET_EPOCHS,
        help="Epochs for the custom ConvNet in run-all",
    )
    all_alias.add_argument(
        "--pretrained-epochs",
        type=int,
        default=RUN_ALL_PRETRAINED_EPOCHS,
        help="Epochs for each pretrained deep model in run-all",
    )
    all_alias.add_argument("--skip-convnet", action="store_true")
    all_alias.add_argument("--skip-pretrained", action="store_true")
    all_alias.add_argument("--skip-glcm", action="store_true")
    _add_pretrained_options(all_alias)
    _add_glcm_options(all_alias)

    prepare = commands.add_parser(
        "prepare-big2015",
        help="Stream BIG2015 byte files from train.7z into PNG images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    prepare.add_argument("--source", type=Path, default=DEFAULT_BIG2015_PATH)
    prepare.add_argument("--output", type=Path)
    prepare.add_argument("--workers", type=int, default=4)
    prepare.add_argument("--limit", type=int)
    prepare.add_argument(
        "--samples-per-class",
        type=int,
        help="Stream a balanced subset instead of a global prefix",
    )
    prepare.add_argument("--overwrite", action="store_true")
    return parser


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Reject invalid numeric options before expensive work starts."""
    if hasattr(args, "train_split") and not 0 < args.train_split < 1:
        parser.error("--train-split must be between 0 and 1")
    limits = {
        "batch_size": 1,
        "test_batch_size": 1,
        "pretrained_batch_size": 1,
        "num_workers": 0,
        "epochs": 1,
        "convnet_epochs": 1,
        "pretrained_epochs": 1,
        "big2015_workers": 1,
        "big2015_samples_per_class": 1,
        "max_samples_per_class": 2,
        "glcm_image_size": 8,
        "glcm_levels": 2,
    }
    for name, minimum in limits.items():
        if (
            hasattr(args, name)
            and getattr(args, name) is not None
            and getattr(args, name) < minimum
        ):
            parser.error(f"--{name.replace('_', '-')} must be at least {minimum}")
    if hasattr(args, "glcm_levels") and (args.glcm_levels > 256 or 256 % args.glcm_levels != 0):
        parser.error("--glcm-levels must be a divisor of 256 between 2 and 256")


def _apply_training_profile(args: argparse.Namespace) -> None:
    """Apply the exact paper profile when explicitly requested."""
    if not getattr(args, "paper_settings", False):
        return
    args.epochs = PAPER_EPOCHS
    args.image_size = PAPER_IMAGE_SIZE
    args.batch_size = PAPER_TRAIN_BATCH_SIZE
    args.test_batch_size = PAPER_TEST_BATCH_SIZE
    args.pretrained_batch_size = PAPER_PRETRAINED_BATCH_SIZE
    args.pretrained_lr = PAPER_LEARNING_RATE
    args.lr = PAPER_LEARNING_RATE
    args.momentum = PAPER_MOMENTUM


def _write_run_metadata(args: argparse.Namespace) -> None:
    """Record runtime and arguments needed to reproduce the latest run."""
    payload = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "python": sys.version,
        "platform": platform.platform(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (RESULTS_DIR / "last_run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_all(args: argparse.Namespace) -> None:
    """Execute every requested pipeline and create global comparisons."""
    from .glcm_pipeline import run_glcm
    from .reporting import save_comparison_plots
    from .trainers import run_convnet, run_pretrained

    results = []
    if not args.skip_convnet:
        convnet_args = copy.copy(args)
        convnet_args.epochs = args.epochs or args.convnet_epochs
        print(f"ConvNet training epochs: {convnet_args.epochs}")
        results.extend(run_convnet(convnet_args))
    if not args.skip_pretrained:
        pretrained_args = copy.copy(args)
        pretrained_args.epochs = args.epochs or args.pretrained_epochs
        print(f"Pretrained model training epochs: {pretrained_args.epochs}")
        results.extend(run_pretrained(pretrained_args))
    if not args.skip_glcm:
        results.extend(run_glcm(args))
    save_comparison_plots(results, prefix="all_models")


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch one CLI command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_training_profile(args)
    _validate(args, parser)

    if args.command == "prepare-big2015":
        from .big2015 import prepare_big2015

        prepare_big2015(
            args.source,
            args.output,
            args.workers,
            args.overwrite,
            args.limit,
            args.samples_per_class,
        )
        return

    ensure_output_dirs()
    _write_run_metadata(args)
    print(f"Command: {args.command}")

    if args.command == "convnet":
        from .trainers import run_convnet

        run_convnet(args)
    elif args.command == "pretrained":
        from .trainers import run_pretrained

        run_pretrained(args)
    elif args.command == "glcm":
        from .glcm_pipeline import run_glcm

        run_glcm(args)
    else:
        _run_all(args)
