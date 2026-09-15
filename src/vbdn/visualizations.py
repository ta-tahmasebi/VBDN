"""Dataset and feature visualizations used by experiment reports."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from skimage.feature import graycomatrix

from .config import PLOT_DIR
from .reporting import safe_name


def save_dataset_gallery(
    paths,
    labels,
    classes: list[str],
    dataset_name: str,
    samples_per_class: int = 2,
) -> Path:
    """Show representative malware images from every class."""
    paths = np.asarray(paths)
    labels = np.asarray(labels)
    columns = min(6, max(1, len(classes)))
    rows = math.ceil(len(classes) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 2.6, rows * 2.6),
        squeeze=False,
    )
    for class_index, class_name in enumerate(classes):
        axis = axes.flat[class_index]
        class_paths = paths[labels == class_index][:samples_per_class]
        if len(class_paths):
            with Image.open(class_paths[0]) as image:
                axis.imshow(image.convert("L"), cmap="gray")
        axis.set_title(f"{class_name}\n(n={int(np.sum(labels == class_index))})", fontsize=9)
        axis.axis("off")
    for axis in axes.flat[len(classes) :]:
        axis.axis("off")
    fig.suptitle(f"Representative malware images - {dataset_name}")
    fig.tight_layout()
    path = PLOT_DIR / f"{safe_name(dataset_name)}__class_gallery.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_glcm_gallery(paths, labels, classes: list[str], dataset_name: str) -> Path:
    """Compare representative grayscale images with their GLCM maps."""
    paths = np.asarray(paths)
    labels = np.asarray(labels)
    selected = np.linspace(0, len(classes) - 1, min(6, len(classes)), dtype=int)
    fig, axes = plt.subplots(len(selected), 2, figsize=(9, 3.2 * len(selected)), squeeze=False)
    for row, class_index in enumerate(selected):
        sample = paths[labels == class_index][0]
        with Image.open(sample) as image:
            gray = image.convert("L").resize((128, 128), Image.Resampling.LANCZOS)
            quantized = np.asarray(gray, dtype=np.uint8) // 32
        matrix = graycomatrix(
            quantized,
            distances=(1,),
            angles=(0,),
            levels=8,
            symmetric=True,
            normed=True,
        )[:, :, 0, 0]
        axes[row, 0].imshow(gray, cmap="gray")
        axes[row, 0].set_title(f"{classes[class_index]} - image")
        sns.heatmap(matrix, cmap="mako", ax=axes[row, 1], cbar=True)
        axes[row, 1].set_title(f"{classes[class_index]} - GLCM")
        axes[row, 0].axis("off")
        axes[row, 1].set(xlabel="Neighbor gray level", ylabel="Reference gray level")
    fig.suptitle(f"Image and GLCM examples - {dataset_name}")
    fig.tight_layout()
    path = PLOT_DIR / f"{safe_name(dataset_name)}__glcm_gallery.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_augmentation_gallery(paths, labels, classes: list[str], dataset_name: str) -> Path:
    """Compare original samples with the paper's flip and rotation transforms."""
    paths = np.asarray(paths)
    labels = np.asarray(labels)
    selected = np.linspace(0, len(classes) - 1, min(6, len(classes)), dtype=int)
    fig, axes = plt.subplots(len(selected), 3, figsize=(10, 3.1 * len(selected)), squeeze=False)
    for row, class_index in enumerate(selected):
        sample = paths[labels == class_index][0]
        with Image.open(sample) as source:
            original = source.convert("L").resize((256, 256), Image.Resampling.LANCZOS)
        variants = (
            ("Original", original),
            ("Horizontal flip", original.transpose(Image.Transpose.FLIP_LEFT_RIGHT)),
            ("Rotation (+20°)", original.rotate(20, resample=Image.Resampling.BILINEAR)),
        )
        for column, (title, image) in enumerate(variants):
            axes[row, column].imshow(image, cmap="gray")
            axes[row, column].set_title(f"{classes[class_index]} - {title}")
            axes[row, column].axis("off")
    fig.suptitle(f"Data augmentation examples - {dataset_name}")
    fig.tight_layout()
    path = PLOT_DIR / f"{safe_name(dataset_name)}__augmentation_examples.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_balance_comparison(
    raw_labels,
    balanced_labels,
    classes: list[str],
    dataset_name: str,
) -> Path:
    """Compare training class counts before and after weighted resampling."""
    raw_counts = np.bincount(raw_labels, minlength=len(classes))
    balanced_counts = np.bincount(balanced_labels, minlength=len(classes))
    positions = np.arange(len(classes))
    fig, axis = plt.subplots(figsize=(max(10, len(classes) * 0.55), 5.5))
    axis.bar(positions - 0.2, raw_counts, width=0.4, label="Raw train")
    axis.bar(positions + 0.2, balanced_counts, width=0.4, label="Balanced + augmented")
    axis.set_xticks(positions, classes, rotation=60, ha="right")
    axis.set(title=f"Training class balance - {dataset_name}", ylabel="Samples")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = PLOT_DIR / f"{safe_name(dataset_name)}__balance_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_feature_importance(
    model,
    feature_names: tuple[str, ...],
    dataset_name: str,
    model_name: str,
    variant: str,
) -> Path | None:
    """Plot native feature importance for compatible tree models."""
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return None
    order = np.argsort(values)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.barh(np.asarray(feature_names)[order], np.asarray(values)[order], color="teal")
    axis.set(
        title=f"GLCM feature importance - {model_name}\n{dataset_name} / {variant}",
        xlabel="Importance",
    )
    axis.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = PLOT_DIR / (
        f"{safe_name(dataset_name)}__{safe_name(model_name)}__"
        f"{safe_name(variant)}__feature_importance.png"
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path
