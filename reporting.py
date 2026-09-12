"""Metrics, predictions, and publication-ready experiment charts."""

from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_curve,
)

from config import CSV_DIR, PLOT_DIR, PREDICTION_DIR, REPORT_DIR

warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples.*",
    category=UserWarning,
    module="sklearn",
)


def safe_name(value: str) -> str:
    """Convert a label into a stable file name."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_").lower()


def _artifact_stem(dataset_name: str, model_name: str) -> str:
    """Join dataset and model names for artifact file paths."""
    return f"{safe_name(dataset_name)}__{safe_name(model_name)}"


def plot_confusion_matrices(cm: np.ndarray, classes: list[str], stem: str) -> Path:
    """Save raw and row-normalized confusion matrices side by side."""
    row_totals = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(
        cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0
    )
    size = max(8, min(18, len(classes) * 0.75))
    fig, axes = plt.subplots(1, 2, figsize=(size * 2, size))
    sns.heatmap(
        cm,
        annot=len(classes) <= 15,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        xticklabels=classes,
        yticklabels=classes,
    )
    sns.heatmap(
        normalized * 100,
        annot=len(classes) <= 15,
        fmt=".1f",
        cmap="Blues",
        ax=axes[1],
        xticklabels=classes,
        yticklabels=classes,
        vmin=0,
        vmax=100,
    )
    axes[0].set_title("Counts")
    axes[1].set_title("Row-normalized (%)")
    for axis in axes:
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
    fig.suptitle(stem.replace("__", " - ").replace("_", " ").title())
    fig.tight_layout()
    path = PLOT_DIR / f"{stem}__confusion_matrix.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_history(history: list[dict], dataset_name: str, model_name: str) -> Path | None:
    """Save the paper-style epoch accuracy and loss curves."""
    if not history:
        return None
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(frame["epoch"], frame["loss"], marker="o")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].plot(frame["epoch"], frame["accuracy"], marker="o", color="tab:green")
    axes[1].set(title="Training accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.suptitle(f"{model_name} on {dataset_name}")
    fig.tight_layout()
    path = PLOT_DIR / f"{_artifact_stem(dataset_name, model_name)}__learning_curve.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_per_class(report: pd.DataFrame, stem: str) -> Path:
    """Plot precision, recall, and F1 for each malware family."""
    per_class = report.loc[~report.index.isin(["accuracy", "macro avg", "weighted avg"])]
    metrics = [column for column in ("precision", "recall", "f1-score") if column in per_class]
    axis = per_class[metrics].plot(kind="bar", figsize=(max(9, len(per_class) * 0.8), 5))
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Per-class classification metrics")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(loc="lower right")
    plt.tight_layout()
    path = PLOT_DIR / f"{stem}__per_class_metrics.png"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def _plot_roc_pr(
    y_true: np.ndarray, scores: np.ndarray, classes: list[str], stem: str
) -> Path | None:
    """Plot one-vs-rest ROC and precision-recall curves."""
    if (
        scores.ndim != 2
        or scores.shape[1] != len(classes)
        or not np.isfinite(scores).all()
    ):
        return None
    one_hot = np.eye(len(classes), dtype=int)[y_true]
    valid = [index for index in range(len(classes)) if np.unique(one_hot[:, index]).size == 2]
    if not valid:
        return None

    legend_columns = min(3, max(1, math.ceil(len(valid) / 10)))
    legend_rows = math.ceil(len(valid) / legend_columns)
    legend_height = max(1.2, legend_rows * 0.28)
    fig = plt.figure(figsize=(16, 6 + legend_height), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(6, legend_height))
    axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]))
    legend_axes = (fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]))
    for index in valid:
        false_positive, true_positive, _ = roc_curve(one_hot[:, index], scores[:, index])
        precision, recall, _ = precision_recall_curve(one_hot[:, index], scores[:, index])
        axes[0].plot(
            false_positive,
            true_positive,
            label=f"{classes[index]} (AUC={auc(false_positive, true_positive):.3f})",
        )
        average_precision = average_precision_score(one_hot[:, index], scores[:, index])
        axes[1].plot(recall, precision, label=f"{classes[index]} (AP={average_precision:.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set(title="One-vs-rest ROC", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="One-vs-rest precision-recall", xlabel="Recall", ylabel="Precision")
    for axis, legend_axis in zip(axes, legend_axes, strict=True):
        axis.grid(alpha=0.3)
        handles, labels = axis.get_legend_handles_labels()
        legend_axis.axis("off")
        legend_axis.legend(
            handles,
            labels,
            loc="upper center",
            ncol=legend_columns,
            fontsize=8,
            frameon=True,
            columnspacing=1.0,
            handlelength=2.2,
        )
    fig.suptitle(stem.replace("__", " - ").replace("_", " ").title())
    path = PLOT_DIR / f"{stem}__roc_pr_curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_evaluation(
    y_true,
    y_pred,
    classes: list[str],
    dataset_name: str,
    model_name: str,
    sample_ids=None,
    scores: np.ndarray | None = None,
) -> dict:
    """Persist predictions, detailed metrics, and diagnostic plots."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = np.arange(len(classes))
    stem = _artifact_stem(dataset_name, model_name)
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    report = pd.DataFrame(report_dict).transpose()
    report.index.name = "class"
    report.to_csv(REPORT_DIR / f"{stem}__classification_report.csv")
    _plot_per_class(report, stem)
    plot_confusion_matrices(confusion_matrix(y_true, y_pred, labels=labels), classes, stem)

    predictions = pd.DataFrame(
        {
            "sample_id": np.arange(len(y_true)) if sample_ids is None else sample_ids,
            "true_index": y_true,
            "true_class": [classes[index] for index in y_true],
            "predicted_index": y_pred,
            "predicted_class": [classes[index] for index in y_pred],
            "correct": y_true == y_pred,
        }
    )
    if scores is not None and scores.ndim == 2 and scores.shape[1] == len(classes):
        for index, class_name in enumerate(classes):
            predictions[f"score_{safe_name(class_name)}"] = scores[:, index]
        _plot_roc_pr(y_true, scores, classes, stem)
    predictions.to_csv(PREDICTION_DIR / f"{stem}__predictions.csv", index=False)

    metrics = {
        "Accuracy (%)": accuracy_score(y_true, y_pred) * 100,
        "Balanced Accuracy (%)": balanced_accuracy_score(y_true, y_pred) * 100,
        "Macro Precision": report_dict["macro avg"]["precision"],
        "Macro Recall": report_dict["macro avg"]["recall"],
        "Macro F1": report_dict["macro avg"]["f1-score"],
        "Weighted F1": report_dict["weighted avg"]["f1-score"],
        "Matthews Correlation": matthews_corrcoef(y_true, y_pred),
        "Cohen Kappa": cohen_kappa_score(y_true, y_pred),
        "Test Samples": len(y_true),
    }
    (REPORT_DIR / f"{stem}__metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def save_results_csv(results: list[dict], csv_path: str | Path) -> None:
    """Write a stable tabular summary for one pipeline."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(path, index=False)
    print(f"Summary saved to {path}")


def save_comparison_plots(results: list[dict], prefix: str) -> None:
    """Compare accuracy and runtime across completed experiments."""
    if not results:
        return
    frame = pd.DataFrame(results).copy()
    if "Variant" in frame:
        values = frame["Variant"].fillna("").astype(str)
        variant = values.map(lambda value: f" / {value}" if value else "")
    else:
        variant = ""
    frame["Experiment"] = frame["Dataset"] + " / " + frame["Model"] + variant

    if (frame["Model"] == "ConvNet").any():
        reference_times = (
            frame.loc[frame["Model"] == "ConvNet"]
            .drop_duplicates("Dataset")
            .set_index("Dataset")["Eval Time (s)"]
        )
        frame["Inference Improvement vs ConvNet (%)"] = frame.apply(
            lambda row: (
                100
                * (row["Eval Time (s)"] - reference_times[row["Dataset"]])
                / row["Eval Time (s)"]
                if row["Model"] != "ConvNet"
                and row["Dataset"] in reference_times
                and row["Eval Time (s)"] > 0
                else np.nan
            ),
            axis=1,
        )
    frame.to_csv(CSV_DIR / f"{safe_name(prefix)}__summary.csv", index=False)

    for column, suffix, title in (
        ("Accuracy (%)", "accuracy_comparison", "Model accuracy comparison"),
        ("Train Time (s)", "training_time_comparison", "Training time comparison"),
        (
            "Inference Improvement vs ConvNet (%)",
            "inference_improvement",
            "Inference-time improvement relative to ConvNet",
        ),
    ):
        if column not in frame:
            continue
        available = frame.dropna(subset=[column]).sort_values(column, ascending=False)
        if available.empty:
            continue
        fig, axis = plt.subplots(figsize=(11, max(5, len(available) * 0.42)))
        sns.barplot(data=available, x=column, y="Experiment", ax=axis, color="steelblue")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"{safe_name(prefix)}__{suffix}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
