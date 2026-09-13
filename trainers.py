"""CNN training pipelines."""

from __future__ import annotations

import gc
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from config import CSV_DIR, MODEL_DIR, SEED
from data_utils import get_dataset_paths, stratified_indices
from models import ConvNet, get_pretrained_model
from reporting import (
    plot_training_history,
    safe_name,
    save_comparison_plots,
    save_evaluation,
    save_results_csv,
)
from samplers import DatasetSampler, TransformWrapper
from visualizations import (
    save_augmentation_gallery,
    save_balance_comparison,
    save_dataset_gallery,
)


def _save_dataset_overview(
    args,
    dataset_name: str,
    dataset_path: str,
    excluded_classes: tuple[str, ...] = (),
) -> None:
    """Save representative images and sampler balance for CNN-only runs."""
    dataset = _image_folder(
        dataset_path,
        excluded_classes,
        max_samples_per_class=args.max_samples_per_class,
    )
    paths = np.asarray([path for path, _ in dataset.samples])
    labels = np.asarray(dataset.targets)
    train_indices, _ = stratified_indices(labels, args.train_split)
    train_labels = labels[train_indices]
    maximum = int(np.bincount(train_labels).max())
    balanced_labels = np.concatenate(
        [np.full(maximum, class_index, dtype=int) for class_index in range(len(dataset.classes))]
    )
    save_dataset_gallery(paths, labels, dataset.classes, dataset_name)
    save_augmentation_gallery(paths, labels, dataset.classes, dataset_name)
    save_balance_comparison(train_labels, balanced_labels, dataset.classes, dataset_name)


def _image_folder(
    dataset_path: str,
    excluded_classes: tuple[str, ...] = (),
    max_samples_per_class: int | None = None,
) -> ImageFolder:
    """Load ImageFolder and optionally remove classes with contiguous remapping."""
    dataset = ImageFolder(root=dataset_path)
    excluded = {name.casefold() for name in excluded_classes}
    if excluded:
        kept_classes = [name for name in dataset.classes if name.casefold() not in excluded]
        if len(kept_classes) == len(dataset.classes):
            raise ValueError(
                f"None of {sorted(excluded_classes)} exists in {dataset_path}; "
                f"available classes: {dataset.classes}"
            )
        old_to_new = {dataset.class_to_idx[name]: index for index, name in enumerate(kept_classes)}
        dataset.samples = [
            (path, old_to_new[label]) for path, label in dataset.samples if label in old_to_new
        ]
        dataset.classes = kept_classes
        dataset.class_to_idx = {name: index for index, name in enumerate(kept_classes)}
    if max_samples_per_class is not None:
        counts = {index: 0 for index in range(len(dataset.classes))}
        limited = []
        for path, label in dataset.samples:
            if counts[label] < max_samples_per_class:
                limited.append((path, label))
                counts[label] += 1
        dataset.samples = limited
    dataset.imgs = dataset.samples
    dataset.targets = [label for _, label in dataset.samples]
    return dataset


def _make_loaders(
    args,
    dataset_path: str,
    pretrained: bool,
    excluded_classes: tuple[str, ...] = (),
):
    """Create stratified, balanced train and deterministic test loaders."""
    normalization = (
        ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) if pretrained else ([0.5] * 3, [0.5] * 3)
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(*normalization),
        ]
    )
    base = _image_folder(
        dataset_path,
        excluded_classes,
        max_samples_per_class=args.max_samples_per_class,
    )
    train_indices, test_indices = stratified_indices(base.targets, args.train_split)
    train_raw = Subset(base, train_indices)
    test_raw = Subset(base, test_indices)
    train_data = TransformWrapper(train_raw, train_transform)
    test_data = TransformWrapper(test_raw, test_transform)
    sampler = DatasetSampler(train_data, [base.targets[index] for index in train_indices])
    shared_options = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    training_batch_size = args.pretrained_batch_size if pretrained else args.batch_size
    train_loader = DataLoader(
        train_data, batch_size=training_batch_size, sampler=sampler, **shared_options
    )
    test_loader = DataLoader(
        test_data, batch_size=args.test_batch_size, shuffle=False, **shared_options
    )
    sample_ids = [base.samples[index][0] for index in test_indices]
    return train_loader, test_loader, base.classes, sample_ids, len(train_indices)


def _train_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    """Optimize one epoch and return mean loss and sampled accuracy."""
    model.train()
    loss_total = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_total += loss.item() * labels.size(0)
        correct += outputs.argmax(dim=1).eq(labels).sum().item()
        count += labels.size(0)
    return loss_total / count, 100.0 * correct / count


def _evaluate(model, loader, device):
    """Collect labels, predictions, and class probabilities."""
    model.eval()
    predictions, labels, scores = [], [], []
    with torch.inference_mode():
        for images, batch_labels in loader:
            probabilities = torch.softmax(model(images.to(device, non_blocking=True)), dim=1)
            scores.append(probabilities.cpu().numpy())
            predictions.extend(probabilities.argmax(dim=1).cpu().numpy())
            labels.extend(batch_labels.numpy())
    return np.asarray(labels), np.asarray(predictions), np.concatenate(scores)


def _train_cnn(
    args,
    model,
    train_loader,
    test_loader,
    classes,
    sample_ids,
    train_samples,
    dataset_name,
    model_name,
    device,
) -> dict:
    """Train or load one CNN, evaluate it, and persist all artifacts."""
    stem = f"{safe_name(dataset_name)}__{safe_name(model_name)}__epochs_{args.epochs}"
    model_path = MODEL_DIR / f"{stem}.pth"
    history = []
    train_time = None
    trained = not args.save_models or not model_path.exists() or args.force_retrain

    if trained:
        print(f"Training {model_name} on {dataset_name} ({device})...")
        criterion = nn.CrossEntropyLoss()
        learning_rate = args.lr if model_name == "ConvNet" else args.pretrained_lr
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=args.momentum)
        started = time.perf_counter()
        epochs = tqdm(range(1, args.epochs + 1), desc=f"{model_name} / {dataset_name}")
        for epoch in epochs:
            loss, accuracy = _train_epoch(model, train_loader, criterion, optimizer, device)
            history.append({"epoch": epoch, "loss": loss, "accuracy": accuracy})
            epochs.set_postfix(loss=f"{loss:.4f}", accuracy=f"{accuracy:.2f}%")
        train_time = time.perf_counter() - started
        if args.save_models:
            torch.save(model.state_dict(), model_path)
        plot_training_history(history, dataset_name, model_name)
    else:
        print(f"Loading existing checkpoint: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    started = time.perf_counter()
    y_true, y_pred, scores = _evaluate(model, test_loader, device)
    eval_time = time.perf_counter() - started
    metrics = save_evaluation(y_true, y_pred, classes, dataset_name, model_name, sample_ids, scores)
    result = {
        "Dataset": dataset_name,
        "Model": model_name,
        "Epochs": args.epochs,
        **metrics,
        "Train Samples": train_samples,
        "Train Time (s)": train_time,
        "Eval Time (s)": eval_time,
        "Checkpoint": str(model_path) if args.save_models else "",
        "Trained This Run": trained,
    }
    print(
        f"{model_name} / {dataset_name}: accuracy={metrics['Accuracy (%)']:.2f}%, "
        f"macro-F1={metrics['Macro F1']:.4f}, eval={eval_time:.2f}s"
    )
    return result


def _seed_everything() -> None:
    """Seed CPU and GPU random generators for reproducible experiments."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _run_cnn_models(args, model_names: list[str] | None) -> list[dict]:
    """Run the requested CNN family over every resolved dataset."""
    _seed_everything()
    datasets = get_dataset_paths(args.datasets, download=args.download, args=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    visualized = set()
    names = model_names or ["ConvNet"]
    for model_name in names:
        for dataset_name, dataset_path in datasets:
            if dataset_name not in visualized:
                _save_dataset_overview(args, dataset_name, dataset_path)
                visualized.add(dataset_name)
            variants = [(dataset_name, ())]
            if (
                model_names is None
                and args.malevis_without_other
                and dataset_name.casefold() == "malevis"
            ):
                variants.append(("Malevis_without_Other", ("Other", "Others")))
            for variant_name, excluded_classes in variants:
                if excluded_classes and variant_name not in visualized:
                    _save_dataset_overview(args, variant_name, dataset_path, excluded_classes)
                    visualized.add(variant_name)
                loaders = _make_loaders(
                    args,
                    dataset_path,
                    pretrained=model_names is not None,
                    excluded_classes=excluded_classes,
                )
                train_loader, test_loader, classes, sample_ids, train_samples = loaders
                model = (
                    get_pretrained_model(model_name, len(classes))
                    if model_names is not None
                    else ConvNet(len(classes))
                ).to(device)
                result = _train_cnn(
                    args,
                    model,
                    train_loader,
                    test_loader,
                    classes,
                    sample_ids,
                    train_samples,
                    variant_name,
                    model_name,
                    device,
                )
                results.append(result)
                del model, train_loader, test_loader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return results


def run_convnet(args) -> list[dict]:
    """Run VBDN, including the paper's MaleVis no-Other experiment."""
    results = _run_cnn_models(args, None)
    save_results_csv(results, CSV_DIR / "convnet_results.csv")
    save_comparison_plots(results, "convnet")
    return results


def run_pretrained(args) -> list[dict]:
    """Run ImageNet baselines with the paper's comparison batch size."""
    results = _run_cnn_models(args, list(args.models))
    save_results_csv(results, CSV_DIR / "pretrained_results.csv")
    save_comparison_plots(results, "pretrained")
    return results
