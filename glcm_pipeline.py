"""Raw and balanced-augmented GLCM experiment pipelines."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from config import CSV_DIR, SEED
from data_utils import get_dataset_paths
from reporting import save_comparison_plots, save_evaluation, save_results_csv
from visualizations import (
    save_augmentation_gallery,
    save_balance_comparison,
    save_dataset_gallery,
    save_feature_importance,
    save_glcm_gallery,
)

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError as exc:
    graycomatrix = graycoprops = None
    SKIMAGE_IMPORT_ERROR = exc
else:
    SKIMAGE_IMPORT_ERROR = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
GLCM_PROPERTIES = (
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
    "ASM",
)


def _augment_image(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Apply the paper's random flip and 20-degree rotation policy."""
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return image.rotate(float(rng.uniform(-20, 20)), resample=Image.Resampling.BILINEAR)


def extract_glcm_features(
    image_path: str | Path,
    target_size: tuple[int, int] = (128, 128),
    levels: int = 8,
    distances: tuple[int, ...] = (1,),
    angles: tuple[float, ...] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    augment: bool = False,
    seed: int = SEED,
) -> np.ndarray:
    """Extract six direction-averaged GLCM texture properties."""
    if SKIMAGE_IMPORT_ERROR:
        raise ImportError("scikit-image is required for GLCM mode") from SKIMAGE_IMPORT_ERROR
    with Image.open(image_path) as source:
        image = source.convert("L")
        if augment:
            image = _augment_image(image, np.random.default_rng(seed))
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        quantized = np.asarray(image, dtype=np.uint8) // (256 // levels)
    matrix = graycomatrix(
        quantized,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )
    return np.asarray([graycoprops(matrix, name).mean() for name in GLCM_PROPERTIES])


def index_image_dataset(dataset_path: str | Path, max_samples_per_class: int | None = None):
    """Index ImageFolder-compatible files without decoding all images."""
    root = Path(dataset_path)
    classes = sorted(path.name for path in root.iterdir() if path.is_dir())
    if not classes:
        raise ValueError(f"No class folders found in {root}")
    paths, labels = [], []
    for label, class_name in enumerate(classes):
        class_paths = sorted(
            path
            for path in (root / class_name).rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if max_samples_per_class is not None:
            class_paths = class_paths[:max_samples_per_class]
        print(f"{class_name}: {len(class_paths)} images")
        paths.extend(class_paths)
        labels.extend([label] * len(class_paths))
    if not paths:
        raise ValueError(f"No supported images found in {root}")
    return np.asarray(paths, dtype=object), np.asarray(labels, dtype=int), classes


def _extract_feature_matrix(
    paths, labels, augment: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features while retaining only successfully decoded samples."""
    features, valid_labels, valid_paths = [], [], []
    description = "GLCM augmented" if augment else "GLCM raw"
    samples = zip(paths, labels, strict=True)
    for index, (path, label) in enumerate(tqdm(samples, total=len(paths), desc=description)):
        try:
            features.append(extract_glcm_features(path, augment=augment, seed=SEED + index))
            valid_labels.append(label)
            valid_paths.append(str(path))
        except (OSError, ValueError) as exc:
            print(f"Skipping unreadable image {path}: {exc}")
    if not features:
        raise ValueError("No GLCM features could be extracted")
    return np.asarray(features), np.asarray(valid_labels), np.asarray(valid_paths)


def _balanced_training_index(paths, labels) -> tuple[np.ndarray, np.ndarray]:
    """Oversample each training class to the majority count without copying files."""
    rng = np.random.default_rng(SEED)
    class_counts = np.bincount(labels)
    target_count = int(class_counts.max())
    sampled = []
    for class_index in range(len(class_counts)):
        candidates = np.flatnonzero(labels == class_index)
        sampled.extend(rng.choice(candidates, size=target_count, replace=True))
    sampled = np.asarray(sampled)
    rng.shuffle(sampled)
    return paths[sampled], labels[sampled]


def _classifiers(args) -> dict:
    """Build fresh classical classifiers matching the paper comparison."""
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "GaussianNB": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=args.knn_neighbors),
        "DecisionTree": DecisionTreeClassifier(random_state=SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=args.rf_estimators,
            random_state=SEED,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=args.gbdt_estimators, random_state=SEED
        ),
        "SVM": SVC(kernel="rbf", probability=True, random_state=SEED),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=SEED),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=args.xgb_estimators,
            random_state=SEED,
            eval_metric="mlogloss",
            verbosity=0,
            n_jobs=-1,
        )
    else:
        print("XGBoost is unavailable and will be skipped.")
    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=args.gbdt_estimators,
            random_state=SEED,
            verbosity=-1,
            n_jobs=-1,
        )
    else:
        print("LightGBM is unavailable and will be skipped.")
    return models


def _fit_variant(
    args,
    dataset_name: str,
    variant: str,
    classes: list[str],
    train_features,
    train_labels,
    test_features,
    test_labels,
    test_paths,
) -> list[dict]:
    """Fit and report every classical model for one GLCM variant."""
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_features)
    x_test = scaler.transform(test_features)
    results = []
    for model_name, model in _classifiers(args).items():
        started = time.perf_counter()
        model.fit(x_train, train_labels)
        train_time = time.perf_counter() - started
        started = time.perf_counter()
        predictions = model.predict(x_test)
        scores = model.predict_proba(x_test) if hasattr(model, "predict_proba") else None
        eval_time = time.perf_counter() - started
        report_name = f"{model_name}__{variant}"
        metrics = save_evaluation(
            test_labels,
            predictions,
            classes,
            dataset_name,
            report_name,
            test_paths,
            scores,
        )
        save_feature_importance(model, GLCM_PROPERTIES, dataset_name, model_name, variant)
        results.append(
            {
                "Dataset": dataset_name,
                "Model": model_name,
                "Variant": variant,
                **metrics,
                "Train Samples": len(train_labels),
                "Train Time (s)": train_time,
                "Eval Time (s)": eval_time,
                "Checkpoint": "",
                "Trained This Run": True,
            }
        )
        print(
            f"{model_name} / {dataset_name} / {variant}: "
            f"accuracy={metrics['Accuracy (%)']:.2f}%, macro-F1={metrics['Macro F1']:.4f}"
        )
    return results


def run_glcm(args) -> list[dict]:
    """Compare raw GLCM with balanced and augmented GLCM experiments."""
    if SKIMAGE_IMPORT_ERROR:
        raise ImportError("scikit-image is required for GLCM mode") from SKIMAGE_IMPORT_ERROR
    datasets = get_dataset_paths(args.datasets, download=args.download, args=args)
    results = []
    for dataset_name, dataset_path in datasets:
        print(f"Indexing GLCM dataset: {dataset_name}")
        paths, labels, classes = index_image_dataset(dataset_path, args.max_samples_per_class)
        save_dataset_gallery(paths, labels, classes, dataset_name)
        save_augmentation_gallery(paths, labels, classes, dataset_name)
        save_glcm_gallery(paths, labels, classes, dataset_name)
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            paths,
            labels,
            train_size=args.train_split,
            random_state=SEED,
            stratify=labels,
        )
        test_features, test_labels, test_paths = _extract_feature_matrix(
            test_paths, test_labels, augment=False
        )

        if "raw" in args.glcm_variants:
            raw_features, raw_labels, _ = _extract_feature_matrix(
                train_paths, train_labels, augment=False
            )
            results.extend(
                _fit_variant(
                    args,
                    dataset_name,
                    "raw",
                    classes,
                    raw_features,
                    raw_labels,
                    test_features,
                    test_labels,
                    test_paths,
                )
            )

        if "balanced-augmented" in args.glcm_variants:
            balanced_paths, balanced_labels = _balanced_training_index(train_paths, train_labels)
            save_balance_comparison(train_labels, balanced_labels, classes, dataset_name)
            balanced_features, balanced_labels, _ = _extract_feature_matrix(
                balanced_paths, balanced_labels, augment=True
            )
            results.extend(
                _fit_variant(
                    args,
                    dataset_name,
                    "balanced-augmented",
                    classes,
                    balanced_features,
                    balanced_labels,
                    test_features,
                    test_labels,
                    test_paths,
                )
            )

    save_results_csv(results, CSV_DIR / "glcm_results.csv")
    save_comparison_plots(results, "glcm_variants")
    return results
