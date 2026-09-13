"""Dataset discovery and reproducible splitting utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from config import KNOWN_DATASETS, SEED

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _known_name(value: str) -> str | None:
    """Normalize a case-insensitive built-in dataset name."""
    names = {name.casefold(): name for name in (*KNOWN_DATASETS, "BIG2015")}
    return names.get(value.casefold())


def _cached_kaggle_root(slug: str) -> Path | None:
    """Return the newest version from kagglehub's default local download cache."""
    versions = Path.home() / ".cache" / "kagglehub" / "datasets" / slug / "versions"
    if not versions.is_dir():
        return None
    candidates = sorted(
        (path for path in versions.iterdir() if path.is_dir()),
        key=lambda path: (
            (
                0,
                int(path.name),
            )
            if path.name.isdigit()
            else (1, path.name)
        ),
    )
    return candidates[-1] if candidates else None


def _is_class_folder(path: Path) -> bool:
    """Check whether a directory directly contains supported images."""
    return any(item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES for item in path.iterdir())


def _discover_imagefolder_root(root: Path) -> Path:
    """Find the strongest ImageFolder candidate inside an unknown Kaggle layout."""
    candidates = []
    for directory in (root, *root.rglob("*")):
        if not directory.is_dir():
            continue
        class_folders = [
            child for child in directory.iterdir() if child.is_dir() and _is_class_folder(child)
        ]
        if len(class_folders) >= 2:
            image_count = sum(
                1
                for folder in class_folders
                for item in folder.iterdir()
                if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
            )
            candidates.append((len(class_folders), image_count, directory))
    if not candidates:
        raise FileNotFoundError(f"No ImageFolder-compatible directory found below {root}")
    return max(candidates, key=lambda candidate: candidate[:2])[2]


def get_dataset_paths(dataset_args, download: bool = True, args=None) -> list[tuple[str, str]]:
    """Resolve Kaggle datasets, BIG2015, and local ImageFolder paths."""
    datasets = []
    for raw_item in dataset_args:
        item = str(raw_item)
        known = _known_name(item)
        if known == "BIG2015":
            if args is None:
                raise ValueError("BIG2015 requires CLI conversion options")
            from big2015 import prepare_big2015

            path = prepare_big2015(
                args.big2015_path,
                args.big2015_images,
                args.big2015_workers,
                args.rebuild_big2015,
                args.big2015_limit,
                args.big2015_samples_per_class,
            )
            datasets.append((known, str(path)))
            continue

        if known in KNOWN_DATASETS:
            dataset_config = KNOWN_DATASETS[known]
            if download:
                print(f"Resolving {known} from Kaggle...")
                try:
                    import kagglehub
                except ImportError as exc:
                    root = _cached_kaggle_root(dataset_config["kaggle"])
                    if root is None:
                        raise ImportError(
                            "Kaggle dependencies are incompatible and no cached copy exists. "
                            "Reinstall requirements, or use --no-download."
                        ) from exc
                    print(f"Using cached Kaggle data: {root}")
                else:
                    root = Path(kagglehub.dataset_download(dataset_config["kaggle"]))
            else:
                root = _cached_kaggle_root(dataset_config["kaggle"])
                if root is None:
                    expected = (
                        Path.home()
                        / ".cache"
                        / "kagglehub"
                        / "datasets"
                        / dataset_config["kaggle"]
                        / "versions"
                    )
                    raise FileNotFoundError(
                        f"Dataset '{known}' is not available in the Kaggle cache at "
                        f"{expected}. Run once with --download or pass a dataset path."
                    )
                print(f"Using cached Kaggle data for {known}: {root}")
            subpath = dataset_config["subpath"]
            path = root / subpath if subpath else _discover_imagefolder_root(root)
            if not path.is_dir():
                raise FileNotFoundError(
                    f"Dataset '{known}' was not found inside its Kaggle cache at {path}. "
                    "Run once with --download or pass a dataset path."
                )
            datasets.append((known, str(path.resolve())))
            continue

        path = Path(item).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")
        datasets.append((path.resolve().name, str(path.resolve())))
    return datasets


def stratified_indices(targets, train_fraction: float) -> tuple[list[int], list[int]]:
    """Return deterministic train/test indices while preserving class ratios."""
    indices = np.arange(len(targets))
    try:
        train, test = train_test_split(
            indices,
            train_size=train_fraction,
            random_state=SEED,
            stratify=np.asarray(targets),
        )
    except ValueError as exc:
        raise ValueError(
            "A stratified split needs at least two samples per class and enough "
            "samples in both splits."
        ) from exc
    return train.tolist(), test.tolist()
