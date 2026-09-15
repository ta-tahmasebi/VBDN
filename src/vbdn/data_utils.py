"""Dataset discovery and reproducible splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .config import KNOWN_DATASETS, SEED

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ResolvedDataset:
    """A dataset's paper-aligned training and evaluation sources."""

    name: str
    train_path: str
    test_path: str | None = None
    train_size: int | float | None = None


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


def resolve_dataset_splits(  # noqa: PLR0912
    dataset_args, download: bool = True, args=None
) -> list[ResolvedDataset]:
    """Resolve paper-aligned train/test sources without mixing official test data."""
    datasets = []
    for raw_item in dataset_args:
        item = str(raw_item)
        known = _known_name(item)
        if known == "BIG2015":
            if args is None:
                raise ValueError("BIG2015 requires CLI conversion options")
            from .big2015 import prepare_big2015

            path = prepare_big2015(
                args.big2015_path,
                args.big2015_images,
                args.big2015_workers,
                args.rebuild_big2015,
                args.big2015_limit,
                args.big2015_samples_per_class,
            )
            datasets.append(
                ResolvedDataset(
                    known,
                    str(path),
                    train_size=args.train_split,
                )
            )
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
            subpath = dataset_config.get("subpath")
            train_path = root / subpath if subpath else _discover_imagefolder_root(root)
            if not train_path.is_dir():
                raise FileNotFoundError(
                    f"Dataset '{known}' was not found inside its Kaggle cache at {train_path}. "
                    "Run once with --download or pass a dataset path."
                )
            test_subpath = dataset_config.get("test_subpath")
            test_path = root / test_subpath if test_subpath else None
            if test_path is not None and not test_path.is_dir():
                raise FileNotFoundError(
                    f"Dataset '{known}' test split was not found at {test_path}."
                )
            datasets.append(
                ResolvedDataset(
                    known,
                    str(train_path.resolve()),
                    str(test_path.resolve()) if test_path is not None else None,
                    dataset_config.get(
                        "paper_train_size",
                        args.train_split if args is not None else 0.7,
                    ),
                )
            )
            continue

        path = Path(item).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")
        datasets.append(
            ResolvedDataset(
                path.resolve().name,
                str(path.resolve()),
                train_size=args.train_split if args is not None else 0.7,
            )
        )
    return datasets


def get_dataset_paths(dataset_args, download: bool = True, args=None) -> list[tuple[str, str]]:
    """Backward-compatible resolver returning each dataset's training source."""
    return [
        (dataset.name, dataset.train_path)
        for dataset in resolve_dataset_splits(dataset_args, download=download, args=args)
    ]


def stratified_indices(targets, train_fraction: int | float) -> tuple[list[int], list[int]]:
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


def effective_train_size(
    configured_size: int | float | None,
    sample_count: int,
    fallback_fraction: float,
) -> int | float:
    """Use an exact paper split unless a deliberately limited smoke set is smaller."""
    if isinstance(configured_size, int) and configured_size >= sample_count:
        return fallback_fraction
    return configured_size or fallback_fraction
