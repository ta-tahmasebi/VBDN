"""Memory-conscious BIG2015 conversion from archived bytes to PNG files."""

from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm import tqdm

IMAGE_WIDTHS = (
    (10_000, 32),
    (30_000, 64),
    (60_000, 128),
    (100_000, 256),
    (200_000, 384),
    (500_000, 512),
    (1_000_000, 768),
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ConversionResult:
    sample_id: str
    label: str
    status: str
    pixels: int = 0
    error: str = ""


def _find_file(root: Path, names: tuple[str, ...]) -> Path:
    """Find a required BIG2015 file case-insensitively."""
    matches = [path for path in root.rglob("*") if path.is_file() and path.name.lower() in names]
    if not matches:
        raise FileNotFoundError(f"Could not find {', '.join(names)} below {root}")
    return matches[0]


def _find_7z() -> str:
    """Resolve a supported 7-Zip executable from PATH."""
    executable = shutil.which("7z") or shutil.which("7zz")
    if not executable:
        raise RuntimeError("7z is required. Install p7zip-full (Linux) or 7-Zip.")
    return executable


def _load_labels(path: Path) -> dict[str, str]:
    """Read the official sample-to-class mapping."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"Id", "Class"}.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain Id and Class columns")
        return {row["Id"].strip(): row["Class"].strip() for row in reader}


def _archive_members(seven_zip: str, archive: Path) -> list[str]:
    """List byte-view members without extracting archive contents."""
    process = subprocess.run(
        [seven_zip, "l", "-slt", "-ba", str(archive)],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"7z could not list {archive}: {process.stderr.strip()}")
    members = []
    for line in process.stdout.splitlines():
        if line.startswith("Path = "):
            member = line.removeprefix("Path = ").strip()
            if member.lower().endswith(".bytes"):
                members.append(member)
    if not members:
        raise ValueError(f"No .bytes files were found in {archive}")
    return members


def _image_width(pixel_count: int) -> int:
    """Choose a standard malware-image width based on byte count."""
    for upper_bound, width in IMAGE_WIDTHS:
        if pixel_count < upper_bound:
            return width
    return 1024


def _stream_pixels(seven_zip: str, archive: Path, member: str) -> bytearray:
    """Decode one archived .bytes member through a 7z stdout pipe."""
    command = [seven_zip, "x", "-so", "-bd", "-y", "-mmt=1", str(archive), member]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    pixels = bytearray()
    for line in process.stdout:
        fields = line.split()
        for token in fields[1:17]:
            try:
                pixels.append(0 if token == b"??" else int(token, 16))
            except ValueError:
                continue
    stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(stderr.strip() or f"7z exited with status {return_code}")
    if not pixels:
        raise ValueError("the byte stream did not contain any pixels")
    return pixels


def _convert_member(
    seven_zip: str,
    archive: Path,
    member: str,
    labels: dict[str, str],
    output: Path,
    overwrite: bool,
) -> ConversionResult:
    """Convert one archive member atomically into its class folder."""
    sample_id = Path(member).stem
    label = labels.get(sample_id)
    if label is None:
        return ConversionResult(sample_id, "unknown", "failed", error="missing label")
    target = output / f"class_{label}" / f"{sample_id}.png"
    if target.exists() and not overwrite:
        return ConversionResult(sample_id, label, "skipped")

    try:
        pixels = _stream_pixels(seven_zip, archive, member)
        width = _image_width(len(pixels))
        height = math.ceil(len(pixels) / width)
        pixels.extend(b"\x00" * (width * height - len(pixels)))
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f".{os.getpid()}.tmp")
        Image.frombytes("L", (width, height), bytes(pixels)).save(temporary, format="PNG")
        temporary.replace(target)
        return ConversionResult(sample_id, label, "converted", width * height)
    except (OSError, RuntimeError, ValueError) as exc:
        return ConversionResult(sample_id, label, "failed", error=str(exc))


def _write_manifest(output: Path, results: list[ConversionResult]) -> Path:
    """Record resumable conversion status for every requested sample."""
    manifest = output / "conversion_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ConversionResult.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(result.__dict__ for result in results)
    return manifest


def _prepared_images_exist(output: Path) -> bool:
    """Detect an existing ImageFolder tree, including numeric BIG2015 classes."""
    class_folders = [path for path in output.iterdir() if path.is_dir()]
    populated = 0
    for folder in class_folders:
        if any(
            path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES for path in folder.iterdir()
        ):
            populated += 1
    return populated >= 2


def _select_members(
    members: list[str],
    labels: dict[str, str],
    limit: int | None,
    samples_per_class: int | None,
) -> list[str]:
    """Select either a global prefix or a balanced per-class archive sample."""
    if samples_per_class is None:
        return members[:limit] if limit is not None else members

    selected = []
    counts: Counter[str] = Counter()
    target_classes = set(labels.values())
    for member in members:
        label = labels[Path(member).stem]
        if counts[label] >= samples_per_class:
            continue
        selected.append(member)
        counts[label] += 1
        if all(counts[label] >= samples_per_class for label in target_classes):
            break
    missing = sorted(label for label in target_classes if counts[label] < samples_per_class)
    if missing:
        raise ValueError(f"Archive lacks {samples_per_class} samples for classes: {missing}")
    return selected


def prepare_big2015(
    source: str | Path,
    output: str | Path | None = None,
    workers: int = 4,
    overwrite: bool = False,
    limit: int | None = None,
    samples_per_class: int | None = None,
) -> Path:
    """Convert archive members independently without extracting train.7z."""
    source = Path(source).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"BIG2015 source folder does not exist: {source}")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    if samples_per_class is not None and samples_per_class < 1:
        raise ValueError("samples_per_class must be at least 1")
    if limit is not None and samples_per_class is not None:
        raise ValueError("limit and samples_per_class cannot be used together")

    output = Path(output).expanduser().resolve() if output else source / "images"
    output.mkdir(parents=True, exist_ok=True)
    if not overwrite and samples_per_class is None and _prepared_images_exist(output):
        print(f"Using existing BIG2015 image dataset: {output}")
        return output
    archive = _find_file(source, ("train.7z", "train.zip.7z"))
    label_file = _find_file(source, ("trainlabels.csv",))
    seven_zip = _find_7z()
    labels = _load_labels(label_file)
    members = [
        member for member in _archive_members(seven_zip, archive) if Path(member).stem in labels
    ]
    members = _select_members(members, labels, limit, samples_per_class)

    print(f"BIG2015 archive: {archive}")
    print(f"Streaming {len(members)} samples with {workers} workers into {output}")
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_convert_member, seven_zip, archive, member, labels, output, overwrite)
            for member in members
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="BIG2015 conversion"):
            results.append(future.result())

    results.sort(key=lambda result: result.sample_id)
    manifest = _write_manifest(output, results)
    counts = {
        status: sum(result.status == status for result in results)
        for status in ("converted", "skipped", "failed")
    }
    print(f"Conversion complete: {counts}; manifest: {manifest}")
    if counts["failed"]:
        print("Failed samples are recorded in the manifest and can be retried safely.")
    return output
