#!/usr/bin/env python3
"""Convert frame-level speech datasets to the DFG NPZ schema.

The training code expects:

    image: float array with shape (256, 256, 3)
    label: uint8 array with shape (256, 256), values in [0, num_classes)

This script converts:

    <subject>/<subject>_frame<number>_img.npy
    <subject>/<subject>_frame<number>_mask.npy

or unlabeled frames:

    <subject>/<subject>_..._frame<number>.npy

into archives named "<subject>_<frame>.npz". Unlabeled frames receive an
all-zero placeholder mask because the repository's dataloader always reads a
`label` array, including during target feature aggregation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_SOURCE = Path(
    "/home/shared/datasets/speech_preproc/BARTS_preprocessed_4train"
)
DEFAULT_TARGET_TRAIN = Path(
    "/home/shared/datasets/speech_preproc/USC75_preprocessed_4train"
)
DEFAULT_TARGET_TEST = Path(
    "/home/shared/datasets/speech_preproc/USC16_preprocessed_4test"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/ronan/Documents/GitHub/DFG/data/speech_preproc"
)

FRAME_RE = re.compile(r"frame[_-]?(\d+)", re.IGNORECASE)
SAFE_TOKEN_RE = re.compile(r"[^A-Za-z0-9_.-]+")
NUM_CLASSES = 5


@dataclass(frozen=True)
class Frame:
    image_path: Path
    mask_path: Path | None
    subject: str
    frame_id: str
    output_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess BARTS/USC frame data for DFG training."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--target-train-root", type=Path, default=DEFAULT_TARGET_TRAIN
    )
    parser.add_argument(
        "--target-test-root", type=Path, default=DEFAULT_TARGET_TEST
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument("--source-site", default="barts")
    parser.add_argument("--target-site", default="USC")
    parser.add_argument(
        "--source-val-subjects",
        nargs="+",
        default=["Subject5"],
        help=(
            "BARTS subjects assigned to the source validation split. "
            "All other subjects are used for source training. Default: Subject5"
        ),
    )
    parser.add_argument(
        "--image-min",
        type=float,
        default=0.0,
        help="Fixed lower intensity bound before normalization (default: 0).",
    )
    parser.add_argument(
        "--image-max",
        type=float,
        default=255.0,
        help="Fixed upper intensity bound before normalization (default: 255).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Required height and width of input images (default: 256).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["all", "source", "target-train", "target-test"],
        default=["all"],
        help="Datasets to convert (default: all).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel writer processes (default: min(8, CPU count)).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum frames per dataset; useful for a smoke test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output NPZ files.",
    )
    parser.add_argument(
        "--uncompressed",
        action="store_true",
        help="Use np.savez instead of the default np.savez_compressed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and count frames without writing output files.",
    )
    args = parser.parse_args()
    if args.image_max <= args.image_min:
        parser.error("--image-max must be greater than --image-min")
    if args.size <= 0:
        parser.error("--size must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def safe_token(value: str) -> str:
    token = SAFE_TOKEN_RE.sub("_", value).strip("._")
    return token or "sample"


def frame_identity(path: Path) -> tuple[str, str]:
    subject = safe_token(path.parent.name)
    match = FRAME_RE.search(path.name)
    if match:
        frame_id = f"{int(match.group(1)):05d}"
    else:
        stem = path.name
        for suffix in (
            "_resampled_img.npy",
            "_resampled_mask.npy",
            "_img.npy",
            "_mask.npy",
            ".npy",
        ):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        frame_id = safe_token(stem)
    return subject, frame_id


def discover_paired(root: Path) -> tuple[list[Path], list[Path]]:
    image_paths = sorted(root.rglob("*_img.npy"))
    if not image_paths:
        return [], []

    mask_paths = []
    missing = []
    for image_path in image_paths:
        mask_path = image_path.with_name(image_path.name[:-8] + "_mask.npy")
        if mask_path.is_file():
            mask_paths.append(mask_path)
        else:
            missing.append(mask_path)
    if missing:
        preview = "\n".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} image files have no matching mask under {root}:\n"
            f"{preview}"
        )
    return image_paths, mask_paths


def discover_unlabeled(root: Path) -> list[Path]:
    paths = sorted(root.rglob("*.npy"))
    paired_masks = [path for path in paths if path.name.endswith("_mask.npy")]
    if paired_masks:
        raise ValueError(
            f"Found mask files without the expected *_img.npy convention "
            f"under {root}"
        )
    return paths


def build_frames(
    root: Path,
    output_dir: Path,
    *,
    require_labels: bool,
    allow_unlabeled: bool,
) -> tuple[list[Frame], bool]:
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {root}")

    image_paths, mask_paths = discover_paired(root)
    labeled = bool(image_paths)
    if not labeled:
        if require_labels:
            raise FileNotFoundError(
                f"No *_img.npy/*_mask.npy pairs found under {root}"
            )
        if not allow_unlabeled:
            raise ValueError(f"No input frames found under {root}")
        image_paths = discover_unlabeled(root)
        mask_paths = [None] * len(image_paths)

    if not image_paths:
        raise ValueError(f"No input .npy frames found under {root}")

    frames = []
    seen: dict[str, Path] = {}
    for image_path, mask_path in zip(image_paths, mask_paths):
        subject, frame_id = frame_identity(image_path)
        stem = f"{subject}_{frame_id}"
        if stem in seen:
            raise ValueError(
                f"Output name collision for {stem}: {seen[stem]} and "
                f"{image_path}"
            )
        seen[stem] = image_path
        frames.append(
            Frame(
                image_path=image_path,
                mask_path=mask_path,
                subject=subject,
                frame_id=frame_id,
                output_path=output_dir / f"{stem}.npz",
            )
        )
    return frames, labeled


def load_image(
    path: Path, expected_size: int, image_min: float, image_max: float
) -> np.ndarray:
    image = np.load(path, allow_pickle=False)
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3 or image.shape[-1] not in (1, 3):
        raise ValueError(
            f"Expected image shape (H, W), (H, W, 1), or (H, W, 3); "
            f"got {image.shape} for {path}"
        )
    if image.shape[:2] != (expected_size, expected_size):
        raise ValueError(
            f"Expected a {expected_size}x{expected_size} image, got "
            f"{image.shape} for {path}"
        )
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError(f"Image dtype must be numeric, got {image.dtype}: {path}")

    image = image.astype(np.float32, copy=False)
    image = (np.clip(image, image_min, image_max) - image_min) / (
        image_max - image_min
    )
    if not np.isfinite(image).all():
        raise ValueError(f"Image contains NaN or infinite values: {path}")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return np.ascontiguousarray(image, dtype=np.float32)


def load_label(path: Path | None, shape: tuple[int, int], num_classes: int) -> np.ndarray:
    if path is None:
        return np.zeros(shape, dtype=np.uint8)

    label = np.load(path, allow_pickle=False)
    if label.shape != shape:
        raise ValueError(
            f"Mask shape {label.shape} does not match image shape {shape}: {path}"
        )
    if not np.issubdtype(label.dtype, np.integer):
        rounded = np.rint(label)
        if not np.allclose(label, rounded):
            raise ValueError(f"Mask contains non-integer labels: {path}")
        label = rounded

    label_min = int(label.min()) if label.size else 0
    label_max = int(label.max()) if label.size else 0
    if label_min < 0 or label_max >= num_classes:
        raise ValueError(
            f"Mask values must be in [0, {num_classes - 1}], got "
            f"[{label_min}, {label_max}] for {path}"
        )
    return np.ascontiguousarray(label, dtype=np.uint8)


def write_frame(
    frame: Frame,
    *,
    expected_size: int,
    num_classes: int,
    image_min: float,
    image_max: float,
    overwrite: bool,
    uncompressed: bool,
) -> str:
    if frame.output_path.exists() and not overwrite:
        return "skipped"

    frame.output_path.parent.mkdir(parents=True, exist_ok=True)
    image = load_image(
        frame.image_path, expected_size, image_min=image_min, image_max=image_max
    )
    label = load_label(frame.mask_path, image.shape[:2], num_classes)

    temp_path = frame.output_path.with_name(
        f".{frame.output_path.name}.{os.getpid()}.tmp.npz"
    )
    saver = np.savez if uncompressed else np.savez_compressed
    try:
        saver(temp_path, image=image, label=label)
        os.replace(temp_path, frame.output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return "written"


def selected_datasets(names: Iterable[str]) -> set[str]:
    selected = set(names)
    if "all" in selected:
        return {"source", "target-train", "target-test"}
    return selected


def assign_source_splits(
    frames: list[Frame], validation_subjects: list[str]
) -> tuple[list[Frame], list[Frame], list[str]]:
    available = sorted({frame.subject for frame in frames})
    available_lookup = {subject.lower(): subject for subject in available}
    requested = [safe_token(subject) for subject in validation_subjects]
    resolved = []
    missing = []
    for subject in requested:
        match = available_lookup.get(subject.lower())
        if match is None:
            missing.append(subject)
        elif match not in resolved:
            resolved.append(match)
    if missing:
        raise ValueError(
            f"Unknown source validation subjects: {missing}. "
            f"Available subjects: {available}"
        )

    validation = {subject.lower() for subject in resolved}
    train_frames = [
        frame for frame in frames if frame.subject.lower() not in validation
    ]
    val_frames = [
        frame for frame in frames if frame.subject.lower() in validation
    ]
    if not train_frames:
        raise ValueError("The source validation split leaves no training frames")
    if not val_frames:
        raise ValueError("The source validation split contains no frames")
    return train_frames, val_frames, resolved


def move_frames_to_dir(
    frames: list[Frame], output_dir: Path
) -> list[Frame]:
    return [
        Frame(
            image_path=frame.image_path,
            mask_path=frame.mask_path,
            subject=frame.subject,
            frame_id=frame.frame_id,
            output_path=output_dir / frame.output_path.name,
        )
        for frame in frames
    ]


def write_manifest(
    output_root: Path,
    manifest: dict,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    temp_path = output_root / ".manifest.json.tmp"
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    os.replace(temp_path, manifest_path)


def main() -> int:
    args = parse_args()
    selected = selected_datasets(args.datasets)

    datasets: dict[str, list[Frame]] = {}
    labeled: dict[str, bool] = {}
    metadata: dict[str, dict] = {}

    if "source" in selected:
        source_frames, source_labeled = build_frames(
            args.source_root,
            args.output_root / args.source_site / "train",
            require_labels=True,
            allow_unlabeled=False,
        )
        source_train, source_val, val_subjects = assign_source_splits(
            source_frames, args.source_val_subjects
        )
        datasets["source_train"] = move_frames_to_dir(
            source_train, args.output_root / args.source_site / "train"
        )
        datasets["source_test"] = move_frames_to_dir(
            source_val, args.output_root / args.source_site / "test"
        )
        labeled["source_train"] = source_labeled
        labeled["source_test"] = source_labeled
        metadata["source_input_root"] = str(args.source_root)
        metadata["source_validation_subjects"] = val_subjects

    if "target-train" in selected:
        target_train, train_labeled = build_frames(
            args.target_train_root,
            args.output_root / args.target_site / "train",
            require_labels=False,
            allow_unlabeled=True,
        )
        datasets["target_train"] = target_train
        labeled["target_train"] = train_labeled
        metadata["target_train_input_root"] = str(args.target_train_root)

    if "target-test" in selected:
        target_test, test_labeled = build_frames(
            args.target_test_root,
            args.output_root / args.target_site / "test",
            require_labels=True,
            allow_unlabeled=False,
        )
        datasets["target_test"] = target_test
        labeled["target_test"] = test_labeled
        metadata["target_test_input_root"] = str(args.target_test_root)

    for name in list(datasets):
        if args.limit is not None:
            datasets[name] = datasets[name][: args.limit]

    for name, frames in datasets.items():
        label_state = "labeled" if labeled[name] else "placeholder zero masks"
        print(f"{name}: {len(frames)} frames ({label_state})")

    if args.dry_run:
        return 0

    run_dirs = {
        "target_refine_dir": args.output_root / f"{args.target_site}_refine",
        "target_refine_postprocess_dir": (
            args.output_root / f"{args.target_site}_refine_postprocess"
        ),
    }
    for path in run_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    tasks = [frame for frames in datasets.values() for frame in frames]
    counts = {"written": 0, "skipped": 0}
    for index, name in enumerate(datasets, start=1):
        print(f"[{index}/{len(datasets)}] Writing {name} ({len(datasets[name])} files)")
    if args.workers == 1:
        for i, frame in enumerate(tasks, start=1):
            status = write_frame(
                frame,
                expected_size=args.size,
                num_classes=NUM_CLASSES,
                image_min=args.image_min,
                image_max=args.image_max,
                overwrite=args.overwrite,
                uncompressed=args.uncompressed,
            )
            counts[status] += 1
            if i % 1000 == 0:
                print(f"processed {i}/{len(tasks)} files")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    write_frame,
                    frame,
                    expected_size=args.size,
                    num_classes=NUM_CLASSES,
                    image_min=args.image_min,
                    image_max=args.image_max,
                    overwrite=args.overwrite,
                    uncompressed=args.uncompressed,
                )
                for frame in tasks
            ]
            for i, future in enumerate(as_completed(futures), start=1):
                counts[future.result()] += 1
                if i % 1000 == 0:
                    print(f"processed {i}/{len(tasks)} files")

    manifest = {
        "schema": {
            "keys": ["image", "label"],
            "image": {
                "shape": [args.size, args.size, 3],
                "dtype": "float32",
                "normalization": {
                    "fixed_min": args.image_min,
                    "fixed_max": args.image_max,
                    "formula": "(clip(image, min, max) - min) / (max - min)",
                    "channels": "RGB-shaped; grayscale inputs are duplicated",
                },
            },
            "label": {
                "shape": [args.size, args.size],
                "dtype": "uint8",
                "value_range": [0, NUM_CLASSES - 1],
                "background": 0,
                "foreground_classes": NUM_CLASSES - 1,
            },
            "compression": (
                "np.savez" if args.uncompressed else "np.savez_compressed"
            ),
        },
        "class_count_including_background": NUM_CLASSES,
        "source_site": args.source_site,
        "target_site": args.target_site,
        "dataset_counts": {name: len(frames) for name, frames in datasets.items()},
        "dataset_labeled": labeled,
        **metadata,
        **{name: str(path) for name, path in run_dirs.items()},
        "write_counts": counts,
    }
    write_manifest(args.output_root, manifest)

    print(
        f"Complete: {counts['written']} written, {counts['skipped']} skipped. "
        f"Manifest: {args.output_root / 'manifest.json'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
