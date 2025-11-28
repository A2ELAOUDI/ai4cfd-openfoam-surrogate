"""Assemble extracted CSVs into a normalised PyTorch dataset.

Reads per-case extracted_fields.csv files together with cases_config.json
to build (features, targets) tensors, applies min-max normalisation, splits
into train/val/test (70/15/15), and saves dataset.pt.

Works with both OpenFOAM-generated data and the included sample_data/.

Usage
-----
python scripts/build_dataset.py \\
    --extracted-dir sample_data/ \\
    --config        sample_data/cases_config.json \\
    --output        data/dataset.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

FEATURE_COLS = ["water_height", "water_width", "time"]
TARGET_COLS = ["wave_front_x", "max_velocity"]


def _load_all_rows(extracted_dir: Path, config: list[dict]) -> pd.DataFrame:
    """Merge per-case time series with their parameter metadata."""
    frames: list[pd.DataFrame] = []

    for meta in config:
        case_id = meta["case_id"]
        csv_path = extracted_dir / case_id / "extracted_fields.csv"

        if not csv_path.exists():
            log.warning("Missing CSV: %s — skipping", csv_path)
            continue

        df = pd.read_csv(csv_path)
        df["water_height"] = meta["water_height"]
        df["water_width"] = meta["water_width"]
        df["case_id"] = case_id
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No extracted_fields.csv files found under {extracted_dir}. "
            "Run extract_fields.py first, or use sample_data/."
        )

    combined = pd.concat(frames, ignore_index=True)
    log.info(
        "Loaded %d rows from %d cases", len(combined), len(frames)
    )
    return combined


def _train_val_test_split(
    n: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def build_dataset(
    extracted_dir: Path,
    config_path: Path,
    output_path: Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> None:
    with config_path.open() as f:
        config = json.load(f)

    df = _load_all_rows(extracted_dir, config)

    # Validate required columns
    for col in FEATURE_COLS + TARGET_COLS:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' missing from dataset. Found: {df.columns.tolist()}")

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df[TARGET_COLS].to_numpy(dtype=np.float32)

    # Min-max normalisation (fit on full data — small dataset, no leakage concern)
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    y_min, y_max = y.min(axis=0), y.max(axis=0)

    # Avoid division by zero for constant columns
    X_range = np.where(X_max - X_min > 0, X_max - X_min, 1.0)
    y_range = np.where(y_max - y_min > 0, y_max - y_min, 1.0)

    X_norm = (X - X_min) / X_range
    y_norm = (y - y_min) / y_range

    train_idx, val_idx, test_idx = _train_val_test_split(
        len(df), train_frac, val_frac, seed
    )

    dataset = {
        # Raw tensors
        "X": torch.from_numpy(X),
        "y": torch.from_numpy(y),
        # Normalised tensors
        "X_norm": torch.from_numpy(X_norm),
        "y_norm": torch.from_numpy(y_norm),
        # Normalisation statistics (to invert predictions)
        "X_min": torch.from_numpy(X_min),
        "X_max": torch.from_numpy(X_max),
        "y_min": torch.from_numpy(y_min),
        "y_max": torch.from_numpy(y_max),
        # Split indices
        "train_idx": torch.from_numpy(train_idx),
        "val_idx": torch.from_numpy(val_idx),
        "test_idx": torch.from_numpy(test_idx),
        # Metadata
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
        "n_samples": len(df),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)

    log.info(
        "Dataset saved → %s  (%d train / %d val / %d test)",
        output_path,
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )
    log.info("Feature ranges: %s", list(zip(FEATURE_COLS, X_min.tolist(), X_max.tolist())))
    log.info("Target  ranges: %s", list(zip(TARGET_COLS, y_min.tolist(), y_max.tolist())))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--extracted-dir",
        type=Path,
        default=REPO_ROOT / "sample_data",
        help="Directory with per-case extracted_fields.csv files",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "sample_data" / "cases_config.json",
        help="Path to cases_config.json",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "dataset.pt",
        help="Output path for dataset.pt",
    )
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        extracted_dir=args.extracted_dir,
        config_path=args.config,
        output_path=args.output,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
