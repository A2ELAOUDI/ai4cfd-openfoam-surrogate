"""Evaluate the trained surrogate model on the held-out test set.

Computes MAE, RMSE, R² per output, saves metrics to results/metrics.csv,
and generates prediction vs ground truth figures.

Usage
-----
python models/evaluate.py \\
    --dataset     data/dataset.pt \\
    --checkpoint  checkpoints/best_model.pt \\
    --figures-dir results/figures/
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.metrics import compute_all, format_metrics_table  # noqa: E402
from models.model import SurrogateModel                        # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _denorm(arr: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    return arr * (vmax - vmin) + vmin


def _load_model(checkpoint_path: Path, device: torch.device) -> SurrogateModel:
    model = SurrogateModel(n_inputs=3, n_outputs=2).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(
        "Loaded checkpoint: epoch=%d  val_loss=%.5f",
        ckpt.get("epoch", -1),
        ckpt.get("val_loss", float("nan")),
    )
    return model


def _predict(
    model: SurrogateModel,
    X_norm: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(X_norm.to(device))
    return out.cpu().numpy()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_loss_curves(loss_csv: Path, figures_dir: Path) -> None:
    if not loss_csv.exists():
        log.warning("Loss CSV not found: %s — skipping loss curve plot", loss_csv)
        return

    import pandas as pd
    df = pd.read_csv(loss_csv)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["epoch"], df["train_loss"], label="Train loss", lw=1.5)
    ax.plot(df["epoch"], df["val_loss"], label="Val loss", lw=1.5, ls="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (physics-informed MSE)")
    ax.set_title("Training Loss Curves")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = figures_dir / "loss_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


def _plot_predictions_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: list[str],
    figures_dir: Path,
) -> None:
    """4-panel figure: parity + residuals for each output."""
    units = {"wave_front_x": "m", "max_velocity": "m/s"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Surrogate Model — Test Set Evaluation", fontsize=13, fontweight="bold")

    for col_i, col in enumerate(target_cols):
        yt = y_true[:, col_i]
        yp = y_pred[:, col_i]
        unit = units.get(col, "")
        label = col.replace("_", " ").title()

        # Parity
        ax = axes[col_i, 0]
        lims = [min(yt.min(), yp.min()) * 0.95, max(yt.max(), yp.max()) * 1.05]
        ax.scatter(yt, yp, s=18, alpha=0.55, zorder=3)
        ax.plot(lims, lims, "k--", lw=1.2, label="1:1")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"OpenFOAM [{unit}]")
        ax.set_ylabel(f"Surrogate [{unit}]")
        ax.set_title(f"{label} — Parity")
        ax.set_aspect("equal")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Residuals
        ax = axes[col_i, 1]
        residuals = yp - yt
        ax.scatter(yt, residuals, s=18, alpha=0.55, color="tomato", zorder=3)
        ax.axhline(0, color="black", lw=1.2, ls="--")
        ax.set_xlabel(f"OpenFOAM (true) [{unit}]")
        ax.set_ylabel(f"Residual (pred − true) [{unit}]")
        ax.set_title(f"{label} — Residuals")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = figures_dir / "predictions_grid.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(
    dataset_path: Path,
    checkpoint_path: Path,
    figures_dir: Path,
    metrics_csv: Path,
) -> dict:
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset = torch.load(dataset_path, weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _load_model(checkpoint_path, device)

    test_idx = dataset["test_idx"]
    X_norm_test = dataset["X_norm"][test_idx]
    y_norm_true = dataset["y_norm"][test_idx].numpy()

    y_min = dataset["y_min"].numpy()
    y_max = dataset["y_max"].numpy()
    target_cols: list[str] = dataset["target_cols"]

    # Predict in normalised space, then denormalise
    y_norm_pred = _predict(model, X_norm_test, device)
    y_true = _denorm(y_norm_true, y_min, y_max)
    y_pred = _denorm(y_norm_pred, y_min, y_max)

    # Compute metrics in physical units
    metrics = compute_all(y_true, y_pred, target_cols)

    table = format_metrics_table(metrics)
    log.info("\n%s", table)

    # Save metrics.csv
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with metrics_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["output", "MAE", "RMSE", "R2", "MaxAE", "RelErr"])
        writer.writeheader()
        for col, vals in metrics.items():
            writer.writerow({"output": col, **vals})
    log.info("Metrics → %s", metrics_csv)

    # Figures
    _plot_predictions_grid(y_true, y_pred, target_cols, figures_dir)

    loss_csv = checkpoint_path.parent / "loss_curves.csv"
    _plot_loss_curves(loss_csv, figures_dir)

    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=REPO_ROOT / "data" / "dataset.pt")
    p.add_argument(
        "--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "best_model.pt"
    )
    p.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "results" / "figures")
    p.add_argument("--metrics-csv", type=Path, default=REPO_ROOT / "results" / "metrics.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        figures_dir=args.figures_dir,
        metrics_csv=args.metrics_csv,
    )


if __name__ == "__main__":
    main()
