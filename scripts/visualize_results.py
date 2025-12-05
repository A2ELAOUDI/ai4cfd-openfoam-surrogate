"""Generate publication-quality figures comparing surrogate model vs OpenFOAM.

Produces:
  - Wave front position over time (model vs ground truth, per case)
  - Max velocity over time (model vs ground truth, per case)
  - Error distribution histograms
  - Pipeline methodology diagram

Usage
-----
python scripts/visualize_results.py \\
    --dataset     data/dataset.pt \\
    --checkpoint  checkpoints/best_model.pt \\
    --figures-dir results/figures/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.model import SurrogateModel  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Consistent colour palette
COLORS = {
    "openfoam": "#1f77b4",
    "surrogate": "#d62728",
    "error": "#2ca02c",
    "neutral": "#7f7f7f",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    }
)


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def _predict(model: SurrogateModel, X_norm: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(X_norm)


def _denorm(tensor: torch.Tensor, vmin: torch.Tensor, vmax: torch.Tensor) -> np.ndarray:
    return (tensor * (vmax - vmin) + vmin).numpy()


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------

def plot_time_series(
    dataset: dict,
    y_pred_raw: np.ndarray,
    figures_dir: Path,
) -> None:
    """Plot wave front and max velocity over time for each unique parameter set."""
    X = dataset["X"].numpy()
    y_true = dataset["y"].numpy()

    figures_dir.mkdir(parents=True, exist_ok=True)

    # Group rows by (water_height, water_width)
    param_keys = np.round(X[:, :2], 4)
    unique_params = np.unique(param_keys, axis=0)

    for params in unique_params:
        mask = np.all(param_keys == params, axis=1)
        t = X[mask, 2]
        order = np.argsort(t)
        t = t[order]
        truth = y_true[mask][order]
        pred = y_pred_raw[mask][order]

        h, w = params
        tag = f"h{h:.2f}_w{w:.2f}".replace(".", "p")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
        fig.suptitle(
            f"Dam-break: water height={h:.2f} m, width={w:.2f} m",
            fontsize=13,
            fontweight="bold",
        )

        labels = ["Wave Front x [m]", "Max Velocity [m/s]"]
        for ax, col_idx, label in zip(axes, range(2), labels):
            ax.plot(
                t,
                truth[:, col_idx],
                color=COLORS["openfoam"],
                lw=2,
                label="OpenFOAM",
                marker="o",
                ms=4,
            )
            ax.plot(
                t,
                pred[:, col_idx],
                color=COLORS["surrogate"],
                lw=2,
                ls="--",
                label="Surrogate MLP",
                marker="s",
                ms=4,
            )
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(label)
            ax.legend()
            ax.grid(alpha=0.3)

        fig.tight_layout()
        out = figures_dir / f"timeseries_{tag}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved %s", out)


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: list[str],
    figures_dir: Path,
) -> None:
    """Histogram of absolute errors for each output."""
    errors = np.abs(y_true - y_pred)
    fig, axes = plt.subplots(1, len(target_cols), figsize=(5 * len(target_cols), 4))
    if len(target_cols) == 1:
        axes = [axes]

    for ax, col, err in zip(axes, target_cols, errors.T):
        unit = "[m]" if "x" in col else "[m/s]"
        ax.hist(err, bins=20, color=COLORS["error"], edgecolor="white", alpha=0.85)
        ax.axvline(err.mean(), color="black", ls="--", lw=1.5, label=f"Mean={err.mean():.3f}")
        ax.set_xlabel(f"Absolute error {unit}")
        ax.set_ylabel("Count")
        ax.set_title(col.replace("_", " ").title())
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Error Distribution — Test Set", fontweight="bold")
    fig.tight_layout()
    out = figures_dir / "error_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: list[str],
    figures_dir: Path,
) -> None:
    """Parity (predicted vs true) scatter plot."""
    fig, axes = plt.subplots(1, len(target_cols), figsize=(5 * len(target_cols), 4.5))
    if len(target_cols) == 1:
        axes = [axes]

    for ax, col, yt, yp in zip(axes, target_cols, y_true.T, y_pred.T):
        lims = [min(yt.min(), yp.min()) * 0.95, max(yt.max(), yp.max()) * 1.05]
        ax.scatter(yt, yp, alpha=0.5, s=20, color=COLORS["openfoam"], zorder=3)
        ax.plot(lims, lims, "k--", lw=1.5, zorder=4, label="Perfect prediction")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        unit = "[m]" if "x" in col else "[m/s]"
        label = col.replace("_", " ").title()
        ax.set_xlabel(f"OpenFOAM {unit}")
        ax.set_ylabel(f"Surrogate {unit}")
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    fig.suptitle("Parity Plot — Test Set", fontweight="bold")
    fig.tight_layout()
    out = figures_dir / "parity_plot.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


def plot_methodology(figures_dir: Path) -> None:
    """Generate a schematic pipeline overview figure."""
    fig = plt.figure(figsize=(14, 4))
    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.05)

    boxes = [
        ("Parameter\nSweep", "#AED6F1"),
        ("OpenFOAM\ninterFoam", "#A9DFBF"),
        ("Field\nExtraction", "#FAD7A0"),
        ("MLP\nSurrogate", "#F1948A"),
        ("Evaluation\n& Figures", "#D7BDE2"),
    ]
    arrows = ["→", "→", "→", "→"]

    for i, (label, color) in enumerate(boxes):
        ax = fig.add_subplot(gs[i])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        ax.add_patch(
            FancyBboxPatch(
                (0.05, 0.2),
                0.90,
                0.60,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor="#555",
                lw=1.5,
            )
        )
        ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=11, fontweight="bold")
        if i < len(arrows):
            ax.text(1.0, 0.5, arrows[i], ha="center", va="center", fontsize=18, color="#555",
                    transform=ax.transAxes)

    fig.suptitle("AI4CFD Surrogate Pipeline", fontsize=14, fontweight="bold", y=1.02)
    out = figures_dir / "pipeline_overview.png"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def visualize(
    dataset_path: Path,
    checkpoint_path: Path,
    figures_dir: Path,
) -> None:
    dataset = torch.load(dataset_path, weights_only=False)
    test_idx = dataset["test_idx"]
    X_norm = dataset["X_norm"][test_idx]
    y_min = dataset["y_min"]
    y_max = dataset["y_max"]

    model = SurrogateModel(n_inputs=3, n_outputs=2)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_pred_norm = _predict(model, X_norm)
    y_pred = _denorm(y_pred_norm, y_min, y_max)
    y_true = dataset["y"][test_idx].numpy()

    target_cols: list[str] = dataset["target_cols"]

    plot_parity(y_true, y_pred, target_cols, figures_dir)
    plot_error_distribution(y_true, y_pred, target_cols, figures_dir)

    # Time series uses all data (not just test split) for full curves
    all_pred_norm = _predict(model, dataset["X_norm"])
    all_pred = _denorm(all_pred_norm, y_min, y_max)
    plot_time_series(dataset, all_pred, figures_dir)

    plot_methodology(figures_dir)
    log.info("All figures saved to %s", figures_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=REPO_ROOT / "data" / "dataset.pt")
    p.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "best_model.pt")
    p.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "results" / "figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    visualize(args.dataset, args.checkpoint, args.figures_dir)


if __name__ == "__main__":
    main()
