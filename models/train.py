"""Training script for the CFD surrogate model.

Trains SurrogateModel with Adam + cosine annealing LR scheduler and early
stopping. Saves best checkpoint and logs loss curves to CSV.

Usage
-----
python models/train.py \\
    --dataset  data/dataset.pt \\
    --epochs   200 \\
    --output   checkpoints/
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(REPO_ROOT))

from models.model import PhysicsInformedLoss, SurrogateModel  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_splits(
    dataset_path: Path,
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    data = torch.load(dataset_path, weights_only=False)
    X, y = data["X_norm"], data["y_norm"]
    train = TensorDataset(X[data["train_idx"]], y[data["train_idx"]])
    val = TensorDataset(X[data["val_idx"]], y[data["val_idx"]])
    test = TensorDataset(X[data["test_idx"]], y[data["test_idx"]])
    log.info(
        "Dataset: %d train / %d val / %d test",
        len(train),
        len(val),
        len(test),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    dataset_path: Path,
    output_dir: Path,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 20,
    physics_weight: float = 0.1,
    seed: int = 42,
) -> Path:
    torch.manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _ = load_splits(dataset_path)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on %s", device)

    model = SurrogateModel(n_inputs=3, n_outputs=2).to(device)
    log.info("Model parameters: %d", model.count_parameters())

    criterion = PhysicsInformedLoss(physics_weight=physics_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    log_path = output_dir / "loss_curves.csv"
    best_val_loss = float("inf")
    patience_counter = 0
    best_ckpt = output_dir / "best_model.pt"
    t0 = time.perf_counter()

    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "train_mse",
                        "val_mse", "physics_penalty", "lr"],
        )
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            # ---- Train ----
            model.train()
            tr_losses: list[dict] = []
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                pred = model(X_batch)
                loss, breakdown = criterion(pred, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                tr_losses.append(breakdown)

            tr_total = sum(d["total"] for d in tr_losses) / len(tr_losses)
            tr_mse = sum(d["mse"] for d in tr_losses) / len(tr_losses)
            tr_phys = sum(d["physics_penalty"] for d in tr_losses) / len(tr_losses)

            # ---- Validate ----
            model.eval()
            val_losses: list[dict] = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    pred = model(X_batch)
                    _, breakdown = criterion(pred, y_batch)
                    val_losses.append(breakdown)

            val_total = sum(d["total"] for d in val_losses) / len(val_losses)
            val_mse = sum(d["mse"] for d in val_losses) / len(val_losses)

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            # ---- Logging ----
            if epoch % 10 == 0 or epoch == 1:
                elapsed = time.perf_counter() - t0
                log.info(
                    "Epoch %3d/%d  train=%.5f  val=%.5f  phys=%.5f  lr=%.1e  t=%.1fs",
                    epoch,
                    epochs,
                    tr_total,
                    val_total,
                    tr_phys,
                    current_lr,
                    elapsed,
                )

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": round(tr_total, 7),
                    "val_loss": round(val_total, 7),
                    "train_mse": round(tr_mse, 7),
                    "val_mse": round(val_mse, 7),
                    "physics_penalty": round(tr_phys, 7),
                    "lr": round(current_lr, 8),
                }
            )

            # ---- Early stopping ----
            if val_total < best_val_loss:
                best_val_loss = val_total
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_total,
                        "train_loss": tr_total,
                    },
                    best_ckpt,
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    log.info(
                        "Early stopping at epoch %d (best val=%.5f)",
                        epoch,
                        best_val_loss,
                    )
                    break

    total_time = time.perf_counter() - t0
    log.info(
        "Training complete: best val loss=%.5f  total time=%.1f s",
        best_val_loss,
        total_time,
    )
    log.info("Best checkpoint → %s", best_ckpt)
    log.info("Loss curves     → %s", log_path)
    return best_ckpt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=REPO_ROOT / "data" / "dataset.pt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--physics-weight", type=float, default=0.1)
    p.add_argument("--output", type=Path, default=REPO_ROOT / "checkpoints")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        physics_weight=args.physics_weight,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
