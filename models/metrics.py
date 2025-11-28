"""Regression metrics for surrogate model evaluation.

All functions accept numpy arrays of shape (N,) or (N, K) and return
scalar floats (or per-column arrays when input is 2-D).
"""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float | np.ndarray:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred), axis=0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float | np.ndarray:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float | np.ndarray:
    """Coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    return 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, 1.0)


def max_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float | np.ndarray:
    """Maximum Absolute Error (worst-case bound)."""
    return np.max(np.abs(y_true - y_pred), axis=0)


def relative_error(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Mean relative absolute error (percentage error / 100)."""
    return np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps), axis=0)


def compute_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    col_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute all metrics per output column.

    Returns a nested dict: {col_name: {metric: value, ...}}.
    """
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]

    n_cols = y_true.shape[1]
    if col_names is None:
        col_names = [f"output_{i}" for i in range(n_cols)]

    result: dict[str, dict[str, float]] = {}
    for i, name in enumerate(col_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        result[name] = {
            "MAE": float(mae(yt, yp)),
            "RMSE": float(rmse(yt, yp)),
            "R2": float(r2_score(yt, yp)),
            "MaxAE": float(max_absolute_error(yt, yp)),
            "RelErr": float(relative_error(yt, yp)),
        }
    return result


def format_metrics_table(metrics: dict[str, dict[str, float]]) -> str:
    """Return a Markdown table string."""
    metric_keys = list(next(iter(metrics.values())).keys())
    header = "| Output | " + " | ".join(metric_keys) + " |"
    sep = "|--------|" + "|".join(["-" * (len(k) + 2) for k in metric_keys]) + "|"
    rows = [header, sep]
    for col, vals in metrics.items():
        row = f"| {col} | " + " | ".join(f"{vals[k]:.4f}" for k in metric_keys) + " |"
        rows.append(row)
    return "\n".join(rows)
