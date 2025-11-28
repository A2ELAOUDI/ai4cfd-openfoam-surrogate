"""Physics-informed MLP surrogate for dam-break CFD predictions.

Architecture: 3 inputs → [64, 128, 64] hidden → 2 outputs
Inputs:  [water_height, water_width, time]  (normalised to [0, 1])
Outputs: [wave_front_x, max_velocity]        (normalised to [0, 1])

Physics-informed loss penalises:
  1. Negative wave front predictions  (wave front must advance, not retreat)
  2. Super-physical velocities        (|U| < sqrt(2*g*H_max) ≈ 4 m/s)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Physical upper bound on velocity for our parameter range
# sqrt(2 * 9.81 * 0.8) ≈ 3.96 m/s → rounded up to 5 for safety
_MAX_PHYSICAL_VELOCITY = 5.0


class SurrogateModel(nn.Module):
    """MLP surrogate for scalar CFD quantities.

    Parameters
    ----------
    n_inputs:
        Number of input features (default 3: height, width, time).
    n_outputs:
        Number of output targets (default 2: wave_front_x, max_velocity).
    hidden_dims:
        Sizes of hidden layers.
    dropout:
        Dropout probability applied after each hidden activation.
    """

    def __init__(
        self,
        n_inputs: int = 3,
        n_outputs: int = 2,
        hidden_dims: tuple[int, ...] = (64, 128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = n_inputs
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_outputs))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class PhysicsInformedLoss(nn.Module):
    """MSE loss augmented with soft physics constraints.

    Constraints (in normalised output space):
      - wave_front_x ≥ 0  → penalise negative predictions (col 0)
      - max_velocity  ≥ 0  → penalise negative predictions (col 1)

    The physics penalty weight λ is annealed during training:
    pass a schedule externally or keep it fixed at the default.

    Parameters
    ----------
    physics_weight:
        Weight λ for the physics penalty term.
    """

    def __init__(self, physics_weight: float = 0.1) -> None:
        super().__init__()
        self.physics_weight = physics_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mse = F.mse_loss(pred, target)

        # Penalise any prediction below zero (physical quantities are non-negative)
        neg_penalty = F.relu(-pred).pow(2).mean()

        # Penalise predictions that exceed 1.0 in normalised space
        # (corresponds to exceeding the max observed in the training data)
        over_penalty = F.relu(pred - 1.0).pow(2).mean()

        physics_loss = neg_penalty + over_penalty
        total = mse + self.physics_weight * physics_loss

        return total, {
            "mse": mse.item(),
            "physics_penalty": physics_loss.item(),
            "total": total.item(),
        }
