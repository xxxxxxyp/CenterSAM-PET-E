from __future__ import annotations

import torch
import torch.nn.functional as F


def heatmap_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCEWithLogitsLoss, reduction='mean'."""
    return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
