from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ClassifierConfig:
    in_channels: int = 1
    base_channels: int = 8


class ProposalClassifier(nn.Module):
    def __init__(self, cfg: ClassifierConfig):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(cfg.in_channels, cfg.base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(cfg.base_channels, cfg.base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.head = nn.Conv3d(cfg.base_channels * 2, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.head(feat).flatten(1)
        return logits


def _run_epoch(
    model: ProposalClassifier,
    loader: DataLoader,
    device: torch.device,
    train: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> float:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    count = 0

    for batch in loader:
        patches = batch["patch"].to(device)
        labels = batch["y"].to(device)

        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        logits = model(patches)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")

        if train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        count += 1

    return total_loss / max(1, count)


def train_classifier(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    model: ProposalClassifier,
    device: torch.device,
    epochs: int,
    lr: float,
    checkpoint_path: str,
    config: ClassifierConfig,
    train_meta: Dict[str, Any],
) -> Dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}

    model.to(device)

    for epoch in range(1, epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        train_loss = _run_epoch(model, train_loader, device, train=True, optimizer=optimizer)
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            if hasattr(val_loader.dataset, "set_epoch"):
                val_loader.dataset.set_epoch(epoch)
            val_loss = _run_epoch(model, val_loader, device, train=False)
            history["val_loss"].append(val_loss)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "train_meta": {
            **train_meta,
            "epochs": int(epochs),
            "history": history,
        },
    }
    torch.save(ckpt, checkpoint_path)

    return history
