from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .dataset import CenterNetDataset, CenterNetTrainConfig
from .losses import heatmap_bce_loss
from .model import CenterNetConfig, CenterNetPET


def _run_epoch(
    model: CenterNetPET,
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
        images = batch["image"].to(device)
        targets = batch["target_heatmap"].to(device)

        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = heatmap_bce_loss(logits, targets)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        count += 1

    return total_loss / max(1, count)


def train_centernet(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    model: CenterNetPET,
    device: torch.device,
    epochs: int,
    lr: float,
    save_every: Optional[int],
    checkpoint_path: str,
    config: CenterNetConfig,
    train_meta: Dict[str, Any],
) -> Dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}

    model.to(device)

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, device, train=True, optimizer=optimizer)
        history["train_loss"].append(train_loss)

        val_loss = None
        if val_loader is not None:
            val_loss = _run_epoch(model, val_loader, device, train=False)
            history["val_loss"].append(val_loss)

        if save_every is not None and save_every > 0 and (epoch % save_every == 0):
            _save_checkpoint(checkpoint_path, model, config, train_meta, history, epoch)

    _save_checkpoint(checkpoint_path, model, config, train_meta, history, epochs)
    return history


def _save_checkpoint(
    path: str,
    model: CenterNetPET,
    config: CenterNetConfig,
    train_meta: Dict[str, Any],
    history: Dict[str, Any],
    epoch: int,
) -> None:
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "train_meta": {
            **train_meta,
            "epoch": int(epoch),
            "history": history,
        },
    }
    torch.save(ckpt, path)
