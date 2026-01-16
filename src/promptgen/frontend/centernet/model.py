from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CenterNetConfig:
    in_channels: int = 1
    base_channels: int = 16
    num_blocks: int = 4
    use_batchnorm: bool = True


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_batchnorm: bool) -> None:
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_batchnorm),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm3d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CenterNetPET(nn.Module):
    """
    Minimal heatmap-only model.
    Input:  (B, 1, X, Y, Z)
    Output: (B, 1, X, Y, Z) logits (NOT sigmoid applied)
    """

    def __init__(self, cfg: CenterNetConfig):
        super().__init__()
        if cfg.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if cfg.in_channels <= 0 or cfg.base_channels <= 0:
            raise ValueError("in_channels and base_channels must be positive")

        blocks = []
        in_ch = cfg.in_channels
        out_ch = cfg.base_channels
        for _ in range(cfg.num_blocks):
            blocks.append(_ConvBlock(in_ch, out_ch, cfg.use_batchnorm))
            in_ch = out_ch
            out_ch = out_ch * 2

        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Conv3d(in_ch, 1, kernel_size=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        heatmap_logits = self.head(features)
        return heatmap_logits
