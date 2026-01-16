from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch

from .model import CenterNetConfig, CenterNetPET


@dataclass(frozen=True)
class CenterNetWeights:
    model_state_dict: Dict[str, Any]
    config: CenterNetConfig


def load_centernet_checkpoint(path: str, map_location: str = "cpu") -> CenterNetWeights:
    """
    Expect checkpoint dict keys:
      - 'model_state_dict'
      - 'config' (dict compatible with CenterNetConfig)
    Raise ValueError if missing.
    """
    if not path:
        raise ValueError("path must be non-empty")
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=map_location)
    if not isinstance(ckpt, dict):
        raise ValueError("checkpoint must be a dict")
    if "model_state_dict" not in ckpt or "config" not in ckpt:
        raise ValueError("checkpoint missing model_state_dict or config")

    cfg_raw = ckpt["config"]
    if isinstance(cfg_raw, CenterNetConfig):
        cfg = cfg_raw
    elif isinstance(cfg_raw, dict):
        cfg = CenterNetConfig(**cfg_raw)
    else:
        raise ValueError("config must be dict or CenterNetConfig")

    model_state_dict = ckpt["model_state_dict"]
    if not isinstance(model_state_dict, dict):
        raise ValueError("model_state_dict must be a dict")

    return CenterNetWeights(model_state_dict=model_state_dict, config=cfg)


def infer_heatmap_xyz(
    volume_xyz: np.ndarray,  # (X,Y,Z) float32
    checkpoint_path: str,
    device: str = "cpu",
    amp: bool = False,
) -> np.ndarray:
    """
    Returns heatmap_xyz (X,Y,Z) float32 in [0,1].
    Steps:
      1) build model from cfg
      2) load state_dict
      3) forward on tensor (1,1,X,Y,Z)
      4) apply sigmoid
      5) ensure finite + clamp [0,1]
    """
    if volume_xyz.ndim != 3:
        raise ValueError(f"volume_xyz must be 3D, got shape {volume_xyz.shape}")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be non-empty str")

    dev = torch.device(device)
    weights = load_centernet_checkpoint(checkpoint_path, map_location=device)
    model = CenterNetPET(weights.config)
    model.load_state_dict(weights.model_state_dict)
    model.to(dev)
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(volume_xyz.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(dev)
        if amp:
            with torch.autocast(device_type=dev.type, enabled=True):
                logits = model(x)
        else:
            logits = model(x)
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 0.0, 1.0)
        heatmap = probs.squeeze(0).squeeze(0).cpu().numpy()

    if heatmap.shape != volume_xyz.shape:
        raise ValueError("heatmap_xyz shape must match volume_xyz")
    if not np.all(np.isfinite(heatmap)):
        raise ValueError("heatmap contains non-finite values")

    return heatmap.astype(np.float32, copy=False)
