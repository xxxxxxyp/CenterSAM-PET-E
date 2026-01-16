from __future__ import annotations

import os
from typing import Tuple

import numpy as np


def infer_heatmap_xyz(
    volume_xyz: np.ndarray,
    model_path: str,
    device: str,
) -> np.ndarray:
    """Return heatmap_xyz with same shape (X,Y,Z). v1 requires same shape.

    Inputs: volume_xyz (3D numpy array), model_path (path), device (string).
    Outputs: heatmap_xyz with same shape as input.
    Operation: minimal stub; validates input and returns zeros if model exists.
    """
    if volume_xyz.ndim != 3:
        raise ValueError(f"volume_xyz must be 3D, got shape {volume_xyz.shape}")
    if not model_path:
        raise ValueError("model_path must be non-empty")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model_path not found: {model_path}")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be non-empty str")

    heatmap_xyz = np.zeros(volume_xyz.shape, dtype=np.float32)
    return heatmap_xyz
