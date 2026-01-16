from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

from ..prompt.box_generator import Proposal


def infer_classifier_scores(
    volume_xyz: np.ndarray,
    proposals: List[Proposal],
    model_path: str,
    device: str,
    patch_shape_xyz: Tuple[int, int, int],
) -> List[float]:
    """Return classifier_score per proposal (float in [0,1]).

    Inputs: volume_xyz (3D array), proposals (list), model_path (path),
    device (string), patch_shape_xyz (X,Y,Z).
    Outputs: list of classifier scores aligned to proposals.
    Operation: minimal stub; validates inputs and returns 0.0 scores.
    """
    if volume_xyz.ndim != 3:
        raise ValueError(f"volume_xyz must be 3D, got shape {volume_xyz.shape}")
    if not model_path:
        raise ValueError("model_path must be non-empty")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model_path not found: {model_path}")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be non-empty str")
    if not isinstance(patch_shape_xyz, tuple) or len(patch_shape_xyz) != 3:
        raise ValueError("patch_shape_xyz must be a tuple of length 3")
    if not all(isinstance(v, int) and v > 0 for v in patch_shape_xyz):
        raise ValueError("patch_shape_xyz values must be positive ints")

    return [0.0 for _ in proposals]
