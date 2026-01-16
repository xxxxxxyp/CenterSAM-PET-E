from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..common.instances import extract_lesion_instances_xyz

Center = Tuple[int, int, int]  # (x,y,z)


def dummy_from_label_centers(label_xyz: np.ndarray) -> List[Tuple[Center, float]]:
    """Return [(center_xyz, 1.0)] for each lesion instance.

    Inputs: label_xyz (3D label array).
    Outputs: list of (center_xyz, score) tuples.
    Operation: extracts lesion instances and emits their centers with score 1.0.
    """
    instances = extract_lesion_instances_xyz(label_xyz)
    return [(inst.center_xyz, 1.0) for inst in instances]


def dummy_random_centers(
    shape_xyz: Tuple[int, int, int],
    num_centers: int,
    seed: int,
    gating_mask_xyz: Optional[np.ndarray] = None,
) -> List[Tuple[Center, float]]:
    """Sample random centers within bounds or a gating mask.

    Inputs: shape_xyz (X,Y,Z), num_centers (int), seed (int),
    gating_mask_xyz (optional boolean mask).
    Outputs: list of (center_xyz, score) tuples.
    Operation: samples uniformly at random from valid voxels; assigns score 0.5.
    """
    rng = np.random.default_rng(seed)
    if num_centers <= 0:
        return []

    if gating_mask_xyz is None:
        xs = rng.integers(0, shape_xyz[0], size=num_centers)
        ys = rng.integers(0, shape_xyz[1], size=num_centers)
        zs = rng.integers(0, shape_xyz[2], size=num_centers)
        centers = list(zip(xs.astype(int), ys.astype(int), zs.astype(int)))
        return [(c, 0.5) for c in centers]

    if gating_mask_xyz.shape != shape_xyz:
        raise ValueError("gating_mask_xyz shape must match shape_xyz")

    valid = np.argwhere(gating_mask_xyz)
    if valid.size == 0:
        return []

    idx = rng.integers(0, len(valid), size=num_centers)
    chosen = valid[idx]
    centers = [(int(x), int(y), int(z)) for x, y, z in chosen]
    return [(c, 0.5) for c in centers]
