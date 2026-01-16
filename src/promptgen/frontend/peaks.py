from __future__ import annotations

from typing import List, Tuple
import numpy as np

Center = Tuple[int, int, int]


def extract_topk_peaks_3d_xyz(
    heatmap_xyz: np.ndarray,  # (X,Y,Z)
    gating_mask_xyz: np.ndarray,  # (X,Y,Z) bool or None
    topk: int,
    neighborhood: int = 1,  # local maxima radius
) -> List[Tuple[Center, float]]:
    """
    Deterministic local maxima extraction.
    Ties stable: sort by (-score, x, y, z).
    """
    if heatmap_xyz.ndim != 3:
        raise ValueError(f"heatmap_xyz must be 3D, got shape {heatmap_xyz.shape}")
    if topk <= 0:
        return []
    if neighborhood < 0:
        raise ValueError("neighborhood must be non-negative")

    if gating_mask_xyz is not None:
        if gating_mask_xyz.shape != heatmap_xyz.shape:
            raise ValueError("gating_mask_xyz shape must match heatmap_xyz")
        gating = gating_mask_xyz.astype(bool)
    else:
        gating = None

    sx, sy, sz = heatmap_xyz.shape
    candidates: List[Tuple[Center, float]] = []

    if gating is None:
        coords = np.ndindex(heatmap_xyz.shape)
    else:
        coords = (tuple(p) for p in np.argwhere(gating))

    for x, y, z in coords:
        score = float(heatmap_xyz[x, y, z])
        x0 = max(0, x - neighborhood)
        x1 = min(sx, x + neighborhood + 1)
        y0 = max(0, y - neighborhood)
        y1 = min(sy, y + neighborhood + 1)
        z0 = max(0, z - neighborhood)
        z1 = min(sz, z + neighborhood + 1)
        local = heatmap_xyz[x0:x1, y0:y1, z0:z1]
        if score >= float(np.max(local)):
            candidates.append(((int(x), int(y), int(z)), score))

    candidates.sort(key=lambda c: (-c[1], c[0][0], c[0][1], c[0][2]))
    return candidates[:topk]
