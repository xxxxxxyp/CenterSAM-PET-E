from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ...common.instances import extract_lesion_instances_xyz

Center = Tuple[int, int, int]


def make_centers_from_label_xyz(label_xyz: np.ndarray) -> List[Center]:
    """Use 26-connectivity instances; center = np.rint(mean(coords))."""
    instances = extract_lesion_instances_xyz(label_xyz)
    return [inst.center_xyz for inst in instances]


def draw_gaussian_3d(
    heatmap: np.ndarray,  # (X,Y,Z)
    center: Center,
    sigma_vox: float,
    peak: float = 1.0,
) -> None:
    """In-place max with Gaussian."""
    if heatmap.ndim != 3:
        raise ValueError("heatmap must be 3D")
    if sigma_vox <= 0:
        raise ValueError("sigma_vox must be positive")

    cx, cy, cz = center
    sx, sy, sz = heatmap.shape
    radius = int(np.ceil(3.0 * sigma_vox))

    x0 = max(0, cx - radius)
    x1 = min(sx, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(sy, cy + radius + 1)
    z0 = max(0, cz - radius)
    z1 = min(sz, cz + radius + 1)

    xs = np.arange(x0, x1)
    ys = np.arange(y0, y1)
    zs = np.arange(z0, z1)

    dx = (xs - cx).astype(np.float32)
    dy = (ys - cy).astype(np.float32)
    dz = (zs - cz).astype(np.float32)

    gx = dx[:, None, None] ** 2
    gy = dy[None, :, None] ** 2
    gz = dz[None, None, :] ** 2

    denom = 2.0 * float(sigma_vox) * float(sigma_vox)
    gaussian = peak * np.exp(-(gx + gy + gz) / denom)

    patch = heatmap[x0:x1, y0:y1, z0:z1]
    np.maximum(patch, gaussian, out=patch)


def build_heatmap_target_xyz(
    label_xyz: np.ndarray,
    shape_xyz: Tuple[int, int, int],
    sigma_vox: float,
) -> np.ndarray:
    """Return target heatmap in [0,1], float32."""
    if label_xyz.ndim != 3:
        raise ValueError(f"label_xyz must be 3D, got shape {label_xyz.shape}")
    if label_xyz.shape != shape_xyz:
        raise ValueError("label_xyz shape must match shape_xyz")
    if sigma_vox <= 0:
        raise ValueError("sigma_vox must be positive")

    heatmap = np.zeros(shape_xyz, dtype=np.float32)
    centers = make_centers_from_label_xyz(label_xyz)
    for center in centers:
        draw_gaussian_3d(heatmap, center=center, sigma_vox=sigma_vox, peak=1.0)

    heatmap = np.clip(heatmap, 0.0, 1.0)
    return heatmap.astype(np.float32, copy=False)
