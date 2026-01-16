from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class LesionInstance:
    instance_id: int
    voxels_xyz: np.ndarray  # shape (N,3) int
    center_xyz: Tuple[int, int, int]
    bbox_xyz: Tuple[int, int, int, int, int, int]


def _neighbor_offsets_26() -> List[Tuple[int, int, int]]:
    offsets: List[Tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                offsets.append((dx, dy, dz))
    return offsets


def extract_lesion_instances_xyz(label_xyz: np.ndarray) -> List[LesionInstance]:
    """Extract connected lesion instances from a label volume.

    Inputs: label_xyz (3D array), foreground is label>0.
    Outputs: list of LesionInstance.
    Operation: finds 26-connected components, collects voxel coords (XYZ),
    computes center as rounded mean, and bbox as half-open [min, max+1).
    """
    if label_xyz.ndim != 3:
        raise ValueError(f"label_xyz must be 3D, got shape {label_xyz.shape}")

    foreground = label_xyz > 0
    if not np.any(foreground):
        return []

    visited = np.zeros(label_xyz.shape, dtype=bool)
    offsets = _neighbor_offsets_26()
    instances: List[LesionInstance] = []
    instance_id = 1

    xs, ys, zs = np.where(foreground)
    for x, y, z in zip(xs, ys, zs):
        if visited[x, y, z]:
            continue

        stack = [(int(x), int(y), int(z))]
        visited[x, y, z] = True
        voxels: List[Tuple[int, int, int]] = []

        while stack:
            cx, cy, cz = stack.pop()
            voxels.append((cx, cy, cz))
            for dx, dy, dz in offsets:
                nx, ny, nz = cx + dx, cy + dy, cz + dz
                if (
                    0 <= nx < label_xyz.shape[0]
                    and 0 <= ny < label_xyz.shape[1]
                    and 0 <= nz < label_xyz.shape[2]
                ):
                    if not visited[nx, ny, nz] and foreground[nx, ny, nz]:
                        visited[nx, ny, nz] = True
                        stack.append((nx, ny, nz))

        voxels_arr = np.asarray(voxels, dtype=int)
        mins = voxels_arr.min(axis=0)
        maxs = voxels_arr.max(axis=0)
        bbox_xyz = (
            int(mins[0]),
            int(mins[1]),
            int(mins[2]),
            int(maxs[0] + 1),
            int(maxs[1] + 1),
            int(maxs[2] + 1),
        )
        center = tuple(np.rint(voxels_arr.mean(axis=0)).astype(int))
        instances.append(
            LesionInstance(
                instance_id=instance_id,
                voxels_xyz=voxels_arr,
                center_xyz=(int(center[0]), int(center[1]), int(center[2])),
                bbox_xyz=bbox_xyz,
            )
        )
        instance_id += 1

    return instances
