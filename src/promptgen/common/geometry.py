from __future__ import annotations

from typing import Tuple

BoxXYZ = Tuple[int, int, int, int, int, int]


def clamp_box_xyz(box: BoxXYZ, shape_xyz: Tuple[int, int, int]) -> BoxXYZ:
    """Clamp a box to image bounds.

    Inputs: box (x0,y0,z0,x1,y1,z1), shape_xyz (X,Y,Z).
    Outputs: clamped BoxXYZ.
    Operation: clamps start to >=0 and end to <=shape, preserving ordering.
    """
    x0, y0, z0, x1, y1, z1 = box
    sx, sy, sz = shape_xyz
    cx0 = max(0, min(x0, sx))
    cy0 = max(0, min(y0, sy))
    cz0 = max(0, min(z0, sz))
    cx1 = max(0, min(x1, sx))
    cy1 = max(0, min(y1, sy))
    cz1 = max(0, min(z1, sz))
    return (cx0, cy0, cz0, cx1, cy1, cz1)


def box_volume_xyz(box: BoxXYZ) -> int:
    """Compute volume of a half-open XYZ box.

    Inputs: box (x0,y0,z0,x1,y1,z1).
    Outputs: integer volume (0 if invalid or empty).
    Operation: computes max(0, dx)*max(0, dy)*max(0, dz).
    """
    x0, y0, z0, x1, y1, z1 = box
    dx = max(0, x1 - x0)
    dy = max(0, y1 - y0)
    dz = max(0, z1 - z0)
    return int(dx * dy * dz)


def box_iou_xyz(a: BoxXYZ, b: BoxXYZ) -> float:
    """Compute IoU between two half-open XYZ boxes.

    Inputs: a, b (boxes).
    Outputs: IoU in [0,1].
    Operation: computes intersection and union volumes; returns 0 if union is 0.
    """
    ax0, ay0, az0, ax1, ay1, az1 = a
    bx0, by0, bz0, bx1, by1, bz1 = b

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    iz0 = max(az0, bz0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iz1 = min(az1, bz1)

    inter = box_volume_xyz((ix0, iy0, iz0, ix1, iy1, iz1))
    if inter == 0:
        return 0.0
    union = box_volume_xyz(a) + box_volume_xyz(b) - inter
    return 0.0 if union == 0 else inter / union


def center_in_box_xyz(center_xyz: Tuple[int, int, int], box: BoxXYZ) -> bool:
    """Check if a center point lies inside a half-open XYZ box.

    Inputs: center_xyz (x,y,z), box (x0,y0,z0,x1,y1,z1).
    Outputs: True if inside, else False.
    Operation: checks x0<=x<x1, y0<=y<y1, z0<=z<z1.
    """
    x, y, z = center_xyz
    x0, y0, z0, x1, y1, z1 = box
    return (x0 <= x < x1) and (y0 <= y < y1) and (z0 <= z < z1)
