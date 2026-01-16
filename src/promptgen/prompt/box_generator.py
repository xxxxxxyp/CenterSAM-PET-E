from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

BoxXYZ = Tuple[int, int, int, int, int, int]
Center = Tuple[int, int, int]


@dataclass(frozen=True)
class Proposal:
    box_xyz_vox: BoxXYZ
    center_xyz_vox: Center
    heatmap_score: float
    generator: str
    generator_params: Dict


def _clamp_box_to_shape(box: BoxXYZ, shape_xyz: Tuple[int, int, int]) -> BoxXYZ:
    """Clamp box bounds to image shape.

    Inputs: box (x0,y0,z0,x1,y1,z1), shape_xyz (X,Y,Z).
    Outputs: clamped BoxXYZ.
    Operation: clamps lower bounds to >=0 and upper bounds to <=shape.
    """
    x0, y0, z0, x1, y1, z1 = box
    sx, sy, sz = shape_xyz
    return (
        max(0, min(x0, sx)),
        max(0, min(y0, sy)),
        max(0, min(z0, sz)),
        max(0, min(x1, sx)),
        max(0, min(y1, sy)),
        max(0, min(z1, sz)),
    )


def generate_fixed_multiscale_boxes(
    centers: List[Tuple[Center, float]],
    shape_xyz: Tuple[int, int, int],
    spacing_xyz_mm: Tuple[float, float, float],
    edge_mm_set: List[float],
    padding_ratio: float,
) -> List[Proposal]:
    """Generate fixed-multiscale cube proposals around centers.

    Inputs: centers ([(center_xyz, score)]), shape_xyz (X,Y,Z),
    spacing_xyz_mm (mm spacing), edge_mm_set (list of cube edges in mm),
    padding_ratio (extra padding factor).
    Outputs: list of Proposal.
    Operation: for each center and edge length, converts mm to voxels per axis
    using ceil, applies padding, builds a centered cube, clamps to image,
    and drops invalid boxes.
    """
    proposals: List[Proposal] = []
    sx, sy, sz = shape_xyz
    spx, spy, spz = spacing_xyz_mm

    for (center, score) in centers:
        cx, cy, cz = center
        for edge_mm in edge_mm_set:
            if edge_mm <= 0:
                continue
            edge_x = math.ceil(edge_mm / spx)
            edge_y = math.ceil(edge_mm / spy)
            edge_z = math.ceil(edge_mm / spz)

            edge_x = math.ceil(edge_x * (1.0 + padding_ratio))
            edge_y = math.ceil(edge_y * (1.0 + padding_ratio))
            edge_z = math.ceil(edge_z * (1.0 + padding_ratio))

            hx = max(1, edge_x // 2)
            hy = max(1, edge_y // 2)
            hz = max(1, edge_z // 2)

            x0 = cx - hx
            x1 = cx + (edge_x - hx)
            y0 = cy - hy
            y1 = cy + (edge_y - hy)
            z0 = cz - hz
            z1 = cz + (edge_z - hz)

            box = _clamp_box_to_shape((x0, y0, z0, x1, y1, z1), shape_xyz)
            bx0, by0, bz0, bx1, by1, bz1 = box
            if bx1 <= bx0 or by1 <= by0 or bz1 <= bz0:
                continue

            proposals.append(
                Proposal(
                    box_xyz_vox=box,
                    center_xyz_vox=(int(cx), int(cy), int(cz)),
                    heatmap_score=float(score),
                    generator="fixed_multiscale",
                    generator_params={
                        "edge_mm": float(edge_mm),
                        "padding_ratio": float(padding_ratio),
                        "edge_vox_xyz": (int(edge_x), int(edge_y), int(edge_z)),
                        "shape_xyz": (int(sx), int(sy), int(sz)),
                    },
                )
            )

    return proposals
