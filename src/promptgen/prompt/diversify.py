from __future__ import annotations

from typing import Dict, List, Tuple
import math

from .box_generator import Proposal


def diversify_by_grid_cell(
    proposals_scored: List[Tuple[Proposal, float]],
    spacing_xyz_mm: Tuple[float, float, float],
    eps_mm: float,
    max_per_cell: int,
) -> List[Tuple[Proposal, float]]:
    """
    cell_id = (floor(x_mm/eps), floor(y_mm/eps), floor(z_mm/eps))
    Keep up to max_per_cell per cell in score order.
    Deterministic.
    """
    if eps_mm <= 0:
        raise ValueError("eps_mm must be positive")
    if max_per_cell <= 0:
        return []

    spx, spy, spz = spacing_xyz_mm

    indexed: List[Tuple[int, Proposal, float]] = [
        (idx, proposal, score) for idx, (proposal, score) in enumerate(proposals_scored)
    ]

    indexed.sort(key=lambda item: (-float(item[2]), item[0]))

    kept: List[Tuple[Proposal, float]] = []
    counts: Dict[Tuple[int, int, int], int] = {}

    for _, proposal, score in indexed:
        cx, cy, cz = proposal.center_xyz_vox
        x_mm = float(cx) * spx
        y_mm = float(cy) * spy
        z_mm = float(cz) * spz
        cell_id = (
            int(math.floor(x_mm / eps_mm)),
            int(math.floor(y_mm / eps_mm)),
            int(math.floor(z_mm / eps_mm)),
        )

        current = counts.get(cell_id, 0)
        if current >= max_per_cell:
            continue
        counts[cell_id] = current + 1
        kept.append((proposal, float(score)))

    return kept
