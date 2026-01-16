from __future__ import annotations

from typing import List, Tuple

from ..prompt.box_generator import Proposal


def select_prompts_recall_first(
    proposals_scored: List[Tuple[Proposal, float]],  # (proposal, final_score)
    K_min: int,
    t_min: float,
    K_cap_soft: int,
) -> List[Tuple[Proposal, float]]:
    """Select prompts with recall-first strategy and soft cap.

    Inputs: proposals_scored (list of (Proposal, score)), K_min (min count),
    t_min (score threshold), K_cap_soft (max count).
    Outputs: list of (Proposal, score) after selection.
    Operation: stable sort by score desc, keep all score>=t_min, ensure at least
    K_min by adding next highest, then cap to K_cap_soft.
    """
    if K_min < 0 or K_cap_soft < 0:
        raise ValueError("K_min and K_cap_soft must be non-negative")

    sorted_scored = sorted(
        proposals_scored,
        key=lambda p: p[1],
        reverse=True,
    )

    selected: List[Tuple[Proposal, float]] = [
        p for p in sorted_scored if p[1] >= t_min
    ]

    if len(selected) < K_min:
        for p in sorted_scored[len(selected) :]:
            selected.append(p)
            if len(selected) >= K_min:
                break

    if K_cap_soft > 0 and len(selected) > K_cap_soft:
        selected = selected[:K_cap_soft]

    return selected
