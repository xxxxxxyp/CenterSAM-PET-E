from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .geometry import BoxXYZ, box_iou_xyz, center_in_box_xyz
from .instances import LesionInstance


@dataclass(frozen=True)
class CasePromptMetrics:
    case_id: str
    K_out: int
    num_gt: int
    recall_at_k: Dict[int, float]
    fp_at_k: Dict[int, int]


def match_prompts_to_gt(
    prompts_sorted: Sequence[Tuple[BoxXYZ, float]],
    lesions: Sequence[LesionInstance],
    iou_threshold: float = 0.1,
) -> Dict:
    """Greedy one-to-one matching between prompts and GT lesions.

    Inputs: prompts_sorted (list of (box, score) sorted desc), lesions (GT list),
    iou_threshold (float).
    Outputs: dict with prompt_to_gt mapping and unmatched lists.
    Operation: iterates prompts by score; for each prompt, assigns the first
    unmatched GT it covers by center-in-box or IoU>=threshold.
    """
    matched_gt: set[int] = set()
    prompt_to_gt: Dict[int, int] = {}

    for p_idx, (box, _score) in enumerate(prompts_sorted):
        for g_idx, lesion in enumerate(lesions):
            if g_idx in matched_gt:
                continue
            covered = center_in_box_xyz(lesion.center_xyz, box) or (
                box_iou_xyz(box, lesion.bbox_xyz) >= iou_threshold
            )
            if covered:
                matched_gt.add(g_idx)
                prompt_to_gt[p_idx] = g_idx
                break

    unmatched_gt = [i for i in range(len(lesions)) if i not in matched_gt]
    unmatched_prompts = [
        i for i in range(len(prompts_sorted)) if i not in prompt_to_gt
    ]
    return {
        "prompt_to_gt": prompt_to_gt,
        "unmatched_gt": unmatched_gt,
        "unmatched_prompts": unmatched_prompts,
    }


def compute_prompt_metrics(
    case_id: str,
    prompts_sorted: Sequence[Tuple[BoxXYZ, float]],
    lesions: Sequence[LesionInstance],
    ks: Sequence[int] = (10, 20, 50, 100, 200),
) -> CasePromptMetrics:
    """Compute Recall@K and FP@K for a case using greedy matching.

    Inputs: case_id (string), prompts_sorted (sorted list), lesions (GT list),
    ks (sequence of K values).
    Outputs: CasePromptMetrics.
    Operation: matches prompts to GT once, then aggregates metrics per K.
    """
    match_info = match_prompts_to_gt(prompts_sorted, lesions)
    prompt_to_gt = match_info["prompt_to_gt"]

    num_gt = len(lesions)
    recall_at_k: Dict[int, float] = {}
    fp_at_k: Dict[int, int] = {}

    for k in ks:
        k_eff = min(k, len(prompts_sorted))
        matched_prompts = [p for p in prompt_to_gt.keys() if p < k_eff]
        matched_gt = {prompt_to_gt[p] for p in matched_prompts}
        recall = (len(matched_gt) / num_gt) if num_gt > 0 else 0.0
        fp = k_eff - len(matched_prompts)
        recall_at_k[int(k)] = recall
        fp_at_k[int(k)] = fp

    return CasePromptMetrics(
        case_id=case_id,
        K_out=len(prompts_sorted),
        num_gt=num_gt,
        recall_at_k=recall_at_k,
        fp_at_k=fp_at_k,
    )
