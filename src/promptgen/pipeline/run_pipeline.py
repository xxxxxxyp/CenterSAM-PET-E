from __future__ import annotations

import logging
import os
from typing import Optional

from ..common.instances import extract_lesion_instances_xyz
from ..common.io import load_case_volumes
from ..data.case_index import CaseRecord
from ..frontend.dummy import dummy_from_label_centers, dummy_random_centers
from ..prompt.box_generator import generate_fixed_multiscale_boxes
from ..prompt.select_k import select_prompts_recall_first
from .formats import Prompt, PromptFile, save_prompts_json, clamp_score_01

LOGGER = logging.getLogger(__name__)


def _build_output_path(output_root: str, case_id: str) -> str:
    """Build output prompts.json path for a case.

    Inputs: output_root (output root dir), case_id (string).
    Outputs: prompts JSON path.
    Operation: joins output_root with case_id + ".json".
    """
    return os.path.join(output_root, f"{case_id}.json")


def run_case_prompt_generation(
    case: CaseRecord,
    data_root: str,
    output_root: str,
    frontend_type: str,  # "dummy_from_label"|"dummy_random"
    edge_mm_set: list,
    padding_ratio: float,
    K_min: int,
    t_min: float,
    K_cap_soft: int,
    num_random_centers: int,
    seed: int,
) -> str:
    """Run prompt generation for a single case and save prompts JSON.

    Inputs: case metadata, paths, frontend selection, generator params, and
    selection params.
    Outputs: path to prompts.json.
    Operation: loads volumes, selects centers, generates boxes, selects prompts,
    builds PromptFile, saves JSON, and logs per-case summary.
    """
    image, label, body_mask = load_case_volumes(
        image_path=case.image_path,
        label_path=case.label_path,
        body_mask_path=case.body_mask_path,
    )

    num_gt_lesions: Optional[int] = None
    if label is not None:
        num_gt_lesions = len(extract_lesion_instances_xyz(label.array_xyz))

    if frontend_type == "dummy_from_label":
        if label is None:
            raise ValueError("dummy_from_label requires label_path")
        centers = dummy_from_label_centers(label.array_xyz)
    elif frontend_type == "dummy_random":
        gating = body_mask.array_xyz.astype(bool) if body_mask is not None else None
        centers = dummy_random_centers(
            shape_xyz=image.shape_xyz,
            num_centers=num_random_centers,
            seed=seed,
            gating_mask_xyz=gating,
        )
    else:
        raise ValueError(f"unsupported frontend_type: {frontend_type}")

    proposals = generate_fixed_multiscale_boxes(
        centers=centers,
        shape_xyz=image.shape_xyz,
        spacing_xyz_mm=image.spacing_xyz_mm,
        edge_mm_set=edge_mm_set,
        padding_ratio=padding_ratio,
    )

    proposals_scored = [(p, float(p.heatmap_score)) for p in proposals]
    selected = select_prompts_recall_first(
        proposals_scored=proposals_scored,
        K_min=K_min,
        t_min=t_min,
        K_cap_soft=K_cap_soft,
    )

    prompts = []
    for idx, (proposal, score) in enumerate(selected, start=1):
        prompt_id = f"p{idx:04d}"
        prompts.append(
            Prompt(
                prompt_id=prompt_id,
                box_xyz_vox=proposal.box_xyz_vox,
                score=clamp_score_01(score),
                source={
                    "center_xyz_vox": proposal.center_xyz_vox,
                    "heatmap_score": float(proposal.heatmap_score),
                    "generator": proposal.generator,
                    "generator_params": proposal.generator_params,
                },
            )
        )

    prompt_file = PromptFile(
        case_id=case.case_id,
        domain=case.domain,
        image_path=case.image_path,
        shape_xyz=image.shape_xyz,
        spacing_xyz_mm=image.spacing_xyz_mm,
        prompts=prompts,
        run={
            "frontend_type": frontend_type,
            "edge_mm_set": edge_mm_set,
            "padding_ratio": padding_ratio,
            "K_min": K_min,
            "t_min": t_min,
            "K_cap_soft": K_cap_soft,
            "num_random_centers": num_random_centers,
            "seed": seed,
        },
    )

    out_path = _build_output_path(output_root, case.case_id)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_prompts_json(out_path, prompt_file)

    top_scores = [p.score for p in prompts[:5]]
    LOGGER.info(
        "case_pipeline_summary case_id=%s domain=%s shape_xyz=%s spacing_xyz_mm=%s "
        "num_gt_lesions=%s num_centers=%d num_proposals_generated=%d K_out=%d top5_scores=%s",
        case.case_id,
        case.domain,
        image.shape_xyz,
        image.spacing_xyz_mm,
        num_gt_lesions,
        len(centers),
        len(proposals),
        len(prompts),
        top_scores,
    )

    return out_path
