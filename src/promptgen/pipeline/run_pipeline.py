from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from ..common.instances import extract_lesion_instances_xyz
from ..common.io import load_case_volumes
from ..data.case_index import CaseRecord
from ..frontend.dummy import dummy_from_label_centers, dummy_random_centers
from ..frontend.peaks import extract_topk_peaks_3d_xyz
from ..frontend.centernet.infer import infer_heatmap_xyz
from ..prompt.box_generator import generate_fixed_multiscale_boxes
from ..classifier.infer import infer_classifier_scores
from ..prompt.diversify import diversify_by_grid_cell
from ..prompt.select_k import select_prompts_recall_first
from .formats import Prompt, PromptFile, save_prompts_json, clamp_score_01

LOGGER = logging.getLogger(__name__)


def _fuse_scores(
    hm_scores: list,
    cls_scores: list,
    w_cls: float,
) -> list:
    """Fuse heatmap and classifier scores with clamping.

    Inputs: hm_scores (list), cls_scores (list), w_cls (float).
    Outputs: list of fused scores in [0,1].
    Operation: convex combination with clamp_score_01.
    """
    if len(hm_scores) != len(cls_scores):
        raise ValueError("hm_scores and cls_scores must have same length")
    fused: list = []
    for hm, cls in zip(hm_scores, cls_scores):
        fused_score = clamp_score_01(float(w_cls) * float(cls) + (1.0 - float(w_cls)) * float(hm))
        fused.append(fused_score)
    return fused


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
    frontend_type: str,  # "dummy_from_label"|"dummy_random"|"centernet"
    edge_mm_set: list,
    padding_ratio: float,
    K_min: int,
    t_min: float,
    K_cap_soft: int,
    num_random_centers: int,
    seed: int,
    centernet_checkpoint_path: str,
    device: str,
    amp: bool,
    n_peaks: int,
    use_classifier: bool,
    classifier_model_path: str,
    w_cls: float,
    patch_shape_xyz: tuple,
    config_path: str,
    config_hash: str,
    git_commit: Optional[str],
    use_diversify: bool,
    eps_mm: float,
    max_per_cell: int,
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

    heatmap_stats = None
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
    elif frontend_type == "centernet":
        if not centernet_checkpoint_path:
            raise ValueError("centernet requires centernet_checkpoint_path")
        gating = body_mask.array_xyz.astype(bool) if body_mask is not None else None
        heatmap_xyz = infer_heatmap_xyz(
            volume_xyz=image.array_xyz,
            checkpoint_path=centernet_checkpoint_path,
            device=device,
            amp=amp,
        )
        heatmap_stats = {
            "min": float(np.min(heatmap_xyz)),
            "max": float(np.max(heatmap_xyz)),
            "mean": float(np.mean(heatmap_xyz)),
        }
        centers = extract_topk_peaks_3d_xyz(
            heatmap_xyz=heatmap_xyz,
            gating_mask_xyz=gating,
            topk=n_peaks,
            neighborhood=1,
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
    num_after_diversify = len(proposals_scored)
    if use_classifier:
        cls_scores = infer_classifier_scores(
            volume_xyz=image.array_xyz,
            proposals=proposals,
            model_path=classifier_model_path,
            device="cpu",
            patch_shape_xyz=patch_shape_xyz,
        )
        hm_scores = [p.heatmap_score for p in proposals]
        fused_scores = _fuse_scores(hm_scores=hm_scores, cls_scores=cls_scores, w_cls=w_cls)
        proposals_scored = list(zip(proposals, fused_scores))
    if use_diversify:
        proposals_scored = diversify_by_grid_cell(
            proposals_scored=proposals_scored,
            spacing_xyz_mm=image.spacing_xyz_mm,
            eps_mm=eps_mm,
            max_per_cell=max_per_cell,
        )
        num_after_diversify = len(proposals_scored)
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
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_path": config_path,
            "config_hash": config_hash,
            **({"git_commit": git_commit} if git_commit else {}),
            "frontend_type": frontend_type,
            "centernet_checkpoint_path": centernet_checkpoint_path,
            "device": device,
            "amp": bool(amp),
            "n_peaks": int(n_peaks),
            **({"heatmap_stats": heatmap_stats} if heatmap_stats is not None else {}),
            "edge_mm_set": edge_mm_set,
            "padding_ratio": padding_ratio,
            "K_min": K_min,
            "t_min": t_min,
            "K_cap_soft": K_cap_soft,
            "num_random_centers": num_random_centers,
            "seed": seed,
            "use_classifier": bool(use_classifier),
            "classifier_model_path": classifier_model_path,
            "w_cls": float(w_cls),
            "patch_shape_xyz": tuple(patch_shape_xyz),
            "use_diversify": bool(use_diversify),
            "eps_mm": float(eps_mm),
            "max_per_cell": int(max_per_cell),
            "num_peaks": int(len(centers)),
            "num_centers": int(len(centers)),
            "num_proposals_generated": int(len(proposals)),
            "num_after_diversify": int(num_after_diversify),
            "K_out": int(len(prompts)),
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
