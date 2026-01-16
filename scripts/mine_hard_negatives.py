from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

from src.promptgen.common.instances import extract_lesion_instances_xyz
from src.promptgen.common.metrics import match_prompts_to_gt
from src.promptgen.common.io import load_case_volumes
from src.promptgen.data.case_index import build_case_records
from src.promptgen.frontend.centernet.infer import infer_heatmap_xyz
from src.promptgen.frontend.peaks import extract_topk_peaks_3d_xyz
from src.promptgen.prompt.box_generator import generate_fixed_multiscale_boxes


def _write_jsonl(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine hard negatives (FP proposals)")
    parser.add_argument("--split-file", default="data/splits/train.txt")
    parser.add_argument("--frontend-checkpoint", default="models/frontend/centernet.pt")
    parser.add_argument("--output", default="outputs/hnm/hard_negs.jsonl")
    parser.add_argument("--topk-proposals", type=int, default=200)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    args = parser.parse_args()

    data_root = os.environ.get("PROMPTGEN_DATA_ROOT", "data/processed")

    cases = build_case_records(
        data_root=data_root,
        split_file=args.split_file,
        require_labels=True,
        allow_missing_body_mask=True,
    )

    hard_negs: List[dict] = []
    edge_mm_set = [12.0, 20.0, 32.0, 48.0]
    padding_ratio = 0.2

    for case in cases:
        image, label, body_mask = load_case_volumes(
            image_path=case.image_path,
            label_path=case.label_path,
            body_mask_path=case.body_mask_path,
        )

        gating = body_mask.array_xyz.astype(bool) if body_mask is not None else None
        heatmap_xyz = infer_heatmap_xyz(
            volume_xyz=image.array_xyz,
            checkpoint_path=args.frontend_checkpoint,
            device="cpu",
            amp=False,
        )

        centers = extract_topk_peaks_3d_xyz(
            heatmap_xyz=heatmap_xyz,
            gating_mask_xyz=gating,
            topk=max(1, int(args.topk_proposals)),
            neighborhood=1,
        )

        proposals = generate_fixed_multiscale_boxes(
            centers=centers,
            shape_xyz=image.shape_xyz,
            spacing_xyz_mm=image.spacing_xyz_mm,
            edge_mm_set=edge_mm_set,
            padding_ratio=padding_ratio,
        )

        proposals_scored = [(p.box_xyz_vox, float(p.heatmap_score), p) for p in proposals]
        proposals_scored.sort(key=lambda x: x[1], reverse=True)

        if args.score_threshold > 0:
            proposals_scored = [p for p in proposals_scored if p[1] >= args.score_threshold]

        if args.topk_proposals > 0:
            proposals_scored = proposals_scored[: args.topk_proposals]

        prompts_sorted = [(box, score) for box, score, _ in proposals_scored]
        lesions = extract_lesion_instances_xyz(label.array_xyz) if label is not None else []
        match_info = match_prompts_to_gt(prompts_sorted, lesions)
        matched = match_info["prompt_to_gt"]

        for idx, (box, score, _proposal) in enumerate(proposals_scored):
            if idx in matched:
                continue
            hard_negs.append(
                {
                    "case_id": case.case_id,
                    "box_xyz_vox": list(box),
                    "heatmap_score": float(score),
                }
            )

    _write_jsonl(args.output, hard_negs)


if __name__ == "__main__":
    main()
