from __future__ import annotations

import argparse
import csv
import logging
import os
from typing import Dict, List, Sequence, Tuple

from src.promptgen.common.instances import extract_lesion_instances_xyz
from src.promptgen.common.io import load_nifti_xyz
from src.promptgen.common.metrics import compute_prompt_metrics, match_prompts_to_gt
from src.promptgen.pipeline.formats import load_prompts_json
from src.promptgen.data.case_index import load_split_case_ids

LOGGER = logging.getLogger(__name__)


def _parse_ks(ks_str: str) -> Tuple[int, ...]:
    """Parse a comma-separated K list into a sorted tuple of ints.

    Inputs: ks_str (e.g., "10,20,50").
    Outputs: sorted tuple of unique ints.
    Operation: split, cast to int, drop non-positive values, sort unique.
    """
    ks: List[int] = []
    for part in ks_str.split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k > 0:
            ks.append(k)
    return tuple(sorted(set(ks)))


def _prompts_path(prompts_root: str, case_id: str) -> str:
    """Build prompts.json path for a case.

    Inputs: prompts_root (root dir), case_id (string).
    Outputs: prompts JSON path.
    Operation: joins root with case_id + ".json".
    """
    return os.path.join(prompts_root, f"{case_id}.json")


def _label_path(data_root: str, case_id: str) -> str:
    """Build label path for a case.

    Inputs: data_root (processed root), case_id (string).
    Outputs: label path.
    Operation: joins data_root/labels with case_id + ".nii.gz".
    """
    return os.path.join(data_root, "labels", f"{case_id}.nii.gz")


def _sort_prompts_for_eval(prompt_file) -> List[Tuple[Tuple[int, int, int, int, int, int], float]]:
    """Convert prompts to sorted list of (box, score) for evaluation.

    Inputs: prompt_file (PromptFile).
    Outputs: list of (box_xyz, score) sorted by score desc, id asc.
    Operation: sorts prompts and extracts fields.
    """
    prompts_sorted = sorted(prompt_file.prompts, key=lambda p: (-p.score, p.prompt_id))
    return [(p.box_xyz_vox, float(p.score)) for p in prompts_sorted]


def _write_csv(out_path: str, rows: List[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    """Write rows to CSV with header.

    Inputs: out_path (file path), rows (list of dict), fieldnames (header order).
    Outputs: none.
    Operation: creates parent dir and writes CSV.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """CLI entrypoint for evaluating prompts vs GT lesions.

    Inputs: command-line args.
    Outputs: none (writes CSV + logs summary).
    Operation: loads split, reads prompts and labels, computes Recall@K/FP@K,
    writes per-case CSV and logs macro/micro summary.
    """
    parser = argparse.ArgumentParser(description="Evaluate prompt quality")
    parser.add_argument("--data-root", default="data/processed", help="data/processed root")
    parser.add_argument("--prompts-root", default="outputs/prompts", help="prompts root")
    parser.add_argument("--split-file", default="data/splits/val.txt", help="split txt file")
    parser.add_argument("--out", default="outputs/metrics/prompts_eval_val.csv", help="output csv")
    parser.add_argument("--ks", default="10,20,50,100,200", help="comma-separated Ks")
    parser.add_argument("--iou-threshold", type=float, default=0.1)
    args = parser.parse_args()

    ks = _parse_ks(args.ks)
    case_ids = load_split_case_ids(args.split_file)

    rows: List[Dict[str, str]] = []
    macro_recall_sum: Dict[int, float] = {k: 0.0 for k in ks}
    micro_matched_sum: Dict[int, int] = {k: 0 for k in ks}
    fp_sum: Dict[int, int] = {k: 0 for k in ks}
    total_gt = 0
    cases_processed = 0

    for case_id in case_ids:
        label_path = _label_path(args.data_root, case_id)
        prompts_path = _prompts_path(args.prompts_root, case_id)
        if not os.path.exists(label_path):
            LOGGER.warning("missing label for case_id=%s", case_id)
            continue
        if not os.path.exists(prompts_path):
            LOGGER.warning("missing prompts for case_id=%s", case_id)
            continue

        label_vol = load_nifti_xyz(label_path, dtype=None)
        lesions = extract_lesion_instances_xyz(label_vol.array_xyz)
        prompt_file = load_prompts_json(prompts_path)
        prompts_sorted = _sort_prompts_for_eval(prompt_file)

        metrics = compute_prompt_metrics(case_id, prompts_sorted, lesions, ks=ks)
        match_info = match_prompts_to_gt(
            prompts_sorted, lesions, iou_threshold=args.iou_threshold
        )
        prompt_to_gt = match_info["prompt_to_gt"]

        row: Dict[str, str] = {
            "case_id": case_id,
            "num_gt": str(metrics.num_gt),
            "K_out": str(metrics.K_out),
        }

        for k in ks:
            k_eff = min(k, len(prompts_sorted))
            matched_prompts = [p for p in prompt_to_gt.keys() if p < k_eff]
            matched_gt = {prompt_to_gt[p] for p in matched_prompts}

            recall_k = metrics.recall_at_k[int(k)]
            fp_k = metrics.fp_at_k[int(k)]
            row[f"recall@{k}"] = f"{recall_k:.6f}"
            row[f"fp@{k}"] = str(fp_k)

            macro_recall_sum[k] += recall_k
            micro_matched_sum[k] += len(matched_gt)
            fp_sum[k] += fp_k

        rows.append(row)
        total_gt += metrics.num_gt
        cases_processed += 1

        LOGGER.info(
            "case_metrics case_id=%s num_gt=%d K_out=%d recall_at_k=%s fp_at_k=%s",
            case_id,
            metrics.num_gt,
            metrics.K_out,
            metrics.recall_at_k,
            metrics.fp_at_k,
        )

    fieldnames = ["case_id", "num_gt", "K_out"] + [
        f"recall@{k}" for k in ks
    ] + [f"fp@{k}" for k in ks]
    _write_csv(args.out, rows, fieldnames)

    if cases_processed == 0:
        LOGGER.warning("no cases processed")
        return

    macro_recall = {k: macro_recall_sum[k] / cases_processed for k in ks}
    micro_recall = {k: (micro_matched_sum[k] / total_gt) if total_gt > 0 else 0.0 for k in ks}
    mean_fp = {k: fp_sum[k] / cases_processed for k in ks}

    LOGGER.info(
        "summary cases=%d total_gt=%d macro_recall=%s micro_recall=%s mean_fp=%s",
        cases_processed,
        total_gt,
        macro_recall,
        micro_recall,
        mean_fp,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
