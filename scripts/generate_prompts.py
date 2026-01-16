from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

from src.promptgen.data.case_index import build_case_records
from src.promptgen.pipeline.run_pipeline import run_case_prompt_generation

LOGGER = logging.getLogger(__name__)


def _parse_edge_mm_set(edge_mm_set: str) -> List[float]:
    """Parse comma-separated edge sizes into float list.

    Inputs: edge_mm_set (string).
    Outputs: list of floats.
    Operation: splits by comma and converts to float, ignoring empty entries.
    """
    edges: List[float] = []
    for part in edge_mm_set.split(","):
        part = part.strip()
        if not part:
            continue
        edges.append(float(part))
    return edges


def _compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute sha256 of effective config dict.

    Inputs: config (dict).
    Outputs: sha256 hex digest string.
    Operation: dumps JSON with sorted keys then hashes.
    """
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_patch_shape(patch_shape: str) -> Tuple[int, int, int]:
    """Parse comma-separated patch shape into int tuple.

    Inputs: patch_shape (string like "64,64,64").
    Outputs: tuple of 3 ints.
    Operation: splits by comma, converts to int, validates length.
    """
    parts = [p.strip() for p in patch_shape.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("patch-shape must be three comma-separated ints")
    try:
        vals = tuple(int(v) for v in parts)
    except ValueError as exc:
        raise ValueError("patch-shape must be ints") from exc
    if not all(v > 0 for v in vals):
        raise ValueError("patch-shape values must be > 0")
    return vals  # type: ignore[return-value]


def main() -> None:
    """CLI entrypoint to generate prompts for a split.

    Inputs: command-line args.
    Outputs: none (writes prompts JSON files).
    Operation: builds case records, runs pipeline per case, collects failures
    and exits nonzero if any case failed.
    """
    parser = argparse.ArgumentParser(description="Generate prompt files")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--split-file", default="data/splits/dev_smoke.txt")
    parser.add_argument("--output-root", default="outputs/prompts")
    parser.add_argument(
        "--frontend",
        default="dummy_from_label",
        choices=["dummy_from_label", "dummy_random", "centernet"],
    )
    parser.add_argument("--require-labels", type=int, choices=[0, 1], default=0)
    parser.add_argument("--edge-mm-set", default="12,20,32,48")
    parser.add_argument("--padding-ratio", type=float, default=0.2)
    parser.add_argument("--K-min", type=int, default=20)
    parser.add_argument("--t-min", type=float, default=0.05)
    parser.add_argument("--K-cap-soft", type=int, default=200)
    parser.add_argument("--num-random-centers", type=int, default=500)
    parser.add_argument("--model-path", default="models/frontend/model.pth")
    parser.add_argument("--N-peaks", type=int, default=1000)
    parser.add_argument("--use-classifier", type=int, choices=[0, 1], default=0)
    parser.add_argument("--classifier-model-path", default="models/classifier/model.pth")
    parser.add_argument("--w-cls", type=float, default=0.7)
    parser.add_argument("--patch-shape", default="64,64,64")
    parser.add_argument("--use-diversify", type=int, choices=[0, 1], default=0)
    parser.add_argument("--eps-mm", type=float, default=15)
    parser.add_argument("--max-per-cell", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config_path = "cli"
    git_commit = os.environ.get("GIT_COMMIT")

    effective_config: Dict[str, Any] = {
        "data_root": args.data_root,
        "split_file": args.split_file,
        "output_root": args.output_root,
        "frontend": args.frontend,
        "require_labels": args.require_labels,
        "edge_mm_set": args.edge_mm_set,
        "padding_ratio": args.padding_ratio,
        "K_min": args.K_min,
        "t_min": args.t_min,
        "K_cap_soft": args.K_cap_soft,
        "num_random_centers": args.num_random_centers,
        "model_path": args.model_path,
        "N_peaks": args.N_peaks,
        "use_classifier": args.use_classifier,
        "classifier_model_path": args.classifier_model_path,
        "w_cls": args.w_cls,
        "patch_shape": args.patch_shape,
        "use_diversify": args.use_diversify,
        "eps_mm": args.eps_mm,
        "max_per_cell": args.max_per_cell,
        "seed": args.seed,
    }
    config_hash = _compute_config_hash(effective_config)

    require_labels = bool(args.require_labels)
    if args.frontend == "dummy_from_label":
        require_labels = True

    try:
        case_records = build_case_records(
            data_root=args.data_root,
            split_file=args.split_file,
            require_labels=require_labels,
            allow_missing_body_mask=True,
        )
    except ValueError as exc:
        LOGGER.error("build_case_records failed: %s", exc)
        sys.exit(2)

    failed: List[str] = []
    for case in case_records:
        try:
            run_case_prompt_generation(
                case=case,
                data_root=args.data_root,
                output_root=args.output_root,
                frontend_type=args.frontend,
                edge_mm_set=_parse_edge_mm_set(args.edge_mm_set),
                padding_ratio=args.padding_ratio,
                K_min=args.K_min,
                t_min=args.t_min,
                K_cap_soft=args.K_cap_soft,
                num_random_centers=args.num_random_centers,
                seed=args.seed,
                model_path=args.model_path,
                N_peaks=args.N_peaks,
                use_classifier=bool(args.use_classifier),
                classifier_model_path=args.classifier_model_path,
                w_cls=args.w_cls,
                patch_shape_xyz=_parse_patch_shape(args.patch_shape),
                config_path=config_path,
                config_hash=config_hash,
                git_commit=git_commit,
                use_diversify=bool(args.use_diversify),
                eps_mm=args.eps_mm,
                max_per_cell=args.max_per_cell,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("case failed: %s (%s)", case.case_id, exc)
            failed.append(case.case_id)

    if failed:
        print("failed_case_ids:", ",".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
