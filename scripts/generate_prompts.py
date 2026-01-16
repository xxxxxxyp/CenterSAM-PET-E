from __future__ import annotations

import argparse
import logging
import sys
from typing import List

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
    parser.add_argument("--frontend", default="dummy_from_label", choices=["dummy_from_label", "dummy_random"])
    parser.add_argument("--require-labels", type=int, choices=[0, 1], default=0)
    parser.add_argument("--edge-mm-set", default="12,20,32,48")
    parser.add_argument("--padding-ratio", type=float, default=0.2)
    parser.add_argument("--K-min", type=int, default=20)
    parser.add_argument("--t-min", type=float, default=0.05)
    parser.add_argument("--K-cap-soft", type=int, default=200)
    parser.add_argument("--num-random-centers", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

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
