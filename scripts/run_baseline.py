from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, Tuple


def _ensure_repo_root_on_path() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


def _run_generate_prompts(args: argparse.Namespace) -> None:
    _ensure_repo_root_on_path()
    from scripts import generate_prompts

    argv = [
        "generate_prompts.py",
        "--data-root",
        args.data_root,
        "--split-file",
        args.split_file,
        "--output-root",
        args.output_root,
        "--frontend",
        "centernet",
        "--centernet-checkpoint-path",
        args.checkpoint,
        "--device",
        args.device,
        "--amp",
        "1" if args.amp else "0",
        "--n-peaks",
        str(args.n_peaks),
        "--edge-mm-set",
        args.edge_mm_set,
        "--padding-ratio",
        str(args.padding_ratio),
        "--K-cap-soft",
        str(args.K_cap_soft),
        "--use-diversify",
        "1",
        "--eps-mm",
        str(args.eps_mm),
        "--max-per-cell",
        str(args.max_per_cell),
        "--seed",
        str(args.seed),
        "--require-labels",
        "0",
    ]

    sys.argv = argv
    generate_prompts.main()


def _run_evaluate_prompts(args: argparse.Namespace) -> None:
    _ensure_repo_root_on_path()
    from scripts import evaluate_prompts

    argv = [
        "evaluate_prompts.py",
        "--data-root",
        args.data_root,
        "--prompts-root",
        args.output_root,
        "--split-file",
        args.split_file,
        "--out",
        args.metrics_out,
        "--ks",
        "20,50,200",
    ]

    sys.argv = argv
    evaluate_prompts.main()


def _read_metrics(path: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    macro_recall: Dict[str, float] = {"20": 0.0, "50": 0.0, "200": 0.0}
    mean_fp: Dict[str, float] = {"20": 0.0, "50": 0.0, "200": 0.0}
    rows = 0

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            for k in ("20", "50", "200"):
                macro_recall[k] += float(row[f"recall@{k}"])
                mean_fp[k] += float(row[f"fp@{k}"])

    if rows == 0:
        raise ValueError("no rows found in metrics CSV")

    for k in ("20", "50", "200"):
        macro_recall[k] /= rows
        mean_fp[k] /= rows

    return macro_recall, mean_fp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run centernet baseline: generate + evaluate")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--split-file", default="data/splits/val.txt")
    parser.add_argument("--checkpoint", default="models/frontend/centernet.pt")
    parser.add_argument("--output-root", default="outputs/prompts_centernet_baseline")
    parser.add_argument("--metrics-out", default="outputs/metrics/centernet_val.csv")
    parser.add_argument("--n-peaks", type=int, default=1000)
    parser.add_argument("--K-cap-soft", type=int, default=200)
    parser.add_argument("--edge-mm-set", default="12,20,32,48")
    parser.add_argument("--padding-ratio", type=float, default=0.2)
    parser.add_argument("--eps-mm", type=float, default=15)
    parser.add_argument("--max-per-cell", type=int, default=5)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--amp", type=int, choices=[0, 1], default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        _run_generate_prompts(args)
        _run_evaluate_prompts(args)
        macro_recall, mean_fp = _read_metrics(args.metrics_out)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        raise SystemExit(code) from exc

    print(
        "summary macro_recall@20={:.6f} macro_recall@50={:.6f} macro_recall@200={:.6f} "
        "mean_fp@20={:.3f} mean_fp@50={:.3f} mean_fp@200={:.3f}".format(
            macro_recall["20"],
            macro_recall["50"],
            macro_recall["200"],
            mean_fp["20"],
            mean_fp["50"],
            mean_fp["200"],
        )
    )


if __name__ == "__main__":
    main()
