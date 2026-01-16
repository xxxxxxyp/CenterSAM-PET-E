from __future__ import annotations

import argparse
import os
from typing import List

from src.promptgen.data.case_index import infer_domain, load_split_case_ids


def _build_paths(data_root: str, case_id: str) -> dict:
    """Build expected file paths for a case.

    Inputs: data_root (processed root), case_id (string).
    Outputs: dict with image/label/body_mask paths.
    Operation: concatenates standard folder structure with case id.
    """
    return {
        "image": os.path.join(data_root, "images", f"{case_id}.nii.gz"),
        "label": os.path.join(data_root, "labels", f"{case_id}.nii.gz"),
        "body_mask": os.path.join(data_root, "body_masks", f"{case_id}.nii.gz"),
    }


def _path_status(path: str) -> str:
    """Return a compact existence flag for a path.

    Inputs: path (string).
    Outputs: "Y" if exists else "N".
    Operation: filesystem existence check.
    """
    return "Y" if os.path.exists(path) else "N"


def _print_table(rows: List[List[str]]) -> None:
    """Print a simple aligned table.

    Inputs: rows (list of row lists, first row is header).
    Outputs: none (prints to stdout).
    Operation: computes column widths and prints padded columns.
    """
    if not rows:
        return
    col_widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print(" | ".join(r[i].ljust(col_widths[i]) for i in range(len(r))))


def main() -> None:
    """CLI entrypoint for inspecting split metadata and file presence.

    Inputs: command-line args (--data-root, --split-file, --require-labels, --limit).
    Outputs: none (prints a table).
    Operation: loads case ids, infers domains, checks file existence, prints table.
    """
    parser = argparse.ArgumentParser(description="Inspect cases meta and paths")
    parser.add_argument("--data-root", default="data/processed", help="data/processed root")
    parser.add_argument("--split-file", default="data/splits/dev_smoke.txt", help="split txt file")
    parser.add_argument("--require-labels", type=int, choices=[0, 1], default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    case_ids = load_split_case_ids(args.split_file)
    if args.limit is not None:
        case_ids = case_ids[: args.limit]

    rows: List[List[str]] = [
        [
            "case_id",
            "domain",
            "shape",
            "spacing",
            "image",
            "label",
            "body_mask",
        ]
    ]

    for case_id in case_ids:
        domain = infer_domain(case_id)
        paths = _build_paths(args.data_root, case_id)
        rows.append(
            [
                case_id,
                domain,
                "-",
                "-",
                _path_status(paths["image"]),
                _path_status(paths["label"]) if args.require_labels else "-",
                _path_status(paths["body_mask"]),
            ]
        )

    _print_table(rows)


if __name__ == "__main__":
    main()
