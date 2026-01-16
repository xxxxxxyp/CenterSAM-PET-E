from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    domain: str  # "FL"|"DLBCL"|future
    image_path: str
    label_path: Optional[str]
    body_mask_path: Optional[str]


def parse_case_id(line: str) -> str:
    """Parse a split-file line into a 4-digit case id.

    Inputs: line (raw line from split file).
    Outputs: case_id string.
    Operation: strip whitespace, validate 4-digit numeric, raise on error.
    """
    case_id = line.strip()
    if len(case_id) != 4 or not case_id.isdigit():
        raise ValueError(f"invalid case_id: {case_id!r}")
    return case_id


def infer_domain(case_id: str) -> str:
    """Infer domain from case_id prefix.

    Inputs: case_id (4-digit string).
    Outputs: domain string ("FL" or "DLBCL").
    Operation: map leading digit '0'->FL, '1'->DLBCL; raise otherwise.
    """
    if not case_id or not case_id[0].isdigit():
        raise ValueError(f"invalid case_id for domain inference: {case_id!r}")
    if case_id[0] == "0":
        return "FL"
    if case_id[0] == "1":
        return "DLBCL"
    raise ValueError(f"unknown domain for case_id: {case_id!r}")


def load_split_case_ids(splits_path: str) -> List[str]:
    """Load case ids from a split file.

    Inputs: splits_path (path to .txt split file).
    Outputs: list of case_id strings.
    Operation: skip empty/comment lines, parse each line with validation.
    """
    case_ids: List[str] = []
    with open(splits_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            case_ids.append(parse_case_id(stripped))
    return case_ids


def _path_if_exists(path: str) -> Optional[str]:
    """Return path if it exists, otherwise None.

    Inputs: path (string).
    Outputs: same path or None.
    Operation: filesystem existence check.
    """
    return path if os.path.exists(path) else None


def build_case_records(
    data_root: str,  # path to data/processed
    split_file: str,  # path to data/splits/train.txt
    require_labels: bool,
    allow_missing_body_mask: bool = True,
) -> List[CaseRecord]:
    """Build CaseRecord list and validate required files exist.

    Inputs: data_root (processed root), split_file (split txt),
    require_labels (bool), allow_missing_body_mask (bool).
    Outputs: list of CaseRecord.
    Operation: load case ids, infer domain, resolve paths, track missing files,
    log summary, and raise ValueError with missing lists when required files are absent.
    """
    case_ids = load_split_case_ids(split_file)

    missing_images: List[str] = []
    missing_labels: List[str] = []
    missing_body_masks: List[str] = []
    records: List[CaseRecord] = []

    for case_id in case_ids:
        domain = infer_domain(case_id)
        image_path = os.path.join(data_root, "images", f"{case_id}.nii.gz")
        label_path = os.path.join(data_root, "labels", f"{case_id}.nii.gz")
        body_mask_path = os.path.join(data_root, "body_masks", f"{case_id}.nii.gz")

        if not os.path.exists(image_path):
            missing_images.append(case_id)

        label_path_exists = _path_if_exists(label_path)
        if require_labels and label_path_exists is None:
            missing_labels.append(case_id)

        body_mask_path_exists = _path_if_exists(body_mask_path)
        if not allow_missing_body_mask and body_mask_path_exists is None:
            missing_body_masks.append(case_id)

        records.append(
            CaseRecord(
                case_id=case_id,
                domain=domain,
                image_path=image_path,
                label_path=label_path_exists if not require_labels else label_path,
                body_mask_path=body_mask_path_exists,
            )
        )

    LOGGER.info(
        "case_index_summary total_case_ids=%d missing_images=%s missing_labels=%s missing_body_masks=%s",
        len(case_ids),
        missing_images,
        missing_labels if require_labels else [],
        missing_body_masks if not allow_missing_body_mask else [],
    )

    if missing_images or (require_labels and missing_labels) or (
        not allow_missing_body_mask and missing_body_masks
    ):
        raise ValueError(
            "missing files: "
            f"images={missing_images} "
            f"labels={missing_labels if require_labels else []} "
            f"body_masks={missing_body_masks if not allow_missing_body_mask else []}"
        )

    return records
