import os
from typing import List

import pytest

from src.promptgen.data.case_index import load_split_case_ids
from src.promptgen.frontend.centernet.dataset import CenterNetDataset, CenterNetTrainConfig


def _resolve_split_file() -> str:
    env_path = os.environ.get("PROMPTGEN_SPLIT_FILE")
    if env_path:
        return env_path
    dev_smoke = os.path.join("data", "splits", "dev_smoke.txt")
    if os.path.exists(dev_smoke):
        return dev_smoke
    val_list = os.path.join("data", "splits", "val_list.txt")
    return val_list


def _filter_existing_cases(case_ids: List[str], data_root: str) -> List[str]:
    existing: List[str] = []
    for case_id in case_ids:
        image_path = os.path.join(data_root, "images", f"{case_id}.nii.gz")
        label_path = os.path.join(data_root, "labels", f"{case_id}.nii.gz")
        if os.path.exists(image_path) and os.path.exists(label_path):
            existing.append(case_id)
    return existing


def test_centernet_dataset_shapes_devset(tmp_path):
    """Load one real case and validate tensor shapes.

    Inputs: tmp_path.
    Outputs: none (asserts shapes).
    Operation: uses dataset to get image/target shapes.
    """
    data_root = os.environ.get("PROMPTGEN_DATA_ROOT", os.path.join("data", "processed"))
    split_file = _resolve_split_file()
    if not os.path.exists(split_file):
        pytest.skip("split file not found for dataset shape test")

    case_ids = load_split_case_ids(split_file)
    case_ids = _filter_existing_cases(case_ids, data_root)
    if not case_ids:
        pytest.skip("no cases with image+label for dataset shape test")

    case_id = case_ids[0]
    case = type("Case", (), {
        "case_id": case_id,
        "domain": "FL",
        "image_path": os.path.join(data_root, "images", f"{case_id}.nii.gz"),
        "label_path": os.path.join(data_root, "labels", f"{case_id}.nii.gz"),
        "body_mask_path": None,
    })

    ds = CenterNetDataset([case], CenterNetTrainConfig(sigma_vox=2.0), require_labels=True)
    item = ds[0]

    image = item["image"]
    target = item["target_heatmap"]

    assert image.ndim == 4
    assert target.ndim == 4
    assert image.shape[0] == 1
    assert target.shape[0] == 1
    assert image.shape[1:] == target.shape[1:]
