import os
import sys
from typing import List
import importlib.util

import numpy as np
import pytest
import torch

from src.promptgen.data.case_index import load_split_case_ids
from src.promptgen.pipeline.formats import load_prompts_json, validate_prompt_file_dict
from src.promptgen.frontend.centernet.model import CenterNetConfig, CenterNetPET


def _resolve_split_file() -> str:
    """Resolve split file path for smoke test.

    Inputs: none.
    Outputs: split file path string.
    Operation: checks env override, then dev_smoke, then val_list.
    """
    env_path = os.environ.get("PROMPTGEN_SPLIT_FILE")
    if env_path:
        return env_path
    dev_smoke = os.path.join("data", "splits", "dev_smoke.txt")
    if os.path.exists(dev_smoke):
        return dev_smoke
    val_list = os.path.join("data", "splits", "val_list.txt")
    return val_list


def _filter_existing_cases(case_ids: List[str], data_root: str) -> List[str]:
    """Filter case ids that have existing image files.

    Inputs: case_ids (list), data_root (processed root).
    Outputs: filtered list.
    Operation: keeps ids with image file present under data_root/images.
    """
    existing: List[str] = []
    for case_id in case_ids:
        image_path = os.path.join(data_root, "images", f"{case_id}.nii.gz")
        if os.path.exists(image_path):
            existing.append(case_id)
    return existing


def _save_random_checkpoint(path: str) -> None:
    cfg = CenterNetConfig(in_channels=1, base_channels=8, num_blocks=2, use_batchnorm=False)
    model = CenterNetPET(cfg)
    ckpt = {"model_state_dict": model.state_dict(), "config": cfg.__dict__}
    torch.save(ckpt, path)


def test_e2e_smoke_centernet_devset(tmp_path, monkeypatch):
    """Run centernet frontend on 1 case with random checkpoint.

    Inputs: tmp_path, monkeypatch.
    Outputs: none (asserts outputs).
    Operation: runs generate_prompts and validates prompt schema.
    """
    data_root = os.environ.get("PROMPTGEN_DATA_ROOT", os.path.join("data", "processed"))
    split_file = _resolve_split_file()
    if not os.path.exists(split_file):
        pytest.skip("split file not found for centernet smoke test")

    case_ids = load_split_case_ids(split_file)
    case_ids = _filter_existing_cases(case_ids, data_root)
    if not case_ids:
        pytest.skip("no cases with existing images for centernet smoke test")

    case_id = case_ids[0]
    tmp_split = tmp_path / "smoke_split.txt"
    tmp_split.write_text(f"{case_id}\n", encoding="utf-8")

    output_root = tmp_path / "prompts"
    ckpt_path = tmp_path / "centernet_random.pt"
    _save_random_checkpoint(str(ckpt_path))

    argv = [
        "generate_prompts.py",
        "--data-root",
        data_root,
        "--split-file",
        str(tmp_split),
        "--output-root",
        str(output_root),
        "--frontend",
        "centernet",
        "--centernet-checkpoint-path",
        str(ckpt_path),
        "--device",
        "cpu",
        "--amp",
        "0",
        "--n-peaks",
        "10",
        "--require-labels",
        "0",
        "--seed",
        "0",
    ]

    script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "generate_prompts.py")
    spec = importlib.util.spec_from_file_location("generate_prompts", script_path)
    assert spec is not None and spec.loader is not None
    generate_prompts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate_prompts)

    monkeypatch.setattr(sys, "argv", argv)
    generate_prompts.main()

    prompt_path = output_root / f"{case_id}.json"
    assert prompt_path.exists()
    pf = load_prompts_json(str(prompt_path))
    validate_prompt_file_dict(
        {
            "case_id": pf.case_id,
            "domain": pf.domain,
            "image_path": pf.image_path,
            "shape_xyz": list(pf.shape_xyz),
            "spacing_xyz_mm": list(pf.spacing_xyz_mm),
            "prompts": [
                {
                    "prompt_id": p.prompt_id,
                    "box_xyz_vox": list(p.box_xyz_vox),
                    "score": p.score,
                    **({"source": p.source} if p.source is not None else {}),
                }
                for p in pf.prompts
            ],
            **({"run": pf.run} if pf.run is not None else {}),
        }
    )
