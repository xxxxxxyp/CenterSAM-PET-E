import os
import sys
from typing import List, Tuple
import importlib.util

import pytest

from src.promptgen.data.case_index import load_split_case_ids
from src.promptgen.pipeline.formats import load_prompts_json


def _resolve_split_file() -> str:
    """Resolve split file path for regression test.

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


def _run_generate_prompts(tmp_path, case_id: str, data_root: str, seed: int) -> str:
    """Run generate_prompts for a single case id.

    Inputs: tmp_path, case_id, data_root, seed.
    Outputs: prompt json path.
    Operation: writes temp split file and runs CLI module.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    tmp_split = tmp_path / f"split_{case_id}.txt"
    tmp_split.write_text(f"{case_id}\n", encoding="utf-8")

    output_root = tmp_path / f"prompts_{seed}"

    argv = [
        "generate_prompts.py",
        "--data-root",
        data_root,
        "--split-file",
        str(tmp_split),
        "--output-root",
        str(output_root),
        "--frontend",
        "dummy_random",
        "--require-labels",
        "0",
        "--num-random-centers",
        "10",
        "--seed",
        str(seed),
    ]

    script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "generate_prompts.py")
    spec = importlib.util.spec_from_file_location("generate_prompts", script_path)
    assert spec is not None and spec.loader is not None
    generate_prompts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate_prompts)

    sys.argv = argv
    generate_prompts.main()

    prompt_path = output_root / f"{case_id}.json"
    assert prompt_path.exists()
    return str(prompt_path)


def _topk_summary(prompt_path: str, k: int = 3) -> Tuple[int, List[Tuple[Tuple[int, int, int, int, int, int], float]]]:
    """Load prompts and return K_out and top-k (box, score).

    Inputs: prompt_path (str), k (int).
    Outputs: K_out and list of top-k tuples.
    Operation: sorts by score desc, prompt_id asc (same as exporter).
    """
    pf = load_prompts_json(prompt_path)
    prompts_sorted = sorted(pf.prompts, key=lambda p: (-p.score, p.prompt_id))
    topk = [(p.box_xyz_vox, float(p.score)) for p in prompts_sorted[:k]]
    return len(pf.prompts), topk


def test_regression_single_case_stable(tmp_path):
    """Ensure dummy_random outputs stable prompts for a fixed case.

    Inputs: tmp_path.
    Outputs: none (asserts stability).
    Operation: runs twice with same seed and compares K_out and top-3.
    """
    data_root = os.environ.get("PROMPTGEN_DATA_ROOT", os.path.join("data", "processed"))
    split_file = _resolve_split_file()
    if not os.path.exists(split_file):
        pytest.skip("split file not found for regression test")

    case_ids = load_split_case_ids(split_file)
    case_ids = _filter_existing_cases(case_ids, data_root)
    if not case_ids:
        pytest.skip("no cases with existing images for regression test")

    case_id = case_ids[0]

    path_a = _run_generate_prompts(tmp_path / "run_a", case_id, data_root, seed=0)
    path_b = _run_generate_prompts(tmp_path / "run_b", case_id, data_root, seed=0)

    k_out_a, top3_a = _topk_summary(path_a, k=3)
    k_out_b, top3_b = _topk_summary(path_b, k=3)

    assert k_out_a == k_out_b
    assert top3_a == top3_b
