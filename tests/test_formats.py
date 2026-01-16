import math
import os
import tempfile

import pytest

from src.promptgen.pipeline.formats import (
    Prompt,
    PromptFile,
    load_prompts_json,
    save_prompts_json,
    validate_prompt_file_dict,
)


def _minimal_dict() -> dict:
    """Build a minimal valid prompt file dict for tests.

    Inputs: none.
    Outputs: dict matching prompt schema.
    Operation: returns a hard-coded valid example.
    """
    return {
        "case_id": "CASE001",
        "domain": "FL",
        "image_path": "/path/to/image.nii.gz",
        "shape_xyz": [10, 20, 30],
        "spacing_xyz_mm": [4.0, 4.0, 4.0],
        "prompts": [
            {
                "prompt_id": "p0001",
                "box_xyz_vox": [0, 0, 0, 5, 6, 7],
                "score": 0.5,
            }
        ],
    }


def test_valid_minimal_prompt_file_passes():
    """Validate that a minimal prompt dict passes schema checks.

    Inputs: none.
    Outputs: none (asserts no exception).
    Operation: calls validate_prompt_file_dict on minimal data.
    """
    d = _minimal_dict()
    validate_prompt_file_dict(d)


def test_missing_required_field_fails():
    """Validate missing required fields raise ValueError.

    Inputs: none.
    Outputs: none (expects exception).
    Operation: removes shape_xyz and validates.
    """
    d = _minimal_dict()
    d.pop("shape_xyz")
    with pytest.raises(ValueError):
        validate_prompt_file_dict(d)


def test_invalid_box_fails():
    """Validate invalid boxes raise ValueError.

    Inputs: none.
    Outputs: none (expects exception).
    Operation: tests invalid ordering and out-of-bounds box cases.
    """
    d = _minimal_dict()
    d["prompts"][0]["box_xyz_vox"] = [0, 0, 0, 0, 6, 7]
    with pytest.raises(ValueError):
        validate_prompt_file_dict(d)

    d = _minimal_dict()
    d["prompts"][0]["box_xyz_vox"] = [0, 0, 0, 11, 6, 7]
    with pytest.raises(ValueError):
        validate_prompt_file_dict(d)


def test_nan_score_fails():
    """Validate NaN score raises ValueError.

    Inputs: none.
    Outputs: none (expects exception).
    Operation: sets score to NaN and validates.
    """
    d = _minimal_dict()
    d["prompts"][0]["score"] = float("nan")
    with pytest.raises(ValueError):
        validate_prompt_file_dict(d)


def test_save_then_load_roundtrip():
    """Validate save/load roundtrip and score clamping.

    Inputs: none.
    Outputs: none (asserts fields and ordering).
    Operation: saves PromptFile to JSON, loads it, checks equality and clamping.
    """
    pf = PromptFile(
        case_id="CASE002",
        domain="DLBCL",
        image_path="C:/data/case002.nii.gz",
        shape_xyz=(10, 20, 30),
        spacing_xyz_mm=(4.0, 4.0, 4.0),
        prompts=[
            Prompt(prompt_id="p2", box_xyz_vox=(0, 0, 0, 5, 6, 7), score=2.0),
            Prompt(prompt_id="p1", box_xyz_vox=(1, 1, 1, 3, 4, 5), score=-1.0),
        ],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "prompts.json")
        save_prompts_json(path, pf)
        loaded = load_prompts_json(path)

    assert loaded.case_id == pf.case_id
    assert loaded.domain == pf.domain
    assert loaded.image_path == pf.image_path
    assert loaded.shape_xyz == pf.shape_xyz
    assert loaded.spacing_xyz_mm == pf.spacing_xyz_mm
    assert len(loaded.prompts) == 2

    scores = [p.score for p in loaded.prompts]
    assert scores == sorted(scores, reverse=True)
    assert math.isclose(loaded.prompts[0].score, 1.0)
    assert math.isclose(loaded.prompts[1].score, 0.0)
