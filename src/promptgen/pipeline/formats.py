from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

BoxXYZ = Tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    box_xyz_vox: BoxXYZ
    score: float
    source: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class PromptFile:
    case_id: str
    domain: str
    image_path: str
    shape_xyz: Tuple[int, int, int]
    spacing_xyz_mm: Tuple[float, float, float]
    prompts: List[Prompt]
    run: Optional[Dict[str, Any]] = None


def clamp_score_01(score: float) -> float:
    """Clamp score into [0, 1] after validating finiteness.

    Inputs: score (float).
    Outputs: clamped score in [0, 1].
    Operation: raises if not finite; otherwise clamps to bounds.
    """
    if not math.isfinite(score):
        raise ValueError("score must be finite")
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _ensure_tuple_len(name: str, v: Any, n: int) -> Tuple[Any, ...]:
    """Coerce list/tuple to tuple and validate length.

    Inputs: name (field name), v (list/tuple), n (expected length).
    Outputs: tuple of length n.
    Operation: type/length checks; raises on mismatch.
    """
    if not isinstance(v, (list, tuple)):
        raise ValueError(f"{name} must be a list/tuple")
    if len(v) != n:
        raise ValueError(f"{name} must have length {n}")
    return tuple(v)


def validate_box_xyz_vox(box: BoxXYZ, shape_xyz: Tuple[int, int, int]) -> None:
    """Validate a half-open [x0,x1)×[y0,y1)×[z0,z1) box within shape.

    Inputs: box (6 ints), shape_xyz (X,Y,Z).
    Outputs: none.
    Operation: checks integer types, non-negative start, proper ordering,
    and bounds within shape; raises ValueError on error.
    """
    if len(box) != 6:
        raise ValueError("box_xyz_vox must have length 6")
    x0, y0, z0, x1, y1, z1 = box
    for name, v in zip(("x0", "y0", "z0", "x1", "y1", "z1"), box):
        if not isinstance(v, int):
            raise ValueError(f"box_xyz_vox {name} must be int")
    if x0 < 0 or y0 < 0 or z0 < 0:
        raise ValueError("box_xyz_vox start must be >= 0")
    if x1 <= x0 or y1 <= y0 or z1 <= z0:
        raise ValueError("box_xyz_vox must be a valid half-open interval")
    sx, sy, sz = shape_xyz
    if x1 > sx or y1 > sy or z1 > sz:
        raise ValueError("box_xyz_vox out of bounds")


def _validate_domain(domain: Any) -> str:
    """Validate domain against allowed values.

    Inputs: domain (any).
    Outputs: domain string if valid.
    Operation: type check + membership in {"FL","DLBCL"}; raises on error.
    """
    if not isinstance(domain, str):
        raise ValueError("domain must be str")
    if domain not in {"FL", "DLBCL"}:
        raise ValueError("domain must be 'FL' or 'DLBCL'")
    return domain


def _validate_shape_xyz(shape_xyz: Any) -> Tuple[int, int, int]:
    """Validate shape_xyz is length-3 positive ints.

    Inputs: shape_xyz (list/tuple).
    Outputs: tuple of 3 ints.
    Operation: length/type checks and >0 validation; raises on error.
    """
    shape = _ensure_tuple_len("shape_xyz", shape_xyz, 3)
    if not all(isinstance(v, int) for v in shape):
        raise ValueError("shape_xyz must be ints")
    if not all(v > 0 for v in shape):
        raise ValueError("shape_xyz values must be > 0")
    return shape  # type: ignore[return-value]


def _validate_spacing_xyz(spacing_xyz_mm: Any) -> Tuple[float, float, float]:
    """Validate spacing_xyz_mm is length-3 positive floats.

    Inputs: spacing_xyz_mm (list/tuple of numbers).
    Outputs: tuple of 3 floats.
    Operation: converts to float, checks positivity; raises on error.
    """
    spacing = _ensure_tuple_len("spacing_xyz_mm", spacing_xyz_mm, 3)
    try:
        spacing_f = tuple(float(v) for v in spacing)
    except (TypeError, ValueError) as exc:
        raise ValueError("spacing_xyz_mm must be numbers") from exc
    if not all(v > 0 for v in spacing_f):
        raise ValueError("spacing_xyz_mm values must be > 0")
    return spacing_f  # type: ignore[return-value]


def _validate_prompt_dict(p: Dict[str, Any], shape_xyz: Tuple[int, int, int]) -> Prompt:
    """Validate a single prompt dict and convert to Prompt.

    Inputs: p (dict), shape_xyz (for box bounds).
    Outputs: Prompt dataclass.
    Operation: schema checks, box/score validation, optional source typing.
    """
    if not isinstance(p, dict):
        raise ValueError("prompt must be an object")
    if "prompt_id" not in p:
        raise ValueError("prompt_id missing")
    if "box_xyz_vox" not in p:
        raise ValueError("box_xyz_vox missing")
    if "score" not in p:
        raise ValueError("score missing")
    prompt_id = p["prompt_id"]
    if not isinstance(prompt_id, str) or not prompt_id:
        raise ValueError("prompt_id must be non-empty str")
    box = _ensure_tuple_len("box_xyz_vox", p["box_xyz_vox"], 6)
    try:
        box_i = tuple(int(v) for v in box)
    except (TypeError, ValueError) as exc:
        raise ValueError("box_xyz_vox must be ints") from exc
    validate_box_xyz_vox(box_i, shape_xyz)
    score = p["score"]
    if not isinstance(score, (int, float)):
        raise ValueError("score must be number")
    if not math.isfinite(float(score)):
        raise ValueError("score must be finite")
    source = p.get("source")
    if source is not None and not isinstance(source, dict):
        raise ValueError("source must be dict if provided")
    return Prompt(prompt_id=prompt_id, box_xyz_vox=box_i, score=float(score), source=source)


def validate_prompt_file_dict(d: Dict[str, Any]) -> None:
    """Validate minimal prompt-file schema + invariants.

    Inputs: d (dict parsed from JSON).
    Outputs: none.
    Operation: required fields, types, and per-prompt validation; raises on error.
    """
    if not isinstance(d, dict):
        raise ValueError("prompt file must be a dict")
    for key in ("case_id", "domain", "image_path", "shape_xyz", "spacing_xyz_mm", "prompts"):
        if key not in d:
            raise ValueError(f"missing required field: {key}")
    if not isinstance(d["case_id"], str) or not d["case_id"]:
        raise ValueError("case_id must be non-empty str")
    _validate_domain(d["domain"])
    if not isinstance(d["image_path"], str) or not d["image_path"]:
        raise ValueError("image_path must be non-empty str")
    shape_xyz = _validate_shape_xyz(d["shape_xyz"])
    _validate_spacing_xyz(d["spacing_xyz_mm"])
    prompts = d["prompts"]
    if not isinstance(prompts, list):
        raise ValueError("prompts must be a list")
    for p in prompts:
        _validate_prompt_dict(p, shape_xyz)
    run = d.get("run")
    if run is not None and not isinstance(run, dict):
        raise ValueError("run must be dict if provided")


def _promptfile_from_dict(d: Dict[str, Any]) -> PromptFile:
    """Convert a validated dict into a PromptFile dataclass.

    Inputs: d (validated prompt-file dict).
    Outputs: PromptFile.
    Operation: re-validates key fields and constructs dataclasses.
    """
    shape_xyz = _validate_shape_xyz(d["shape_xyz"])
    spacing_xyz_mm = _validate_spacing_xyz(d["spacing_xyz_mm"])
    prompts = [_validate_prompt_dict(p, shape_xyz) for p in d["prompts"]]
    return PromptFile(
        case_id=d["case_id"],
        domain=_validate_domain(d["domain"]),
        image_path=d["image_path"],
        shape_xyz=shape_xyz,
        spacing_xyz_mm=spacing_xyz_mm,
        prompts=prompts,
        run=d.get("run"),
    )


def load_prompts_json(path: str) -> PromptFile:
    """Read JSON file, validate, return PromptFile.

    Inputs: path (json file path).
    Outputs: PromptFile.
    Operation: loads JSON, validates schema, converts to dataclass.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    validate_prompt_file_dict(data)
    return _promptfile_from_dict(data)


def _promptfile_to_dict(pf: PromptFile) -> Dict[str, Any]:
    """Convert PromptFile to a JSON-serializable dict.

    Inputs: pf (PromptFile).
    Outputs: dict suitable for JSON.
    Operation: sorts prompts, clamps scores, and serializes fields.
    """
    prompts_sorted = sorted(pf.prompts, key=lambda p: (-p.score, p.prompt_id))
    prompts_dicts: List[Dict[str, Any]] = []
    for p in prompts_sorted:
        score = clamp_score_01(float(p.score))
        prompts_dicts.append(
            {
                "prompt_id": p.prompt_id,
                "box_xyz_vox": list(p.box_xyz_vox),
                "score": score,
                **({"source": p.source} if p.source is not None else {}),
            }
        )
    d: Dict[str, Any] = {
        "case_id": pf.case_id,
        "domain": pf.domain,
        "image_path": pf.image_path,
        "shape_xyz": list(pf.shape_xyz),
        "spacing_xyz_mm": list(pf.spacing_xyz_mm),
        "prompts": prompts_dicts,
    }
    if pf.run is not None:
        d["run"] = pf.run
    return d


def save_prompts_json(path: str, pf: PromptFile) -> None:
    """Write prompts JSON atomically and validate round-trip.

    Inputs: path (target path), pf (PromptFile).
    Outputs: none.
    Operation: serialize with sorting/clamping, write to temp, replace target,
    and reload to validate; raises on validation failures.
    """
    d = _promptfile_to_dict(pf)
    validate_prompt_file_dict(d)

    dir_path = os.path.dirname(os.path.abspath(path))
    os.makedirs(dir_path, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="prompts_", suffix=".json", dir=dir_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    _ = load_prompts_json(path)
