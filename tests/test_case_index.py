import os

import pytest

from src.promptgen.data.case_index import (
    build_case_records,
    infer_domain,
    load_split_case_ids,
    parse_case_id,
)


def _write_split(path: str, lines: list[str]) -> None:
    """Write split file lines for tests.

    Inputs: path (file path), lines (list of strings).
    Outputs: none.
    Operation: writes each line with newline terminator.
    """
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def _touch(path: str) -> None:
    """Create an empty file at the given path.

    Inputs: path (file path).
    Outputs: none.
    Operation: creates parent dirs and writes empty bytes.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"")


def test_parse_case_id_valid():
    """Ensure valid case_id parses correctly.

    Inputs: none.
    Outputs: none (asserts return value).
    Operation: calls parse_case_id on valid input.
    """
    assert parse_case_id("0001") == "0001"


def test_parse_case_id_invalid():
    """Ensure invalid case_id raises ValueError.

    Inputs: none.
    Outputs: none (expects exception).
    Operation: tries invalid lengths and non-digit ids.
    """
    with pytest.raises(ValueError):
        parse_case_id("01")
    with pytest.raises(ValueError):
        parse_case_id("abcd")


def test_infer_domain():
    """Ensure domain inference works and invalid prefix errors.

    Inputs: none.
    Outputs: none (asserts or expects exception).
    Operation: checks FL/DLBCL mapping and unknown prefix handling.
    """
    assert infer_domain("0001") == "FL"
    assert infer_domain("1001") == "DLBCL"
    with pytest.raises(ValueError):
        infer_domain("2001")


def test_load_split_case_ids(tmp_path):
    """Ensure split loader skips comments/empty lines.

    Inputs: tmp_path fixture.
    Outputs: none (asserts list equality).
    Operation: writes a split file and loads case ids.
    """
    split_path = tmp_path / "train.txt"
    _write_split(str(split_path), ["# comment", "", "0001", "0002"])
    assert load_split_case_ids(str(split_path)) == ["0001", "0002"]


def test_build_case_records_parsing_ok(tmp_path):
    """Ensure build_case_records returns valid records when files exist.

    Inputs: tmp_path fixture.
    Outputs: none (asserts record count and domains).
    Operation: creates fake files and validates returned records.
    """
    data_root = tmp_path / "processed"
    split_path = tmp_path / "train.txt"
    _write_split(str(split_path), ["0001", "1002"])

    for case_id in ["0001", "1002"]:
        _touch(str(data_root / "images" / f"{case_id}.nii.gz"))
        _touch(str(data_root / "labels" / f"{case_id}.nii.gz"))
        _touch(str(data_root / "body_masks" / f"{case_id}.nii.gz"))

    records = build_case_records(
        data_root=str(data_root),
        split_file=str(split_path),
        require_labels=True,
        allow_missing_body_mask=True,
    )

    assert len(records) == 2
    assert records[0].case_id == "0001"
    assert records[0].domain == "FL"
    assert records[1].case_id == "1002"
    assert records[1].domain == "DLBCL"
