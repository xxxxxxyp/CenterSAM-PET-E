import os

import pytest

from src.promptgen.data.case_index import build_case_records


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


def test_missing_image_raises(tmp_path):
    """Ensure missing image triggers ValueError with missing list.

    Inputs: tmp_path fixture.
    Outputs: none (expects exception and message checks).
    Operation: omits one image and asserts error message.
    """
    data_root = tmp_path / "processed"
    split_path = tmp_path / "train.txt"
    _write_split(str(split_path), ["0001", "0002"])

    _touch(str(data_root / "images" / "0001.nii.gz"))
    _touch(str(data_root / "labels" / "0001.nii.gz"))
    _touch(str(data_root / "body_masks" / "0001.nii.gz"))

    _touch(str(data_root / "labels" / "0002.nii.gz"))
    _touch(str(data_root / "body_masks" / "0002.nii.gz"))

    with pytest.raises(ValueError) as excinfo:
        build_case_records(
            data_root=str(data_root),
            split_file=str(split_path),
            require_labels=True,
            allow_missing_body_mask=True,
        )

    msg = str(excinfo.value)
    assert "missing files" in msg
    assert "images=['0002']" in msg
