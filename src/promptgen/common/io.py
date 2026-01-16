from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import nibabel as nib
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NiftiVolume:
    array_xyz: np.ndarray
    shape_xyz: Tuple[int, int, int]
    spacing_xyz_mm: Tuple[float, float, float]
    affine: np.ndarray
    path: str


def _extract_case_id(path: str) -> str:
    """Extract case id from a NIfTI filename.

    Inputs: path (file path).
    Outputs: case_id string.
    Operation: strips .nii.gz or file extension to obtain base name.
    """
    base = os.path.basename(path)
    if base.endswith(".nii.gz"):
        return base[:-7]
    return os.path.splitext(base)[0]


def load_nifti_xyz(path: str, dtype: np.dtype = np.float32) -> NiftiVolume:
    """Load a NIfTI file into XYZ array order.

    Inputs: path (NIfTI file), dtype (numpy dtype for array values).
    Outputs: NiftiVolume with array_xyz and metadata.
    Operation: reads via nibabel, keeps array order as stored (no transpose),
    validates 3D shape, and extracts spacing/affine.
    """
    img = nib.load(path)
    array_xyz = np.asarray(img.get_fdata(dtype=dtype))
    if array_xyz.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {array_xyz.shape} for {path}")
    shape_xyz = (int(array_xyz.shape[0]), int(array_xyz.shape[1]), int(array_xyz.shape[2]))
    zooms = img.header.get_zooms()
    spacing_xyz_mm = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    affine = np.asarray(img.affine)
    return NiftiVolume(
        array_xyz=array_xyz,
        shape_xyz=shape_xyz,
        spacing_xyz_mm=spacing_xyz_mm,
        affine=affine,
        path=path,
    )


def assert_same_shape(a_shape, b_shape, what_a: str, what_b: str) -> None:
    """Assert two shapes are identical.

    Inputs: a_shape, b_shape (iterables), what_a/what_b (labels).
    Outputs: none.
    Operation: compares tuple forms and raises ValueError on mismatch.
    """
    if tuple(a_shape) != tuple(b_shape):
        raise ValueError(
            f"Shape mismatch: {what_a}={tuple(a_shape)} vs {what_b}={tuple(b_shape)}"
        )


def load_case_volumes(
    image_path: str,
    label_path: Optional[str],
    body_mask_path: Optional[str],
) -> Tuple[NiftiVolume, Optional[NiftiVolume], Optional[NiftiVolume]]:
    """Load image/label/body_mask volumes and assert shape alignment.

    Inputs: image_path (required), label_path (optional), body_mask_path (optional).
    Outputs: tuple of (image, label, body_mask) NiftiVolume objects.
    Operation: loads volumes, checks shape consistency with image, logs metadata.
    """
    image = load_nifti_xyz(image_path, dtype=np.float32)

    label = None
    if label_path is not None:
        label = load_nifti_xyz(label_path, dtype=np.int16)
        assert_same_shape(label.shape_xyz, image.shape_xyz, "label", "image")

    body_mask = None
    if body_mask_path is not None:
        body_mask = load_nifti_xyz(body_mask_path, dtype=np.uint8)
        assert_same_shape(body_mask.shape_xyz, image.shape_xyz, "body_mask", "image")

    case_id = _extract_case_id(image_path)
    image_min = float(np.min(image.array_xyz))
    image_max = float(np.max(image.array_xyz))

    LOGGER.info(
        "case_io_summary case_id=%s image_shape_xyz=%s spacing_xyz_mm=%s image_min=%.6f image_max=%.6f has_label=%s has_body_mask=%s",
        case_id,
        image.shape_xyz,
        image.spacing_xyz_mm,
        image_min,
        image_max,
        label is not None,
        body_mask is not None,
    )

    return image, label, body_mask
