import numpy as np

from src.promptgen.classifier.dataset import _extract_patch_centered


def test_classifier_patch_extract_padding_and_shape():
    """Patch extraction should pad correctly and be deterministic.

    Inputs: none.
    Outputs: none (asserts shape and values).
    Operation: extracts patch near boundary and checks padding.
    """
    volume = np.zeros((5, 5, 5), dtype=np.float32)
    volume[0, 0, 0] = 1.0

    patch = _extract_patch_centered(volume, center_xyz=(0, 0, 0), patch_shape_xyz=(4, 4, 4))

    assert patch.shape == (4, 4, 4)
    assert patch[2, 2, 2] == 1.0
    assert patch[0, 0, 0] == 0.0

    patch2 = _extract_patch_centered(volume, center_xyz=(0, 0, 0), patch_shape_xyz=(4, 4, 4))
    assert np.array_equal(patch, patch2)
