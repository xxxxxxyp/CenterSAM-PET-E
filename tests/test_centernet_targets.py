import numpy as np

from src.promptgen.frontend.centernet.targets import build_heatmap_target_xyz


def test_centernet_targets_single_component_peak():
    """Toy label -> heatmap peak at center is 1.0 and within bounds.

    Inputs: none.
    Outputs: none (asserts peak value and location).
    Operation: builds target heatmap and checks center.
    """
    label = np.zeros((5, 5, 5), dtype=np.uint8)
    label[2, 2, 2] = 1

    heatmap = build_heatmap_target_xyz(label_xyz=label, shape_xyz=label.shape, sigma_vox=1.0)

    assert heatmap.shape == label.shape
    assert heatmap[2, 2, 2] >= 0.99
    assert heatmap.max() <= 1.0
    assert heatmap.min() >= 0.0
