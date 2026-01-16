import numpy as np

from src.promptgen.frontend.peaks import extract_topk_peaks_3d_xyz


def test_extract_topk_peaks_toy():
    """Toy heatmap with known maxima returns expected centers.

    Inputs: none.
    Outputs: none (asserts centers and ordering).
    Operation: places a few peaks and checks extracted top-k.
    """
    heatmap = np.zeros((5, 5, 5), dtype=np.float32)
    heatmap[1, 1, 1] = 0.9
    heatmap[3, 3, 3] = 0.8
    heatmap[2, 2, 2] = 0.7

    peaks = extract_topk_peaks_3d_xyz(
        heatmap_xyz=heatmap,
        gating_mask_xyz=None,
        topk=2,
        neighborhood=1,
    )

    centers = [c for c, _ in peaks]
    assert centers == [(1, 1, 1), (3, 3, 3)]
