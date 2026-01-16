import math

from src.promptgen.common.metrics import compute_prompt_metrics
from src.promptgen.common.instances import LesionInstance


def _make_lesions():
    """Create two toy lesions with known centers and bboxes.

    Inputs: none.
    Outputs: list of LesionInstance.
    Operation: constructs two instances with fixed bbox/center.
    """
    return [
        LesionInstance(
            instance_id=1,
            voxels_xyz=None,  # not used in metrics
            center_xyz=(1, 1, 1),
            bbox_xyz=(0, 0, 0, 2, 2, 2),
        ),
        LesionInstance(
            instance_id=2,
            voxels_xyz=None,  # not used in metrics
            center_xyz=(4, 4, 4),
            bbox_xyz=(4, 4, 4, 6, 6, 6),
        ),
    ]


def test_metrics_toy_exact():
    """Validate recall@K and fp@K on a toy example.

    Inputs: none.
    Outputs: none (asserts metric values).
    Operation: uses 3 prompts with 2 GT to check exact metrics.
    """
    lesions = _make_lesions()
    prompts_sorted = [
        ((0, 0, 0, 2, 2, 2), 0.9),
        ((4, 4, 4, 6, 6, 6), 0.8),
        ((10, 10, 10, 12, 12, 12), 0.7),
    ]

    metrics = compute_prompt_metrics(
        case_id="CASE_Toy",
        prompts_sorted=prompts_sorted,
        lesions=lesions,
        ks=(1, 2, 3),
    )

    assert math.isclose(metrics.recall_at_k[1], 0.5)
    assert metrics.fp_at_k[1] == 0

    assert math.isclose(metrics.recall_at_k[2], 1.0)
    assert metrics.fp_at_k[2] == 0

    assert math.isclose(metrics.recall_at_k[3], 1.0)
    assert metrics.fp_at_k[3] == 1
