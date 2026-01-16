import math

from src.promptgen.common.geometry import box_iou_xyz, center_in_box_xyz


def test_iou_known_boxes():
    """Verify IoU for two overlapping cubes with known intersection.

    Inputs: none.
    Outputs: none (asserts IoU value).
    Operation: computes IoU for two boxes with intersection volume 1 and union 15.
    """
    a = (0, 0, 0, 2, 2, 2)
    b = (1, 1, 1, 3, 3, 3)
    iou = box_iou_xyz(a, b)
    assert math.isclose(iou, 1.0 / 15.0, rel_tol=1e-9)


def test_center_in_box():
    """Verify half-open inclusion logic for center points.

    Inputs: none.
    Outputs: none (asserts booleans).
    Operation: checks inside and boundary-excluded points.
    """
    box = (0, 0, 0, 2, 2, 2)
    assert center_in_box_xyz((1, 1, 1), box) is True
    assert center_in_box_xyz((2, 2, 2), box) is False
