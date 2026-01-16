import numpy as np

from src.promptgen.common.instances import extract_lesion_instances_xyz


def _make_label() -> np.ndarray:
    """Create a toy label with two 26-connected components.

    Inputs: none.
    Outputs: 3D numpy array with foreground labels >0.
    Operation: places two voxels diagonally connected plus one isolated voxel.
    """
    label = np.zeros((5, 5, 5), dtype=np.uint8)
    label[0, 0, 0] = 1
    label[1, 1, 1] = 1  # 26-connected to (0,0,0)
    label[3, 3, 3] = 1  # separate component
    return label


def test_extract_instances_26_connectivity_and_bbox_center():
    """Validate 26-connectivity, half-open bbox, and rint center.

    Inputs: none.
    Outputs: none (asserts instance count and attributes).
    Operation: extracts instances and checks bbox/center logic.
    """
    label = _make_label()
    instances = extract_lesion_instances_xyz(label)
    assert len(instances) == 2

    by_bbox = {inst.bbox_xyz: inst for inst in instances}

    bbox_a = (0, 0, 0, 2, 2, 2)
    bbox_b = (3, 3, 3, 4, 4, 4)

    assert bbox_a in by_bbox
    assert bbox_b in by_bbox

    inst_a = by_bbox[bbox_a]
    inst_b = by_bbox[bbox_b]

    assert inst_a.center_xyz == (0, 0, 0)
    assert inst_b.center_xyz == (3, 3, 3)
