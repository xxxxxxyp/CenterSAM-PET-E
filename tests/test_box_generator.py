from src.promptgen.prompt.box_generator import generate_fixed_multiscale_boxes


def test_box_generator_edge_vox_and_padding():
    """Check voxel edge sizes derived from mm and padding.

    Inputs: none.
    Outputs: none (asserts box sizes).
    Operation: generates proposals for a center and verifies box extent.
    """
    centers = [((10, 10, 10), 0.9)]
    shape_xyz = (100, 100, 100)
    spacing_xyz_mm = (2.0, 2.0, 2.0)

    proposals = generate_fixed_multiscale_boxes(
        centers=centers,
        shape_xyz=shape_xyz,
        spacing_xyz_mm=spacing_xyz_mm,
        edge_mm_set=[5.0],
        padding_ratio=0.0,
    )
    assert len(proposals) == 1
    box = proposals[0].box_xyz_vox
    assert box[3] - box[0] == 3
    assert box[4] - box[1] == 3
    assert box[5] - box[2] == 3

    proposals = generate_fixed_multiscale_boxes(
        centers=centers,
        shape_xyz=shape_xyz,
        spacing_xyz_mm=spacing_xyz_mm,
        edge_mm_set=[5.0],
        padding_ratio=0.2,
    )
    box = proposals[0].box_xyz_vox
    assert box[3] - box[0] == 4
    assert box[4] - box[1] == 4
    assert box[5] - box[2] == 4
