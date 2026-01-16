from src.promptgen.prompt.box_generator import Proposal
from src.promptgen.prompt.diversify import diversify_by_grid_cell


def _make_proposal(score: float, center_xyz_vox=(0, 0, 0)) -> Proposal:
    """Create a minimal Proposal for diversify tests.

    Inputs: score (float), center_xyz_vox (tuple).
    Outputs: Proposal.
    Operation: constructs a proposal with fixed geometry.
    """
    return Proposal(
        box_xyz_vox=(0, 0, 0, 1, 1, 1),
        center_xyz_vox=center_xyz_vox,
        heatmap_score=score,
        generator="test",
        generator_params={},
    )


def test_diversify_caps_per_cell():
    """Ensure proposals in same cell are capped by max_per_cell.

    Inputs: none.
    Outputs: none (asserts length and ordering).
    Operation: keeps top scored proposals within a single cell.
    """
    proposals_scored = [
        (_make_proposal(0.9, (1, 1, 1)), 0.9),
        (_make_proposal(0.8, (2, 2, 2)), 0.8),
        (_make_proposal(0.7, (3, 3, 3)), 0.7),
    ]
    kept = diversify_by_grid_cell(
        proposals_scored=proposals_scored,
        spacing_xyz_mm=(1.0, 1.0, 1.0),
        eps_mm=10.0,
        max_per_cell=2,
    )
    assert len(kept) == 2
    assert kept[0][1] == 0.9
    assert kept[1][1] == 0.8


def test_diversify_deterministic_stable():
    """Ensure deterministic ordering is stable for equal scores.

    Inputs: none.
    Outputs: none (asserts ordering).
    Operation: verifies tie-break keeps original order.
    """
    proposals_scored = [
        (_make_proposal(0.5, (0, 0, 0)), 0.5),
        (_make_proposal(0.5, (5, 5, 5)), 0.5),
        (_make_proposal(0.6, (10, 10, 10)), 0.6),
    ]
    kept = diversify_by_grid_cell(
        proposals_scored=proposals_scored,
        spacing_xyz_mm=(1.0, 1.0, 1.0),
        eps_mm=3.0,
        max_per_cell=5,
    )
    assert [p.center_xyz_vox for p, _ in kept] == [(10, 10, 10), (0, 0, 0), (5, 5, 5)]
