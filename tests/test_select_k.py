from src.promptgen.prompt.box_generator import Proposal
from src.promptgen.prompt.select_k import select_prompts_recall_first


def _make_proposal(score: float) -> Proposal:
    """Create a minimal Proposal for selection tests.

    Inputs: score (float).
    Outputs: Proposal.
    Operation: constructs a proposal with fixed geometry.
    """
    return Proposal(
        box_xyz_vox=(0, 0, 0, 1, 1, 1),
        center_xyz_vox=(0, 0, 0),
        heatmap_score=score,
        generator="test",
        generator_params={},
    )


def test_select_k_less_than_kmin():
    """Ensure selection returns all proposals when fewer than K_min.

    Inputs: none.
    Outputs: none (asserts length).
    Operation: checks that available proposals are returned.
    """
    proposals = [(_make_proposal(0.1), 0.1)]
    selected = select_prompts_recall_first(proposals, K_min=3, t_min=0.5, K_cap_soft=10)
    assert len(selected) == 1


def test_select_k_cap_soft():
    """Ensure soft cap truncates selected list.

    Inputs: none.
    Outputs: none (asserts length).
    Operation: creates many proposals and applies K_cap_soft.
    """
    proposals = [(_make_proposal(1.0 - i * 0.01), 1.0 - i * 0.01) for i in range(10)]
    selected = select_prompts_recall_first(proposals, K_min=1, t_min=0.0, K_cap_soft=5)
    assert len(selected) == 5


def test_select_k_all_below_threshold():
    """Ensure K_min is satisfied when all scores below threshold.

    Inputs: none.
    Outputs: none (asserts length and ordering).
    Operation: selects top-K_min despite t_min filter.
    """
    proposals = [(_make_proposal(0.01 * i), 0.01 * i) for i in range(5)]
    selected = select_prompts_recall_first(proposals, K_min=3, t_min=0.5, K_cap_soft=10)
    assert len(selected) == 3
    assert selected[0][1] >= selected[1][1] >= selected[2][1]
