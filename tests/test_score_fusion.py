from src.promptgen.pipeline.run_pipeline import _fuse_scores


def test_score_fusion_clamp():
    """Fixed inputs produce expected fused score with clamp.

    Inputs: none.
    Outputs: none (asserts values).
    Operation: fuses scores and verifies clamping to [0,1].
    """
    hm_scores = [0.2, -1.0]
    cls_scores = [0.8, 2.0]
    fused = _fuse_scores(hm_scores=hm_scores, cls_scores=cls_scores, w_cls=0.7)

    assert fused[0] == 0.62
    assert fused[1] == 1.0
