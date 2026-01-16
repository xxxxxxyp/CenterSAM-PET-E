import os
import tempfile

import numpy as np
import pytest
import torch

from src.promptgen.frontend.centernet.infer import (
    infer_heatmap_xyz,
    load_centernet_checkpoint,
)
from src.promptgen.frontend.centernet.model import CenterNetConfig, CenterNetPET


def _save_checkpoint(path: str, cfg: CenterNetConfig) -> None:
    model = CenterNetPET(cfg)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__,
    }
    torch.save(ckpt, path)


def test_load_checkpoint_missing_keys_fails(tmp_path):
    """Missing required keys should raise ValueError.

    Inputs: tmp_path.
    Outputs: none (expects exception).
    Operation: saves a bad checkpoint and validates error.
    """
    bad_ckpt = {"model_state_dict": {}}
    path = tmp_path / "bad.pt"
    torch.save(bad_ckpt, path)
    with pytest.raises(ValueError):
        load_centernet_checkpoint(str(path))


def test_infer_heatmap_shape_matches_input_using_dummy_weights(tmp_path):
    """Infer returns heatmap with matching shape and valid range.

    Inputs: tmp_path.
    Outputs: none (asserts shape, dtype, range, finiteness).
    Operation: saves dummy checkpoint and runs inference.
    """
    cfg = CenterNetConfig(in_channels=1, base_channels=8, num_blocks=2, use_batchnorm=False)
    ckpt_path = tmp_path / "dummy.pt"
    _save_checkpoint(str(ckpt_path), cfg)

    volume = np.random.rand(16, 17, 18).astype(np.float32)
    heatmap = infer_heatmap_xyz(volume_xyz=volume, checkpoint_path=str(ckpt_path), device="cpu", amp=False)

    assert heatmap.shape == volume.shape
    assert heatmap.dtype == np.float32
    assert np.isfinite(heatmap).all()
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
