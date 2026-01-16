from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ...data.case_index import CaseRecord
from ...common.io import load_nifti_xyz
from .targets import build_heatmap_target_xyz


@dataclass(frozen=True)
class CenterNetTrainConfig:
    sigma_vox: float = 2.0
    intensity_clip: Optional[Tuple[float, float]] = None
    normalize: str = "zscore"  # or "minmax" or "none"


class CenterNetDataset(Dataset):
    def __init__(self, cases: List[CaseRecord], cfg: CenterNetTrainConfig, require_labels: bool = True):
        self.cases = cases
        self.cfg = cfg
        self.require_labels = require_labels
        if cfg.sigma_vox <= 0:
            raise ValueError("sigma_vox must be positive")
        if cfg.normalize not in {"zscore", "minmax", "none"}:
            raise ValueError("normalize must be one of: zscore|minmax|none")

    def __len__(self) -> int:
        return len(self.cases)

    def _normalize(self, image_xyz: np.ndarray) -> np.ndarray:
        if self.cfg.intensity_clip is not None:
            lo, hi = self.cfg.intensity_clip
            image_xyz = np.clip(image_xyz, float(lo), float(hi))

        if self.cfg.normalize == "none":
            return image_xyz.astype(np.float32, copy=False)
        if self.cfg.normalize == "minmax":
            vmin = float(np.min(image_xyz))
            vmax = float(np.max(image_xyz))
            if vmax <= vmin:
                return np.zeros_like(image_xyz, dtype=np.float32)
            return ((image_xyz - vmin) / (vmax - vmin)).astype(np.float32, copy=False)

        mean = float(np.mean(image_xyz))
        std = float(np.std(image_xyz))
        if std <= 0:
            return np.zeros_like(image_xyz, dtype=np.float32)
        return ((image_xyz - mean) / std).astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case = self.cases[idx]
        if self.require_labels and not case.label_path:
            raise ValueError(f"case {case.case_id} requires label_path")

        image_vol = load_nifti_xyz(case.image_path, dtype=np.float32)
        image_xyz = self._normalize(image_vol.array_xyz)

        if case.label_path is not None:
            label_vol = load_nifti_xyz(case.label_path, dtype=np.int16)
            target = build_heatmap_target_xyz(
                label_xyz=label_vol.array_xyz,
                shape_xyz=image_vol.shape_xyz,
                sigma_vox=self.cfg.sigma_vox,
            )
        else:
            target = np.zeros(image_vol.shape_xyz, dtype=np.float32)

        image_tensor = torch.from_numpy(image_xyz).unsqueeze(0)
        target_tensor = torch.from_numpy(target).unsqueeze(0)

        return {
            "image": image_tensor,
            "target_heatmap": target_tensor,
            "case_id": case.case_id,
        }
