from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ..common.geometry import box_iou_xyz, center_in_box_xyz
from ..common.instances import extract_lesion_instances_xyz
from ..common.io import load_case_volumes
from ..data.case_index import CaseRecord
from ..frontend.peaks import extract_topk_peaks_3d_xyz
from ..frontend.centernet.infer import infer_heatmap_xyz
from ..prompt.box_generator import generate_fixed_multiscale_boxes, Proposal


@dataclass(frozen=True)
class ProposalDatasetConfig:
    patch_shape_xyz: Tuple[int, int, int] = (64, 64, 64)
    n_peaks: int = 1000
    edge_mm_set: Tuple[float, ...] = (12.0, 20.0, 32.0, 48.0)
    padding_ratio: float = 0.2
    iou_threshold: float = 0.1
    neg_pos_ratio: int = 3
    seed: int = 0
    hard_neg_path: Optional[str] = None
    hard_neg_fraction: float = 0.5


def _extract_patch_centered(
    volume_xyz: np.ndarray,
    center_xyz: Tuple[int, int, int],
    patch_shape_xyz: Tuple[int, int, int],
) -> np.ndarray:
    px, py, pz = patch_shape_xyz
    sx, sy, sz = volume_xyz.shape

    cx, cy, cz = center_xyz
    hx, hy, hz = px // 2, py // 2, pz // 2

    x0 = cx - hx
    y0 = cy - hy
    z0 = cz - hz
    x1 = x0 + px
    y1 = y0 + py
    z1 = z0 + pz

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_z0 = max(0, z0)
    src_x1 = min(sx, x1)
    src_y1 = min(sy, y1)
    src_z1 = min(sz, z1)

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_z0 = src_z0 - z0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_z1 = dst_z0 + (src_z1 - src_z0)

    patch = np.zeros((px, py, pz), dtype=np.float32)
    patch[dst_x0:dst_x1, dst_y0:dst_y1, dst_z0:dst_z1] = volume_xyz[
        src_x0:src_x1, src_y0:src_y1, src_z0:src_z1
    ]
    return patch


def _proposal_is_positive(proposal: Proposal, lesions, iou_threshold: float) -> bool:
    for lesion in lesions:
        if center_in_box_xyz(lesion.center_xyz, proposal.box_xyz_vox):
            return True
        if box_iou_xyz(proposal.box_xyz_vox, lesion.bbox_xyz) >= iou_threshold:
            return True
    return False


class ProposalClassifierDataset(Dataset):
    def __init__(
        self,
        cases: List[CaseRecord],
        centernet_checkpoint_path: str,
        device: str,
        cfg: ProposalDatasetConfig,
        require_labels: bool = True,
    ):
        self.cases = cases
        self.cfg = cfg
        self.require_labels = require_labels
        self.centernet_checkpoint_path = centernet_checkpoint_path
        self.device = device

        if cfg.neg_pos_ratio < 0:
            raise ValueError("neg_pos_ratio must be non-negative")
        if cfg.n_peaks <= 0:
            raise ValueError("n_peaks must be positive")

        if cfg.hard_neg_fraction < 0.0 or cfg.hard_neg_fraction > 1.0:
            raise ValueError("hard_neg_fraction must be in [0,1]")

        self._rng = np.random.default_rng(cfg.seed)
        self._case_by_id = {c.case_id: c for c in cases}

        self._pos: List[Tuple[CaseRecord, Proposal, float]] = []
        self._neg: List[Tuple[CaseRecord, Proposal, float]] = []
        self._hard_neg: List[Tuple[CaseRecord, Proposal, float]] = []
        self.samples: List[Tuple[CaseRecord, Proposal, float]] = []

        for case in cases:
            image, label, body_mask = load_case_volumes(
                image_path=case.image_path,
                label_path=case.label_path,
                body_mask_path=case.body_mask_path,
            )
            if self.require_labels and label is None:
                raise ValueError(f"case {case.case_id} requires label_path")

            gating = body_mask.array_xyz.astype(bool) if body_mask is not None else None
            heatmap_xyz = infer_heatmap_xyz(
                volume_xyz=image.array_xyz,
                checkpoint_path=centernet_checkpoint_path,
                device=device,
                amp=False,
            )
            centers = extract_topk_peaks_3d_xyz(
                heatmap_xyz=heatmap_xyz,
                gating_mask_xyz=gating,
                topk=cfg.n_peaks,
                neighborhood=1,
            )

            proposals = generate_fixed_multiscale_boxes(
                centers=centers,
                shape_xyz=image.shape_xyz,
                spacing_xyz_mm=image.spacing_xyz_mm,
                edge_mm_set=list(cfg.edge_mm_set),
                padding_ratio=cfg.padding_ratio,
            )

            lesions = extract_lesion_instances_xyz(label.array_xyz) if label is not None else []

            for proposal in proposals:
                is_pos = _proposal_is_positive(proposal, lesions, cfg.iou_threshold)
                sample = (case, proposal, 1.0 if is_pos else 0.0)
                if is_pos:
                    self._pos.append(sample)
                else:
                    self._neg.append(sample)

        if cfg.hard_neg_path:
            self._load_hard_negatives(cfg.hard_neg_path)

        self._refresh_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case, proposal, y = self.samples[idx]
        image_vol = load_case_volumes(
            image_path=case.image_path,
            label_path=None,
            body_mask_path=None,
        )[0]

        patch = _extract_patch_centered(
            volume_xyz=image_vol.array_xyz.astype(np.float32, copy=False),
            center_xyz=proposal.center_xyz_vox,
            patch_shape_xyz=self.cfg.patch_shape_xyz,
        )

        patch_tensor = torch.from_numpy(patch).unsqueeze(0)
        y_tensor = torch.tensor([float(y)], dtype=torch.float32)

        return {
            "patch": patch_tensor,
            "y": y_tensor,
        }

    def set_epoch(self, epoch: int) -> None:
        self._refresh_samples()

    def _load_hard_negatives(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"hard_neg_path not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                case_id = row.get("case_id")
                box = row.get("box_xyz_vox")
                score = row.get("heatmap_score", 0.0)
                if case_id not in self._case_by_id:
                    continue
                if not isinstance(box, (list, tuple)) or len(box) != 6:
                    continue
                box_xyz = tuple(int(v) for v in box)
                x0, y0, z0, x1, y1, z1 = box_xyz
                cx = (x0 + x1 - 1) // 2
                cy = (y0 + y1 - 1) // 2
                cz = (z0 + z1 - 1) // 2
                proposal = Proposal(
                    box_xyz_vox=box_xyz,
                    center_xyz_vox=(int(cx), int(cy), int(cz)),
                    heatmap_score=float(score),
                    generator="hard_negative",
                    generator_params={"source": "hard_negatives_jsonl"},
                )
                self._hard_neg.append((self._case_by_id[case_id], proposal, 0.0))

    def _refresh_samples(self) -> None:
        pos = self._pos
        neg = self._neg
        hard_neg = self._hard_neg

        pos_count = max(1, len(pos))
        max_neg = self.cfg.neg_pos_ratio * pos_count if self.cfg.neg_pos_ratio > 0 else len(neg)

        hard_count = int(round(max_neg * float(self.cfg.hard_neg_fraction)))
        reg_count = max_neg - hard_count

        selected_neg: List[Tuple[CaseRecord, Proposal, float]] = []

        if hard_neg and hard_count > 0:
            if len(hard_neg) <= hard_count:
                selected_neg.extend(hard_neg)
            else:
                idx = self._rng.choice(len(hard_neg), size=hard_count, replace=False)
                selected_neg.extend([hard_neg[i] for i in idx])

        if reg_count > 0 and neg:
            if len(neg) <= reg_count:
                selected_neg.extend(neg)
            else:
                idx = self._rng.choice(len(neg), size=reg_count, replace=False)
                selected_neg.extend([neg[i] for i in idx])

        if not selected_neg and neg:
            selected_neg = neg

        self.samples = list(pos) + selected_neg
