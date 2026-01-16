from __future__ import annotations

import argparse
import logging
import os
import random
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.promptgen.classifier.dataset import ProposalClassifierDataset, ProposalDatasetConfig
from src.promptgen.classifier.train import ClassifierConfig, ProposalClassifier, train_classifier
from src.promptgen.data.case_index import build_case_records

LOGGER = logging.getLogger(__name__)


def _parse_patch_shape(patch_shape: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in patch_shape.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("patch-shape must be three comma-separated ints")
    vals = tuple(int(v) for v in parts)
    if not all(v > 0 for v in vals):
        raise ValueError("patch-shape values must be > 0")
    return vals  # type: ignore[return-value]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train proposal classifier")
    parser.add_argument("--frontend-checkpoint", default="models/frontend/centernet.pt")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--train-split", default="data/splits/train.txt")
    parser.add_argument("--val-split", default="data/splits/val.txt")
    parser.add_argument("--patch-shape", default="64,64,64")
    parser.add_argument("--out-dir", default="models/classifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neg-pos-ratio", type=int, default=3)
    parser.add_argument("--hard-negative-topk", type=int, default=0)
    parser.add_argument("--hard-negatives", default=None)
    parser.add_argument("--hard-neg-fraction", type=float, default=0.5)
    parser.add_argument("--n-peaks", type=int, default=1000)
    parser.add_argument("--edge-mm-set", default="12,20,32,48")
    parser.add_argument("--padding-ratio", type=float, default=0.2)
    args = parser.parse_args()

    _set_seed(args.seed)

    train_cases = build_case_records(
        data_root=args.data_root,
        split_file=args.train_split,
        require_labels=True,
        allow_missing_body_mask=True,
    )
    val_cases = build_case_records(
        data_root=args.data_root,
        split_file=args.val_split,
        require_labels=True,
        allow_missing_body_mask=True,
    ) if args.val_split else None

    patch_shape_xyz = _parse_patch_shape(args.patch_shape)
    edge_mm_set = tuple(float(v) for v in args.edge_mm_set.split(",") if v.strip())

    ds_cfg = ProposalDatasetConfig(
        patch_shape_xyz=patch_shape_xyz,
        n_peaks=args.n_peaks,
        edge_mm_set=edge_mm_set,
        padding_ratio=args.padding_ratio,
        neg_pos_ratio=args.neg_pos_ratio,
        seed=args.seed,
        hard_neg_path=args.hard_negatives,
        hard_neg_fraction=args.hard_neg_fraction,
    )

    train_ds = ProposalClassifierDataset(
        cases=train_cases,
        centernet_checkpoint_path=args.frontend_checkpoint,
        device=args.device,
        cfg=ds_cfg,
        require_labels=True,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if val_cases is not None:
        val_ds = ProposalClassifierDataset(
            cases=val_cases,
            centernet_checkpoint_path=args.frontend_checkpoint,
            device=args.device,
            cfg=ds_cfg,
            require_labels=True,
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model_cfg = ClassifierConfig()
    model = ProposalClassifier(model_cfg)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "classifier.pt")

    train_meta: Dict[str, object] = {
        "data_root": args.data_root,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "patch_shape": args.patch_shape,
        "n_peaks": args.n_peaks,
        "edge_mm_set": args.edge_mm_set,
        "padding_ratio": args.padding_ratio,
        "neg_pos_ratio": args.neg_pos_ratio,
        "hard_negative_topk": args.hard_negative_topk,
        "hard_negatives": args.hard_negatives,
        "hard_neg_fraction": args.hard_neg_fraction,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": args.device,
        "seed": args.seed,
    }

    history = train_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        device=torch.device(args.device),
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_path=ckpt_path,
        config=model_cfg,
        train_meta=train_meta,
    )

    LOGGER.info("training_complete checkpoint=%s history=%s", ckpt_path, history)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
