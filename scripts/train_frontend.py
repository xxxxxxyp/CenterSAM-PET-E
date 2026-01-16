from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

def _ensure_project_root_on_path() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_project_root_on_path()

from src.promptgen.data.case_index import build_case_records
from src.promptgen.frontend.centernet.dataset import CenterNetDataset, CenterNetTrainConfig
from src.promptgen.frontend.centernet.model import CenterNetConfig, CenterNetPET
from src.promptgen.frontend.centernet.train import train_centernet

LOGGER = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CenterNet frontend (heatmap-only)")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--train-split", default="data/splits/train.txt")
    parser.add_argument("--val-split", default="data/splits/val.txt")
    parser.add_argument("--out-dir", default="models/frontend")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma-vox", type=float, default=2.0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=None)
    args = parser.parse_args()

    _set_seed(args.seed)

    train_cases = build_case_records(
        data_root=args.data_root,
        split_file=args.train_split,
        require_labels=True,
        allow_missing_body_mask=True,
    )

    val_cases = None
    if args.val_split:
        val_cases = build_case_records(
            data_root=args.data_root,
            split_file=args.val_split,
            require_labels=True,
            allow_missing_body_mask=True,
        )

    train_cfg = CenterNetTrainConfig(sigma_vox=args.sigma_vox)
    train_ds = CenterNetDataset(train_cases, train_cfg, require_labels=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if val_cases is not None:
        val_ds = CenterNetDataset(val_cases, train_cfg, require_labels=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model_cfg = CenterNetConfig()
    model = CenterNetPET(model_cfg)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "centernet.pt")

    train_meta: Dict[str, Optional[float]] = {
        "data_root": args.data_root,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "sigma_vox": float(args.sigma_vox),
        "device": args.device,
        "seed": int(args.seed),
    }

    history = train_centernet(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        device=torch.device(args.device),
        epochs=args.epochs,
        lr=args.lr,
        save_every=args.save_every,
        checkpoint_path=ckpt_path,
        config=model_cfg,
        train_meta=train_meta,
    )

    LOGGER.info("training_complete checkpoint=%s history=%s", ckpt_path, history)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
