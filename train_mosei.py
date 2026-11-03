"""Train SCBC on CMU-MOSEI with one fixed observed modality subset.

    python train_mosei.py --subset lva --task binary --output runs/mosei/binary/lva
    python train_mosei.py --subset lva --task seven  --output runs/mosei/seven/lva

The binary model gives Acc2/F1 (neutral segments dropped), the seven-class
model gives Acc7.
"""
import argparse
import json
import os
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scbc import MODALITIES, SUBSETS
from scbc.constants import observed_mask
from scbc.data import load_split
from scbc.engine import evaluate_mosei, train_epoch_mosei
from scbc.models import MoseiSCBC
from scbc.utils import load_config, seed_mosei

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True, choices=SUBSETS)
    parser.add_argument("--task", required=True, choices=("binary", "seven"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "mosei.yaml"))
    args = parser.parse_args()

    cfg, train_cfg, model_cfg = load_config(args.config, args.task, args.subset)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_mosei(cfg["seed"])
    device = torch.device("cuda")

    with open(ROOT / cfg["data"], "rb") as f:
        data = pickle.load(f)
    datasets = {split: load_split(data, key, args.task) for split, key in (("train", "train"), ("valid", "val"), ("test", "test"))}
    del data
    loaders = {split: DataLoader(ds, batch_size=train_cfg["batch_size"], shuffle=(split == "train"), pin_memory=True)
               for split, ds in datasets.items()}
    num_classes = 2 if args.task == "binary" else 7
    input_dims = {m: datasets["train"].tensors[i].shape[-1] for i, m in enumerate(MODALITIES)}
    model = MoseiSCBC(input_dims, num_classes, observed_mask(args.subset), model_cfg).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    log_path = out_dir / "metrics.jsonl"
    log_path.write_text("")
    for epoch in range(train_cfg["epochs"]):
        model.corrector.set_epoch(epoch)
        train_loss = train_epoch_mosei(model, loaders["train"], optimizer, train_cfg["grad_clip"], device)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": train_loss,
                  "valid": evaluate_mosei(model, loaders["valid"], device, args.task),
                  "test": evaluate_mosei(model, loaders["test"], device, args.task)}
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"epoch {epoch:3d} loss {record['train_loss']:.4f} | "
              + " | ".join(f"{s} " + " ".join(f"{k}={v * 100:.2f}" for k, v in record[s].items()) for s in ("valid", "test")),
              flush=True)


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
