"""Train SCBC on IEMOCAP-4 (dialogue level) with one fixed observed modality subset.

    python train_iemocap4.py --subset lva --output runs/iemocap4/lva
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from scbc import MODALITIES, SUBSETS
from scbc.backbones.dialogue import NUM_CLASSES
from scbc.constants import LETTERS
from scbc.data import load_dialogues
from scbc.engine import evaluate_iemocap, train_epoch_iemocap
from scbc.models import IemocapSCBC
from scbc.utils import load_config, seed_iemocap

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True, choices=SUBSETS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "iemocap4.yaml"))
    args = parser.parse_args()

    cfg, train_cfg, model_cfg = load_config(args.config, args.subset)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_iemocap(cfg["seed"])
    device = torch.device("cuda")

    with open(ROOT / cfg["data"], "rb") as f:
        data = pickle.load(f)
    train = load_dialogues(data, "train", model_cfg["visual_policy"], device)
    test = load_dialogues(data, "test", model_cfg["visual_policy"], device)
    del data
    feature_shapes = {m: tuple(train[0]["features"][m].shape[1:]) for m in MODALITIES}
    train_labels = torch.cat([d["labels"] for d in train]).cpu().numpy()
    counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_weights = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** train_cfg["class_weight_power"],
                                 dtype=torch.float32, device=device)

    student_modalities = tuple(m for m in MODALITIES if m in {LETTERS[ch] for ch in args.subset})
    model = IemocapSCBC(feature_shapes, student_modalities, class_weights, model_cfg).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    log_path = out_dir / "metrics.jsonl"
    log_path.write_text("")
    order = list(range(len(train)))
    for epoch in range(train_cfg["epochs"]):
        model.corrector.set_epoch(epoch)
        train_loss = train_epoch_iemocap(model, train, order, optimizer, train_cfg["grad_clip"])
        record = {"epoch": epoch, "train_loss": train_loss, "test": evaluate_iemocap(model, test)}
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"epoch {epoch:3d} loss {record['train_loss']:.4f} | test "
              + " ".join(f"{k}={v * 100:.2f}" for k, v in record["test"].items()), flush=True)


if __name__ == "__main__":
    main()
