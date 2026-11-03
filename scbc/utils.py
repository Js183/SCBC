"""Seeding and configuration helpers."""
import random

import numpy as np
import torch
import yaml


def seed_mosei(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_iemocap(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_config(path, *keys):
    """Return the full config and the train / model settings found under subsets/<keys...>."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    run = cfg["subsets"]
    for key in keys:
        run = run[key]
    return cfg, run["train"], run["model"]
