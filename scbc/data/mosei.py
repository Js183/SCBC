"""CMU-MOSEI segments for the binary (Acc2/F1) and seven-class (Acc7) tasks."""
import numpy as np
import torch
from torch.utils.data import TensorDataset

from ..constants import FIELDS, MODALITIES


def load_split(data, split, task):
    """Segments of a split as (features per modality, class index, signed label value)."""
    features = {m: [] for m in MODALITIES}
    labels, values = [], []
    for video in sorted(str(v) for v in data[split]):
        for index, segment in enumerate(data["video_ids"][video]):
            raw = float(np.asarray(data["labels"][video][index], dtype=np.float32).reshape(-1)[0])
            if task == "binary":
                if raw == 0.0:
                    continue
                label, value = int(raw > 0.0), (1 if raw > 0.0 else -1)
            else:
                value = int(np.clip(np.rint(raw), -3, 3))
                label = value + 3
            for m in MODALITIES:
                features[m].append(np.asarray(data[FIELDS[m]][str(segment)], dtype=np.float32))
            labels.append(label)
            values.append(value)
    tensors = [torch.from_numpy(np.stack(features[m])) for m in MODALITIES]
    return TensorDataset(*tensors, torch.tensor(labels), torch.tensor(values))
