"""IEMOCAP-4 dialogues with speaker-ordered visual features."""
import numpy as np
import torch

from ..constants import FIELDS, MODALITIES


def vision_feature(raw, speaker, policy):
    """IEMOCAP stores one visual feature per interlocutor; order them as (speaker, other).
    interaction = [own, other, own-other, own*other]; stack_speaker_other = 2-token sequence [own, other]."""
    female, male = (np.asarray(raw[k], dtype=np.float32).reshape(-1) for k in ("F", "M"))
    own, other = (female, male) if str(speaker).upper().startswith("F") else (male, female)
    if policy == "interaction":
        return np.concatenate([own, other, own - other, own * other], axis=0)
    return np.stack([own, other], axis=0)


def load_dialogues(data, split, policy, device):
    dialogues = []
    for video in sorted(str(v) for v in data[split]):
        segments = [str(s) for s in data["video_ids"][video]]
        speakers = [str(s) for s in data["speakers"][video]]
        features = {}
        for m in MODALITIES:
            rows = [vision_feature(data[FIELDS[m]][s], spk, policy) if m == "vision"
                    else np.asarray(data[FIELDS[m]][s], dtype=np.float32) for s, spk in zip(segments, speakers)]
            features[m] = torch.tensor(np.stack(rows, axis=0), dtype=torch.float32, device=device)
        dialogues.append({
            "features": features,
            "speakers": torch.tensor([0 if s.upper().startswith("M") else 1 for s in speakers], device=device),
            "labels": torch.tensor(np.asarray(data["labels"][video], dtype=np.int64).reshape(-1), device=device),
        })
    return dialogues
