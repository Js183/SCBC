"""Evaluation metrics: Acc2/F1 or Acc7 on CMU-MOSEI, accuracy / weighted F1 / macro F1 on IEMOCAP-4."""
from sklearn.metrics import accuracy_score, f1_score


def mosei_metrics(pred, true, task):
    """pred: predicted class indices; true: signed label values."""
    if task == "binary":
        pred = pred * 2 - 1
        return {"acc2": float(accuracy_score(true, pred)),
                "f1": float(f1_score(true, pred, average="weighted", zero_division=0))}
    return {"acc7": float(accuracy_score(true, pred - 3))}


def iemocap_metrics(pred, true):
    return {"accuracy": float(accuracy_score(true, pred)),
            "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
            "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0))}
