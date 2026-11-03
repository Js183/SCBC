"""Training epochs and evaluation for both benchmarks."""
import random

import torch
from torch import nn

from .metrics import iemocap_metrics, mosei_metrics


def train_epoch_mosei(model, loader, optimizer, grad_clip, device):
    model.train()
    total = 0.0
    for *inputs, labels, _ in loader:
        inputs = [x.to(device, non_blocking=True) for x in inputs]
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model.loss(inputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.item())
    return total / len(loader)


@torch.inference_mode()
def evaluate_mosei(model, loader, device, task):
    model.eval()
    preds, values = [], []
    for *inputs, _, value in loader:
        inputs = [x.to(device, non_blocking=True) for x in inputs]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(inputs)
        preds.append(torch.softmax(logits, dim=-1).float().argmax(dim=-1).cpu())
        values.append(value)
    return mosei_metrics(torch.cat(preds).numpy(), torch.cat(values).numpy(), task)


def train_epoch_iemocap(model, dialogues, order, optimizer, grad_clip):
    model.train()
    random.shuffle(order)
    total = 0.0
    for index in order:
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(dialogues[index])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.item())
    return total / len(order)


@torch.no_grad()
def evaluate_iemocap(model, dialogues):
    model.eval()
    preds = [model(d).argmax(dim=-1).cpu() for d in dialogues]
    pred = torch.cat(preds).numpy()
    true = torch.cat([d["labels"].cpu() for d in dialogues]).numpy()
    return iemocap_metrics(true=true, pred=pred)
