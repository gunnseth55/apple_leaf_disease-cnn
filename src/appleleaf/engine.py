import copy
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def run_epoch(model, loader, criterion, device, optimizer=None) -> dict:
    is_training = optimizer is not None
    model.train(is_training)
    loss_numerator = 0.0
    loss_denominator = 0.0
    labels_all, predictions_all = [], []

    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if is_training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()
            batch_weight = (
                criterion.weight[labels].sum().item()
                if criterion.weight is not None
                else len(labels)
            )
            loss_numerator += loss.item() * batch_weight
            loss_denominator += batch_weight
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(logits.argmax(dim=1).cpu().tolist())

    return {
        "loss": loss_numerator / loss_denominator,
        "accuracy": accuracy_score(labels_all, predictions_all),
        "macro_f1": f1_score(
            labels_all,
            predictions_all,
            labels=list(range(4)),
            average="macro",
            zero_division=0,
        ),
    }


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    history = []
    best_val_f1, best_state, best_epoch = -1.0, None, None
    for epoch in range(1, epochs + 1):
        started = time.time()
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(
                {name: value.cpu() for name, value in model.state_dict().items()}
            )
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Train acc: {train_metrics['accuracy']:.3f} | "
            f"Val acc: {val_metrics['accuracy']:.3f} | "
            f"Val F1: {val_metrics['macro_f1']:.3f} | "
            f"Time: {time.time() - started:.1f}s"
        )
    model.load_state_dict(best_state)
    return pd.DataFrame(history), best_epoch, best_val_f1


@torch.no_grad()
def predict(model, loader, device) -> dict[str, np.ndarray]:
    model.eval()
    labels_all, predictions_all, probabilities_all = [], [], []
    for images, labels in loader:
        logits = model(images.to(device))
        probabilities = torch.softmax(logits, dim=1)
        labels_all.extend(labels.tolist())
        predictions_all.extend(logits.argmax(dim=1).cpu().tolist())
        probabilities_all.extend(probabilities.cpu().tolist())
    return {
        "labels": np.asarray(labels_all),
        "predictions": np.asarray(predictions_all),
        "probabilities": np.asarray(probabilities_all),
    }


def classification_metrics(results: dict, class_names: list[str]) -> dict:
    labels, predictions = results["labels"], results["predictions"]
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels,
            predictions,
            labels=list(range(len(class_names))),
            average="macro",
            zero_division=0,
        ),
        "report": classification_report(
            labels,
            predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            digits=4,
            zero_division=0,
        ),
    }

