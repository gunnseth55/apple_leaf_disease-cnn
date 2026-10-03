import copy
import time

import pandas as pd
import torch


def segmentation_totals(logits, targets, threshold=0.5):
    predictions = torch.sigmoid(logits) >= threshold
    targets = targets >= 0.5
    true_positive = (predictions & targets).sum().item()
    false_positive = (predictions & ~targets).sum().item()
    false_negative = (~predictions & targets).sum().item()
    return true_positive, false_positive, false_negative


def metrics_from_totals(true_positive, false_positive, false_negative, epsilon=1e-7):
    dice = (2 * true_positive + epsilon) / (
        2 * true_positive + false_positive + false_negative + epsilon
    )
    iou = (true_positive + epsilon) / (
        true_positive + false_positive + false_negative + epsilon
    )
    precision = (true_positive + epsilon) / (
        true_positive + false_positive + epsilon
    )
    recall = (true_positive + epsilon) / (
        true_positive + false_negative + epsilon
    )
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}


def calibrate_threshold(model, loader, device, thresholds=None):
    """Select the probability threshold with the best validation Dice score.

    Calibration must be performed on validation data only. Keeping it separate
    from ``run_epoch`` makes it harder to accidentally tune on the test split.
    """
    if thresholds is None:
        thresholds = [round(step / 100, 2) for step in range(10, 91, 5)]
    thresholds = [float(threshold) for threshold in thresholds]
    if not thresholds:
        raise ValueError("At least one calibration threshold is required")

    totals = {threshold: [0, 0, 0] for threshold in thresholds}
    model.eval()
    with torch.no_grad():
        for images, masks, _ in loader:
            logits = model(images.to(device)).cpu()
            for threshold in thresholds:
                batch_totals = segmentation_totals(logits, masks, threshold)
                totals[threshold] = [
                    current + batch
                    for current, batch in zip(totals[threshold], batch_totals)
                ]

    scores = {
        threshold: metrics_from_totals(*threshold_totals)
        for threshold, threshold_totals in totals.items()
    }
    # Prefer the threshold closest to 0.5 when Dice ties (common in tiny sets).
    best_threshold = max(
        thresholds, key=lambda threshold: (scores[threshold]["dice"], -abs(threshold - 0.5))
    )
    return best_threshold, scores[best_threshold]


def run_epoch(model, loader, criterion, device, threshold=0.5, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    sample_total = 0
    tp = fp = fn = 0
    with torch.set_grad_enabled(training):
        for images, masks, _ in loader:
            images, masks = images.to(device), masks.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            if training:
                loss.backward()
                optimizer.step()
            loss_total += loss.item() * len(images)
            sample_total += len(images)
            batch_tp, batch_fp, batch_fn = segmentation_totals(logits, masks, threshold)
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
    return {"loss": loss_total / sample_total, **metrics_from_totals(tp, fp, fn)}


def train_model(model, train_loader, val_loader, criterion, optimizer, device, config):
    history = []
    best_dice = -1.0
    best_epoch = None
    best_state = None
    for epoch in range(1, config.epochs + 1):
        started = time.time()
        train = run_epoch(
            model, train_loader, criterion, device, config.threshold, optimizer
        )
        val = run_epoch(model, val_loader, criterion, device, config.threshold)
        history.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train.items()},
                **{f"val_{key}": value for key, value in val.items()},
            }
        )
        if val["dice"] > best_dice:
            best_dice = val["dice"]
            best_epoch = epoch
            best_state = copy.deepcopy(
                {name: value.cpu() for name, value in model.state_dict().items()}
            )
        print(
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train loss {train['loss']:.4f} dice {train['dice']:.4f} | "
            f"val loss {val['loss']:.4f} dice {val['dice']:.4f} "
            f"IoU {val['iou']:.4f} | {time.time() - started:.1f}s"
        )
    model.load_state_dict(best_state)
    return pd.DataFrame(history), best_epoch, best_dice
