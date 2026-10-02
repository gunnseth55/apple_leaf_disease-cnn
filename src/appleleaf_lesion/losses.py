import torch
import torch.nn as nn
import torch.nn.functional as functional


def dice_loss(logits, targets, smooth=1.0):
    probabilities = torch.sigmoid(logits)
    dimensions = (1, 2, 3)
    intersection = (probabilities * targets).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
    return (1.0 - (2.0 * intersection + smooth) / (denominator + smooth)).mean()


def focal_loss(logits, targets, alpha=0.75, gamma=2.0):
    cross_entropy = functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    probabilities = torch.sigmoid(logits)
    probability_true = probabilities * targets + (1 - probabilities) * (1 - targets)
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_factor * (1 - probability_true).pow(gamma) * cross_entropy).mean()


class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5, alpha=0.75, gamma=2.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        return self.dice_weight * dice_loss(logits, targets) + self.focal_weight * focal_loss(
            logits, targets, self.alpha, self.gamma
        )
