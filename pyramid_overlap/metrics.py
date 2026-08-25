from __future__ import annotations

import torch
from torch import Tensor


def confusion_matrix(
    prediction: Tensor,
    target: Tensor,
    num_classes: int = 4,
    ignore_index: int = -100,
) -> Tensor:
    prediction = prediction.detach().reshape(-1).long()
    target = target.detach().reshape(-1).long()
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    bins = target[valid] * num_classes + prediction[valid].clamp(0, num_classes - 1)
    return torch.bincount(bins, minlength=num_classes**2).reshape(num_classes, num_classes)


def segmentation_metrics(matrix: Tensor, eps: float = 1e-6) -> dict[str, object]:
    matrix = matrix.double()
    true_positive = matrix.diag()
    predicted = matrix.sum(0)
    target = matrix.sum(1)
    union = predicted + target - true_positive
    iou = true_positive / (union + eps)
    precision = true_positive / (predicted + eps)
    recall = true_positive / (target + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    present = target > 0
    return {
        "miou": float(iou[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "class_iou": iou.tolist(),
        "class_precision": precision.tolist(),
        "class_recall": recall.tolist(),
        "class_f1": f1.tolist(),
        "ov_iou": float(iou[3]),
        "ov_f1": float(f1[3]),
    }
