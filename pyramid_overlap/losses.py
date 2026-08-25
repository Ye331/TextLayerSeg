from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def _validate(logits: Tensor, labels: Tensor, ignore_index: int) -> Tensor:
    if logits.ndim != 4:
        raise ValueError(f"logits must be [B,C,H,W], got {tuple(logits.shape)}")
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.shape != (logits.shape[0], logits.shape[2], logits.shape[3]):
        raise ValueError("label and logit spatial shapes do not match")
    labels = labels.long().to(logits.device)
    valid = labels != ignore_index
    invalid = valid & ((labels < 0) | (labels >= logits.shape[1]))
    if invalid.any():
        raise ValueError("labels must be BG=0, T1=1, T2=2, OV=3, or ignore_index")
    return labels


def segmentation_fusion_loss(
    logits: Tensor,
    labels: Tensor,
    class_weights: Sequence[float] = (0.25, 1.0, 1.0, 2.0),
    focal_gamma: float = 2.0,
    ignore_index: int = -100,
    eps: float = 1e-6,
) -> tuple[Tensor, dict[str, Tensor]]:
    labels = _validate(logits, labels, ignore_index)
    classes = logits.shape[1]
    weights = torch.as_tensor(class_weights, device=logits.device, dtype=logits.dtype)
    if weights.numel() != classes:
        raise ValueError("class_weights length must equal the number of classes")
    valid = labels != ignore_index
    safe = torch.where(valid, labels, torch.zeros_like(labels))
    valid_f = valid.to(logits.dtype)
    normalizer = valid_f.sum().clamp_min(1.0)

    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    target_log_probs = log_probs.gather(1, safe[:, None]).squeeze(1)
    target_probs = probs.gather(1, safe[:, None]).squeeze(1)
    pixel_weights = weights[safe] * valid_f
    ce = -(pixel_weights * target_log_probs).sum() / normalizer
    focal = -(
        pixel_weights * (1.0 - target_probs).pow(focal_gamma) * target_log_probs
    ).sum() / normalizer

    one_hot = F.one_hot(safe, classes).permute(0, 3, 1, 2).to(logits.dtype)
    mask = valid_f[:, None]
    intersection = (probs * one_hot * mask).sum((0, 2, 3))
    denominator = ((probs + one_hot) * mask).sum((0, 2, 3))
    dice_per_class = 1.0 - (2.0 * intersection + eps) / (denominator + eps)
    dice = (dice_per_class * weights).sum() / weights.sum()
    return ce + focal + dice, {"ce": ce, "focal": focal, "dice": dice}


def pyramid_segmentation_loss(
    outputs: dict[str, Tensor | list[Tensor]],
    labels: Tensor,
    aux_weight: float = 0.4,
    **kwargs,
) -> tuple[Tensor, dict[str, Tensor]]:
    main, parts = segmentation_fusion_loss(outputs["logits"], labels, **kwargs)
    aux_terms = [
        segmentation_fusion_loss(aux, labels, **kwargs)[0]
        for aux in outputs.get("aux_logits", [])
    ]
    aux = torch.stack(aux_terms).mean() if aux_terms else main.new_zeros(())
    total = main + float(aux_weight) * aux
    return total, {**parts, "aux": aux, "total": total}
