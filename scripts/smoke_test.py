from __future__ import annotations

import json

import torch

from pyramid_overlap import ContextualPyramidSegmenter, pyramid_segmentation_loss


def main() -> None:
    torch.manual_seed(2026)
    model = ContextualPyramidSegmenter(fpn_channels=64)
    image = torch.rand(1, 3, 96, 128)
    label = torch.randint(0, 4, (1, 96, 128))
    outputs = model(image)
    loss, _ = pyramid_segmentation_loss(outputs, label)
    loss.backward()
    report = {
        "status": "pass",
        "logits_shape": list(outputs["logits"].shape),
        "finite_logits": bool(torch.isfinite(outputs["logits"]).all()),
        "finite_loss": bool(torch.isfinite(loss)),
        "classifier_gradient": model.classifier.weight.grad is not None,
        "trainable_parameters_reduced_smoke_model": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    if not all(
        report[key]
        for key in ("finite_logits", "finite_loss", "classifier_gradient")
    ):
        raise RuntimeError(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
