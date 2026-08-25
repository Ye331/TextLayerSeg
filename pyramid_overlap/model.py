from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import resnet50
from torchvision.ops.misc import FrozenBatchNorm2d


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(32 if out_channels >= 32 else 1, out_channels),
            nn.GELU(),
        )


class ResidualRefine(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = 32 if channels >= 32 else 1
        self.block = nn.Sequential(
            ConvNormAct(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.block(x))


class PyramidContext(nn.Module):
    """Multi-dilation context module used only at the deepest feature C5."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if out_channels % 4:
            raise ValueError("out_channels must be divisible by four")
        branch_channels = out_channels // 4
        self.branches = nn.ModuleList(
            [
                ConvNormAct(in_channels, branch_channels, kernel_size=1),
                ConvNormAct(in_channels, branch_channels, dilation=3),
                ConvNormAct(in_channels, branch_channels, dilation=6),
                ConvNormAct(in_channels, branch_channels, dilation=9),
            ]
        )
        self.project = ConvNormAct(out_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class ContextualPyramidSegmenter(nn.Module):
    """Four-class BG/T1/T2/OV segmenter with a contextual feature pyramid.

    The implementation intentionally contains no S/W branch, depth input,
    attention gate, feature transport, selector, or post-processing rule.
    """

    class_names = ("BG", "T1", "T2", "OV")

    def __init__(
        self,
        num_classes: int = 4,
        fpn_channels: int = 256,
        backbone_checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__()
        if num_classes != 4:
            raise ValueError("This release is defined for BG/T1/T2/OV labels")

        backbone = resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.lateral2 = nn.Conv2d(256, fpn_channels, 1)
        self.lateral3 = nn.Conv2d(512, fpn_channels, 1)
        self.lateral4 = nn.Conv2d(1024, fpn_channels, 1)
        self.context5 = PyramidContext(2048, fpn_channels)
        self.smooth2 = ResidualRefine(fpn_channels)
        self.smooth3 = ResidualRefine(fpn_channels)
        self.smooth4 = ResidualRefine(fpn_channels)
        self.smooth5 = ResidualRefine(fpn_channels)

        self.decoder = nn.Sequential(
            ConvNormAct(fpn_channels, fpn_channels),
            ResidualRefine(fpn_channels),
            ConvNormAct(fpn_channels, 128),
            ResidualRefine(128),
        )
        self.classifier = nn.Conv2d(128, num_classes, 1)
        self.aux3 = nn.Conv2d(fpn_channels, num_classes, 1)
        self.aux4 = nn.Conv2d(fpn_channels, num_classes, 1)

        self.register_buffer(
            "image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )
        self.backbone_load_report: dict[str, Any] = {"loaded": False, "checkpoint": ""}
        if backbone_checkpoint is not None:
            self.backbone_load_report = self.load_backbone(backbone_checkpoint)

    @staticmethod
    def _upsample_add(high: Tensor, lateral: Tensor) -> Tensor:
        high = F.interpolate(high, lateral.shape[-2:], mode="bilinear", align_corners=False)
        return lateral + high

    def load_backbone(self, checkpoint: str | Path) -> dict[str, Any]:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]
        if not isinstance(state, dict):
            raise RuntimeError("checkpoint must contain a state dictionary")

        cleaned_state = {}
        for raw_key, value in state.items():
            key = str(raw_key).removeprefix("module.").removeprefix("backbone.")
            # FrozenBatchNorm2d has no num_batches_tracked buffer.  Dropping it
            # makes standard torchvision ResNet-50 checkpoints load cleanly.
            if key.startswith("fc.") or key.endswith("num_batches_tracked"):
                continue
            cleaned_state[key] = value
        state = cleaned_state
        holder = nn.ModuleDict(
            {
                "conv1": self.stem[0],
                "bn1": self.stem[1],
                "layer1": self.layer1,
                "layer2": self.layer2,
                "layer3": self.layer3,
                "layer4": self.layer4,
            }
        )
        result = holder.load_state_dict(state, strict=False)
        missing = [k for k in result.missing_keys if not k.endswith("num_batches_tracked")]
        if missing or result.unexpected_keys:
            raise RuntimeError(
                f"invalid ResNet-50 checkpoint: missing={missing}, "
                f"unexpected={list(result.unexpected_keys)}"
            )
        return {"loaded": True, "checkpoint": str(path.resolve()), "tensors": len(state)}

    def forward(self, image: Tensor) -> dict[str, Tensor | list[Tensor]]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"expected RGB tensor [B,3,H,W], got {tuple(image.shape)}")
        output_size = image.shape[-2:]
        x = (image - self.image_mean.to(image)) / self.image_std.to(image)
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.smooth5(self.context5(c5))
        p4 = self.smooth4(self._upsample_add(p5, self.lateral4(c4)))
        p3 = self.smooth3(self._upsample_add(p4, self.lateral3(c3)))
        p2 = self.smooth2(self._upsample_add(p3, self.lateral2(c2)))

        decoded = self.decoder(p2)
        logits = F.interpolate(
            self.classifier(decoded), output_size, mode="bilinear", align_corners=False
        )
        aux_logits = [
            F.interpolate(self.aux3(p3), output_size, mode="bilinear", align_corners=False),
            F.interpolate(self.aux4(p4), output_size, mode="bilinear", align_corners=False),
        ]
        return {"logits": logits, "aux_logits": aux_logits, "pyramid": [p2, p3, p4, p5]}
