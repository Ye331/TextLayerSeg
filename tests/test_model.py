import torch

from pyramid_overlap import ContextualPyramidSegmenter, pyramid_segmentation_loss


def test_forward_backward_and_parameter_count() -> None:
    model = ContextualPyramidSegmenter()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 34_737_868
    image = torch.rand(1, 3, 64, 80)
    label = torch.randint(0, 4, (1, 64, 80))
    outputs = model(image)
    assert outputs["logits"].shape == (1, 4, 64, 80)
    assert len(outputs["aux_logits"]) == 2
    loss, parts = pyramid_segmentation_loss(outputs, label)
    assert torch.isfinite(loss)
    assert set(parts) == {"ce", "focal", "dice", "aux", "total"}
    loss.backward()
    assert model.classifier.weight.grad is not None


def test_default_loss_matches_frozen_class_weights() -> None:
    logits = torch.randn(1, 4, 8, 8)
    labels = torch.randint(0, 4, (1, 8, 8))
    outputs = {"logits": logits, "aux_logits": []}
    default, _ = pyramid_segmentation_loss(outputs, labels)
    explicit, _ = pyramid_segmentation_loss(
        outputs, labels, class_weights=(0.25, 1.0, 1.0, 2.0)
    )
    torch.testing.assert_close(default, explicit)


def test_release_has_no_geometry_transport_modules() -> None:
    names = {
        name.lower()
        for name, _ in ContextualPyramidSegmenter(fpn_channels=64).named_modules()
    }
    forbidden = ("skeleton", "width", "attention", "transport", "depth")
    assert not any(token in name for name in names for token in forbidden)
