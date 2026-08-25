import torch

from pyramid_overlap import confusion_matrix, segmentation_metrics


def test_perfect_prediction() -> None:
    target = torch.tensor([[0, 1], [2, 3]])
    matrix = confusion_matrix(target, target)
    assert torch.equal(matrix, torch.eye(4, dtype=torch.long))
    metrics = segmentation_metrics(matrix)
    assert abs(metrics["miou"] - 1.0) < 5e-6
    assert abs(metrics["macro_f1"] - 1.0) < 5e-6
