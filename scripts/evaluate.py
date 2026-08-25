from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from pyramid_overlap import (
    ContextualPyramidSegmenter,
    OverlapManifestDataset,
    confusion_matrix,
    load_model_checkpoint,
    segmentation_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a segmentation checkpoint")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = OverlapManifestDataset(args.manifest, args.image_size, args.data_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = ContextualPyramidSegmenter().to(device)
    load_model_checkpoint(model, args.checkpoint)
    model.eval()
    matrix = torch.zeros(4, 4, dtype=torch.long)
    with torch.inference_mode():
        for batch in loader:
            prediction = model(batch["image"].to(device))["logits"].argmax(1).cpu()
            matrix += confusion_matrix(prediction, batch["label"])
    print(json.dumps(segmentation_metrics(matrix), indent=2))


if __name__ == "__main__":
    main()
