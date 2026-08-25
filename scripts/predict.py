from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pyramid_overlap import ContextualPyramidSegmenter, load_model_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a BG/T1/T2/OV label PNG")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with Image.open(args.image) as source:
        rgb = source.convert("RGB")
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    device = torch.device(args.device)
    model = ContextualPyramidSegmenter().to(device)
    load_model_checkpoint(model, args.checkpoint)
    model.eval()
    with torch.inference_mode():
        prediction = model(tensor.to(device))["logits"].argmax(1)[0].byte().cpu().numpy()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prediction, mode="L").save(output)


if __name__ == "__main__":
    main()
