from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def make_sample(index: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    layer1 = Image.new("1", (size, size))
    layer2 = Image.new("1", (size, size))
    draw1 = ImageDraw.Draw(layer1)
    draw2 = ImageDraw.Draw(layer2)
    offset = 6 + index * 2
    width = max(3, size // 24)
    draw1.line((12, size // 2 - offset, size - 12, size // 2 + offset), fill=1, width=width)
    draw1.line((size // 3, 12, size // 3, size - 12), fill=1, width=width)
    draw2.line((12, size // 2 + offset, size - 12, size // 2 - offset), fill=1, width=width)
    draw2.line((2 * size // 3, 12, 2 * size // 3, size - 12), fill=1, width=width)
    a = np.asarray(layer1, dtype=bool)
    b = np.asarray(layer2, dtype=bool)
    label = np.zeros((size, size), dtype=np.uint8)
    label[a & ~b] = 1
    label[b & ~a] = 2
    label[a & b] = 3
    image = np.full((size, size, 3), 238, dtype=np.uint8)
    image[a & ~b] = (74, 78, 88)
    image[b & ~a] = (80, 82, 91)
    image[a & b] = (58, 61, 70)
    return image, label


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a license-free toy dataset")
    parser.add_argument("--output", default="toy_data")
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    splits = {"train": range(6), "val": range(6, 8)}
    for split, indices in splits.items():
        lines = []
        for index in indices:
            image, mask = make_sample(index, args.size)
            image_name = f"image_{index:02d}.png"
            mask_name = f"mask_{index:02d}.png"
            Image.fromarray(image).save(root / image_name)
            Image.fromarray(mask).save(root / mask_name)
            lines.append(json.dumps({"id": str(index), "image": image_name, "mask": mask_name}))
        (root / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root.resolve())


if __name__ == "__main__":
    main()
