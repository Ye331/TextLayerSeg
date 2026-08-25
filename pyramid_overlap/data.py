from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class OverlapManifestDataset(Dataset[dict[str, Tensor | str]]):
    """Read RGB images and four-class PNG masks from a JSONL manifest.

    Each line must contain ``image`` and ``mask``. Relative paths are resolved
    against the manifest directory unless ``data_root`` is supplied.
    """

    def __init__(
        self,
        manifest: str | Path,
        image_size: int | None = None,
        data_root: str | Path | None = None,
        ignore_label: int = 255,
        ignore_index: int = -100,
    ) -> None:
        self.manifest = Path(manifest).resolve()
        if not self.manifest.is_file():
            raise FileNotFoundError(self.manifest)
        self.root = Path(data_root).resolve() if data_root else self.manifest.parent
        self.image_size = image_size
        self.ignore_label = int(ignore_label)
        self.ignore_index = int(ignore_index)
        self.records: list[dict[str, object]] = []
        with self.manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if "image" not in record or "mask" not in record:
                    raise ValueError(f"line {line_number}: expected image and mask fields")
                record["_line_number"] = line_number
                self.records.append(record)
        if not self.records:
            raise ValueError(f"empty manifest: {self.manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def _resolve(self, value: object) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else self.root / path

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        record = self.records[index]
        image_path = self._resolve(record["image"])
        mask_path = self._resolve(record["mask"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(mask_path) as source:
            mask = source.convert("L")
        if image.size != mask.size:
            raise ValueError(f"image/mask size mismatch: {image_path} vs {mask_path}")
        if self.image_size is not None:
            size = (self.image_size, self.image_size)
            image = image.resize(size, Image.Resampling.BILINEAR)
            mask = mask.resize(size, Image.Resampling.NEAREST)

        image_array = np.asarray(image, dtype=np.float32).copy() / 255.0
        mask_array = np.asarray(mask, dtype=np.int64).copy()
        mask_array[mask_array == self.ignore_label] = self.ignore_index
        valid = mask_array != self.ignore_index
        if np.any((mask_array[valid] < 0) | (mask_array[valid] > 3)):
            line_number = record["_line_number"]
            raise ValueError(f"line {line_number}: mask values must be 0,1,2,3 or 255")

        sample_id = str(record.get("id", image_path.stem))
        return {
            "image": torch.from_numpy(image_array).permute(2, 0, 1).contiguous(),
            "label": torch.from_numpy(mask_array),
            "id": sample_id,
        }
