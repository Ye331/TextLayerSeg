import json

import numpy as np
from PIL import Image

from pyramid_overlap import OverlapManifestDataset


def test_manifest_dataset(tmp_path) -> None:
    image = np.full((12, 16, 3), 127, dtype=np.uint8)
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[:, 4:8] = 1
    mask[4:8, :] = 2
    mask[4:8, 4:8] = 3
    Image.fromarray(image).save(tmp_path / "image.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"id": "sample", "image": "image.png", "mask": "mask.png"})
        + "\n",
        encoding="utf-8",
    )
    sample = OverlapManifestDataset(manifest, image_size=8)[0]
    assert sample["image"].shape == (3, 8, 8)
    assert sample["label"].shape == (8, 8)
    assert set(sample["label"].unique().tolist()) == {0, 1, 2, 3}
