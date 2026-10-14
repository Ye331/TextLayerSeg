# Overlapping Text Layer Segmentation

**Paper Accepted at CISP-BMEI 2026**

<img width="1402" height="946" alt="image" src="https://github.com/user-attachments/assets/f8eca7ef-c72d-400a-ad18-bb51af2adcdc" />


Our related paper has been accepted at the **2026 19th International Congress on Image and Signal Processing, BioMedical Engineering and Informatics (CISP-BMEI 2026)**.

*Paper and citation details will be updated upon official publication.*

---

A lightweight, auditable implementation of a context-enhanced Feature Pyramid Network (FPN) for pixel-level overlapping text layer segmentation.

**Architecture:** RGB → ResNet-50 (C2–C5) → Multi-Dilation Context → FPN → P2 Decoder → BG/T1/T2/OV

| ID | Class | Description |
|---|---|---|
| 0 | BG | Background |
| 1 | T1 | Pixels exclusive to text layer 1 |
| 2 | T2 | Pixels exclusive to text layer 2 |
| 3 | OV | Overlapping pixels |

Full layer masks are reconstructed as `T1 ∪ OV` and `T2 ∪ OV`.

This project focuses on pixel-level segmentation, without claiming OCR improvements or recovery of fully occluded characters.

## Repository Contents

- ResNet-50 + FPN + multi-dilation context model
- Training, evaluation, inference, and unit tests
- JSONL data loader and synthetic toy-data generator
- Sanitized ablation results

Datasets, fonts, pretrained weights, checkpoints, S/W branches, attention-based propagation, and private experimental assets are excluded.

## Installation

Requires Python 3.10+ and PyTorch.

```bash
pip install -e .
pip install pytest
```

Run validation:

```bash
python scripts/validate_release.py
python scripts/smoke_test.py
pytest -q
```

The full model contains **34,737,868 trainable parameters**. Smoke tests use a narrower FPN for faster execution.

## Quick Start

```bash
python examples/make_toy_data.py --output toy_data --size 128

python scripts/train.py \
  --train-manifest toy_data/train.jsonl \
  --val-manifest toy_data/val.jsonl \
  --output-dir runs/toy \
  --image-size 128 --batch-size 1 \
  --workers 0 --epochs 1 --max-steps 2

python scripts/evaluate.py \
  --manifest toy_data/val.jsonl \
  --checkpoint runs/toy/best.pt \
  --image-size 128
```

Toy data is intended for functional testing, not benchmarking.

## Data Format

Each JSONL entry specifies an RGB image and segmentation mask:

```json
{"id":"sample-001","image":"images/sample-001.png","mask":"masks/sample-001.png"}
```

Mask labels are `0–3`, with `255` reserved as the ignore label. Relative paths are resolved against the manifest directory.

See [DATA_FORMAT.md](DATA_FORMAT.md).

## Training and Inference

```bash
python scripts/train.py \
  --train-manifest /path/to/train.jsonl \
  --val-manifest /path/to/dev.jsonl \
  --backbone-checkpoint /path/to/resnet50.pth \
  --output-dir runs/fpn_context
```

Single-image inference:

```bash
python scripts/predict.py \
  --image input.png \
  --checkpoint runs/fpn_context/best.pt \
  --output prediction.png
```

The backbone loader supports common torchvision ResNet-50 state-dict formats.

Configuration: [configs/fpn_context.json](configs/fpn_context.json).

The reference trainer does not reproduce the original four-GPU training pipeline or scale-robust augmentation.

## Experimental Results

Internal ablation study (`seed-2026`): 120 training source pages, 14 disjoint development pages, and 8,192 effective updates.

| Model | Dev mIoU |
|---|---:|
| C2-only | 0.7202 |
| FPN | 0.7708 |
| **FPN + Context** | **0.7800** |

**OV IoU:** 0.6395 (FPN + Context).

Results are from a single-seed internal development set, not official benchmarks or SOTA claims. Data and checkpoints are not redistributed pending authorization.

See [evidence/VALIDATION.md](evidence/VALIDATION.md).

## Project Structure

```text
pyramid_overlap/   Model, data, losses, metrics
scripts/           Training, evaluation, inference
examples/          Toy data generation
configs/           Model configuration
tests/             Unit tests
evidence/          Sanitized experimental records
```

## Citation

The related paper has been accepted at CISP-BMEI 2026. Official BibTeX and DOI will be added when available.
