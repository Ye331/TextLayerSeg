from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pyramid_overlap import (
    ContextualPyramidSegmenter,
    OverlapManifestDataset,
    confusion_matrix,
    pyramid_segmentation_loss,
    save_model_checkpoint,
    segmentation_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the released FPN+Context model")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--backbone-checkpoint")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.inference_mode()
def evaluate(model, loader, device: torch.device) -> dict[str, object]:
    model.eval()
    matrix = torch.zeros(4, 4, dtype=torch.long)
    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        prediction = model(image)["logits"].argmax(1)
        matrix += confusion_matrix(prediction.cpu(), label.cpu())
    return segmentation_metrics(matrix)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_set = OverlapManifestDataset(
        args.train_manifest, args.image_size, args.data_root
    )
    val_set = OverlapManifestDataset(args.val_manifest, args.image_size, args.data_root)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    model = ContextualPyramidSegmenter(
        backbone_checkpoint=args.backbone_checkpoint
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_miou = -1.0
    step = 0
    history: list[dict[str, object]] = []
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(image)
            loss, parts = pyramid_segmentation_loss(outputs, label)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step >= args.max_steps:
                break

        metrics = evaluate(model, val_loader, device)
        record = {"epoch": epoch + 1, "step": step, "metrics": metrics}
        history.append(record)
        print(json.dumps(record))
        if float(metrics["miou"]) > best_miou:
            best_miou = float(metrics["miou"])
            save_model_checkpoint(model, output_dir / "best.pt", record)
        if step >= args.max_steps:
            break

    save_model_checkpoint(model, output_dir / "last.pt", {"step": step})
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
