"""Optional SegFormer fine-tune on an OSM-rasterized golf dataset.

Requires the `train` extra: `pip install -e ".[train]"`.
The resulting checkpoint (and optional ONNX export) is meant for
`SEGMENTATION_WEIGHTS` later — this module is not imported by the live pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.catalog import SEMANTIC_CLASSES

DEFAULT_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"


def train_segformer(
    *,
    dataset_dir: Path,
    out_dir: Path,
    model_name: str = DEFAULT_MODEL,
    epochs: int = 12,
    batch_size: int = 2,
    learning_rate: float = 6e-5,
    export_onnx: bool = True,
) -> dict[str, Any]:
    try:
        import torch
        from PIL import Image
        from torch.utils.data import Dataset
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            'Install the training extra first: pip install -e ".[train]"'
        ) from exc

    label_map_path = dataset_dir / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(f"Missing {label_map_path}")
    id2label = {int(k): v for k, v in json.loads(label_map_path.read_text(encoding="utf-8")).items()}
    if not id2label:
        id2label = {i: name for i, name in enumerate(SEMANTIC_CLASSES)}
    label2id = {name: i for i, name in id2label.items()}

    splits = json.loads((dataset_dir / "splits.json").read_text(encoding="utf-8"))
    train_images = [dataset_dir / rel for rel in splits.get("train") or []]
    val_images = [dataset_dir / rel for rel in splits.get("val") or []]
    if not train_images:
        train_images = list((dataset_dir / "images").glob("*.jpg"))
        val_images = train_images[: max(1, len(train_images) // 8)]
        train_images = [p for p in train_images if p not in val_images]
    if not train_images:
        raise RuntimeError(f"No training images under {dataset_dir / 'images'}")

    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    class GolfSegDataset(Dataset):
        def __init__(self, images: list[Path]):
            self.images = [p for p in images if p.exists()]

        def __len__(self) -> int:
            return len(self.images)

        def __getitem__(self, index: int) -> dict[str, Any]:
            image_path = self.images[index]
            mask_path = dataset_dir / "masks" / (image_path.stem + ".png")
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path)
            encoded = processor(images=image, segmentation_maps=mask, return_tensors="pt")
            return {key: value.squeeze(0) for key, value in encoded.items()}

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"pixel_values": pixel_values, "labels": labels}

    out_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(out_dir / "runs"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch" if val_images else "no",
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=GolfSegDataset(train_images),
        eval_dataset=GolfSegDataset(val_images) if val_images else None,
        data_collator=collate,
    )
    trainer.train()
    checkpoint = out_dir / "segformer"
    trainer.save_model(str(checkpoint))
    processor.save_pretrained(str(checkpoint))

    onnx_path = None
    if export_onnx:
        onnx_path = _export_onnx(model, processor, out_dir / "segformer.onnx")

    summary = {
        "checkpoint": str(checkpoint),
        "onnx": str(onnx_path) if onnx_path else None,
        "model": model_name,
        "train_images": len(train_images),
        "val_images": len(val_images),
        "epochs": epochs,
        "classes": id2label,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _export_onnx(model: Any, processor: Any, dest: Path) -> Path | None:
    try:
        import torch
    except ImportError:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = getattr(processor, "size", None) or {}
    height = int(size.get("height") or 512)
    width = int(size.get("width") or 512)
    dummy = torch.randn(1, 3, height, width)
    model.eval()
    try:
        torch.onnx.export(
            model,
            dummy,
            str(dest),
            input_names=["pixel_values"],
            output_names=["logits"],
            opset_version=17,
            dynamo=False,
        )
        return dest
    except (OSError, ValueError, RuntimeError, TypeError):
        return None
