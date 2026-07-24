#!/usr/bin/env python3
"""Phase 2 — fine-tune YOLOv8 on a labeled India-specific ATCC dataset.

This script does NOT invent labels or synthetic images. It trains only when you
provide a real YOLO-format dataset (see config/dataset.example.yaml).

Typical flow:
  1. Label with classes from config/india_classes.yaml (Roboflow or CVAT).
  2. Export YOLO detect format under data/datasets/<name>/.
  3. python scripts/train.py --data data/datasets/<name>/data.yaml
  4. Point camera_config detection.model_path at runs/.../weights/best.pt
     and set classes.mode: custom
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import resolve_path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse training CLI arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for ATCC India classes.")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to YOLO data.yaml (must exist with real train/val images+labels).",
    )
    parser.add_argument("--model", default="yolov8s.pt", help="Base checkpoint to fine-tune.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="", help='e.g. "0", "cpu", or empty for auto')
    parser.add_argument("--project", default="runs/detect", help="Ultralytics project dir")
    parser.add_argument("--name", default="atcc_india", help="Run name under project/")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early-stopping patience (epochs without improvement).",
    )
    return parser.parse_args()


def _validate_dataset(data_yaml: Path) -> None:
    """Fail fast if the dataset YAML or split folders are missing."""
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}\n"
            "Label real video in Roboflow/CVAT, export YOLO format, then copy "
            "config/dataset.example.yaml → <dataset>/data.yaml.\n"
            "Do not generate fake labels."
        )

    import yaml

    with data_yaml.open("r", encoding="utf-8") as fh:
        meta = yaml.safe_load(fh) or {}

    root = Path(str(meta.get("path", data_yaml.parent)))
    if not root.is_absolute():
        root = (ROOT / root).resolve()

    for split in ("train", "val"):
        rel = meta.get(split)
        if not rel:
            raise ValueError(f"data.yaml missing '{split}' key")
        split_dir = root / str(rel)
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Missing {split} images dir: {split_dir}\n"
                "Export a real labeled dataset before training."
            )


def main() -> int:
    """Run Ultralytics fine-tuning."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    data_yaml = resolve_path(args.data)
    try:
        _validate_dataset(data_yaml)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2

    from ultralytics import YOLO

    model = YOLO(args.model)
    train_kwargs: dict = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(resolve_path(args.project)),
        "name": args.name,
        "workers": args.workers,
        "patience": args.patience,
        "exist_ok": True,
    }
    if args.device:
        train_kwargs["device"] = args.device

    logger.info("Starting fine-tune: model=%s data=%s", args.model, data_yaml)
    results = model.train(**train_kwargs)

    best = Path(str(getattr(results, "save_dir", resolve_path(args.project) / args.name))) / "weights" / "best.pt"
    # Ultralytics stores save_dir on trainer; fall back to conventional path.
    conventional = resolve_path(args.project) / args.name / "weights" / "best.pt"
    weights = best if best.is_file() else conventional

    logger.info("Training finished. Best weights (expected): %s", weights)
    print(
        "\nNext steps:\n"
        f"  1. Set detection.model_path: \"{weights.as_posix()}\"\n"
        "  2. Set classes.mode: custom\n"
        "  3. Set classes.target to match your data.yaml names order\n"
        "  4. Re-run scripts/run_on_video.py\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
