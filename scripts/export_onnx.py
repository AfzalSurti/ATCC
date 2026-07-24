#!/usr/bin/env python3
"""Phase 4 — export a YOLO checkpoint to ONNX (and optionally TensorRT engine)."""

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
    """Parse export CLI arguments."""
    parser = argparse.ArgumentParser(description="Export ATCC YOLO weights to ONNX / TensorRT.")
    parser.add_argument("--weights", default="yolov8n.pt", help="Input .pt checkpoint")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="Export device (TensorRT needs CUDA)")
    parser.add_argument(
        "--outdir",
        default="outputs/models",
        help="Directory for exported artifacts",
    )
    parser.add_argument(
        "--tensorrt",
        action="store_true",
        help="Also export TensorRT .engine (requires NVIDIA TensorRT)",
    )
    parser.add_argument("--half", action="store_true", help="FP16 export when supported")
    parser.add_argument("--simplify", action="store_true", default=True, help="Simplify ONNX graph")
    return parser.parse_args()


def main() -> int:
    """Export model artifacts for edge deployment."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    from ultralytics import YOLO

    weights = args.weights
    resolved = resolve_path(weights)
    load_path = str(resolved) if resolved.is_file() else weights

    outdir = resolve_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = YOLO(load_path)
    logger.info("Exporting ONNX from %s", load_path)
    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        simplify=args.simplify,
        half=args.half,
        device=args.device if args.device else None,
    )
    onnx_src = Path(str(onnx_path))
    onnx_dst = outdir / onnx_src.name
    if onnx_src.resolve() != onnx_dst.resolve():
        onnx_dst.write_bytes(onnx_src.read_bytes())
    logger.info("ONNX → %s", onnx_dst)

    engine_dst = None
    if args.tensorrt:
        logger.info("Exporting TensorRT engine (requires CUDA + TensorRT)...")
        try:
            engine_path = model.export(
                format="engine",
                imgsz=args.imgsz,
                half=args.half,
                device=args.device or "0",
            )
            engine_src = Path(str(engine_path))
            engine_dst = outdir / engine_src.name
            if engine_src.resolve() != engine_dst.resolve():
                engine_dst.write_bytes(engine_src.read_bytes())
            logger.info("Engine → %s", engine_dst)
        except Exception as exc:  # noqa: BLE001
            logger.error("TensorRT export failed: %s", exc)
            logger.error("ONNX export succeeded; use detection.backend: onnx instead.")
            return 2

    print(
        "\nWire into config/camera_config.yaml:\n"
        f"  detection.backend: onnx\n"
        f"  detection.model_path: \"{onnx_dst.as_posix()}\"\n"
        + (
            f"  # or detection.backend: tensorrt / model_path: \"{engine_dst.as_posix()}\"\n"
            if engine_dst
            else ""
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
