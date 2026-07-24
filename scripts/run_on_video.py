#!/usr/bin/env python3
"""CLI: run the ATCC Phase-1 pipeline on a recorded video file."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/run_on_video.py` from repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import load_config, resolve_path
from src.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ATCC Phase 1 — run vehicle counting on a recorded video file."
    )
    parser.add_argument(
        "--config",
        default="config/camera_config.yaml",
        help="Path to camera/pipeline YAML config (default: config/camera_config.yaml)",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Override source.path with a video file path",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override detection.model_path (e.g. yolov8n.pt / yolov8s.pt)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable annotated video output",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override logging.level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def main() -> int:
    """Entrypoint."""
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)

    if args.video:
        config.setdefault("source", {})["type"] = "file"
        config["source"]["path"] = args.video
    if args.model:
        config.setdefault("detection", {})["model_path"] = args.model
    if args.no_viz:
        config.setdefault("visualization", {})["enabled"] = False

    level_name = (args.log_level or config.get("logging", {}).get("level", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if config.get("source", {}).get("type", "file") != "file":
        logging.error("run_on_video.py requires source.type=file. Use scripts/run_on_stream.py for RTSP.")
        return 2

    report = Pipeline(config).run()
    logging.info("Report written to %s", report)
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
