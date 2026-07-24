#!/usr/bin/env python3
"""CLI: run ATCC on an RTSP / webcam live stream (Phase 3)."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import load_config, resolve_path
from src.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ATCC Phase 3 — run vehicle counting on RTSP / webcam."
    )
    parser.add_argument(
        "--config",
        default="config/camera_config.yaml",
        help="Path to camera/pipeline YAML config",
    )
    parser.add_argument(
        "--rtsp",
        default=None,
        help="RTSP URL or webcam index (overrides source.path)",
    )
    parser.add_argument("--model", default=None, help="Override detection.model_path")
    parser.add_argument("--no-viz", action="store_true", help="Disable annotated MP4 writer")
    parser.add_argument("--log-level", default=None)
    return parser.parse_args()


def main() -> int:
    """Entrypoint for live streams."""
    args = parse_args()
    config = load_config(resolve_path(args.config))

    config.setdefault("source", {})
    config["source"]["type"] = "rtsp"
    if args.rtsp:
        config["source"]["path"] = args.rtsp
        config["source"]["url"] = args.rtsp
    if args.model:
        config.setdefault("detection", {})["model_path"] = args.model
    if args.no_viz:
        config.setdefault("visualization", {})["enabled"] = False

    level_name = (args.log_level or config.get("logging", {}).get("level", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    pipeline = Pipeline(config)

    def _handle_sig(_signum: int, _frame: object) -> None:
        logging.info("Stop signal received — flushing Excel and exiting")
        pipeline.request_stop()

    signal.signal(signal.SIGINT, _handle_sig)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_sig)

    report = pipeline.run()
    logging.info("Report written to %s", report)
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
