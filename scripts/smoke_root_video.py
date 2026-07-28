#!/usr/bin/env python3
"""Smoke-run the ATCC pipeline on the root WhatsApp video and save outputs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import load_config
from src.pipeline import Pipeline


def main() -> int:
    video = ROOT / "WhatsApp Video 2026-07-27 at 17.09.58.mp4"
    if not video.is_file():
        print(f"Video not found: {video}")
        return 2

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = load_config(ROOT / "config" / "camera_config.yaml")
    cfg["source"] = {"type": "file", "path": str(video)}
    # Faster CPU smoke settings while still covering the full video.
    cfg.setdefault("frame_sampling", {})["target_fps"] = 3
    cfg.setdefault("preprocessing", {})["imgsz"] = 416
    cfg.setdefault("visualization", {})["enabled"] = True
    cfg["visualization"]["output_dir"] = "outputs/annotated"
    cfg["visualization"]["publish_stats"] = False
    cfg.setdefault("export", {})
    cfg["export"]["mode"] = "session"
    cfg["export"]["filename_prefix"] = "smoke_whatsapp"
    cfg["export"]["output_dir"] = "outputs/reports"
    cfg.setdefault("logging", {})
    cfg["logging"]["per_frame_timing"] = False
    cfg["logging"]["progress_every_n_frames"] = 25
    # Prefer local weights already in repo root
    cfg.setdefault("detection", {})["model_path"] = str(ROOT / "yolov8n.pt")
    cfg["detection"]["device"] = "cpu"

    logging.info("Starting smoke pipeline on %s", video.name)
    report = Pipeline(cfg).run()
    print(f"\nPIPELINE OK")
    print(f"Excel report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
