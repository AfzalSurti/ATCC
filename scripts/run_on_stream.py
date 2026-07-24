#!/usr/bin/env python3
"""CLI: run ATCC on an RTSP / live stream.

Phase 3 stub — not implemented in the Phase 1 MVP.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    """Print Phase-3 guidance and exit non-zero."""
    parser = argparse.ArgumentParser(
        description="ATCC Phase 3 stub — RTSP/live stream runner (not implemented yet)."
    )
    parser.add_argument("--config", default="config/camera_config.yaml")
    parser.add_argument("--rtsp", default=None, help="RTSP URL override")
    parser.parse_args()

    print(
        "Phase 3 not implemented.\n"
        "Live RTSP support + Streamlit dashboard are planned for Phase 3.\n"
        "For Phase 1, use: python scripts/run_on_video.py --video <path>\n"
        "See docs/TECHNICAL_DESIGN.md for the roadmap."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
