#!/usr/bin/env python3
"""Launch the Streamlit ATCC dashboard."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    """Spawn ``streamlit run dashboard/app.py`` with optional passthrough args."""
    parser = argparse.ArgumentParser(description="Launch ATCC Streamlit dashboard")
    parser.add_argument("--config", default="config/camera_config.yaml")
    parser.add_argument("--source", default=None, help="file / rtsp / webcam override")
    args, extra = parser.parse_known_args()

    app = ROOT / "dashboard" / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--",
        "--config",
        args.config,
    ]
    if args.source:
        cmd.extend(["--source", args.source])
    cmd.extend(extra)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
