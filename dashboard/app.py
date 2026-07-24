"""Streamlit live-view dashboard for ATCC (Phase 3).

Run:
  streamlit run dashboard/app.py -- --config config/camera_config.yaml

Or via helper:
  python scripts/run_dashboard.py
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import load_config, resolve_path
from src.pipeline import Pipeline
from src.stats_store import GLOBAL_STATS


def _parse_streamlit_args() -> tuple[str, str | None]:
    """Extract --config / --source from streamlit's passthrough args."""
    config = "config/camera_config.yaml"
    source = None
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    i = 0
    while i < len(argv):
        if argv[i] == "--config" and i + 1 < len(argv):
            config = argv[i + 1]
            i += 2
            continue
        if argv[i] == "--source" and i + 1 < len(argv):
            source = argv[i + 1]
            i += 2
            continue
        i += 1
    return config, source


def _ensure_pipeline_thread(config_path: str, source_override: str | None) -> None:
    """Start the ATCC pipeline once in a background thread."""
    if st.session_state.get("pipeline_started"):
        return

    config = load_config(resolve_path(config_path))
    if source_override:
        config.setdefault("source", {})
        # Heuristic: digit → webcam; rtsp:// → live; else file
        if source_override.isdigit() or source_override.lower().startswith("rtsp"):
            config["source"]["type"] = "rtsp"
            config["source"]["path"] = source_override
            config["source"]["url"] = source_override
        else:
            config["source"]["type"] = "file"
            config["source"]["path"] = source_override

    # Dashboard owns visualization via stats store; skip writing MP4 by default.
    config.setdefault("visualization", {})
    config["visualization"]["enabled"] = False
    config["visualization"]["publish_stats"] = True
    config.setdefault("logging", {})
    config["logging"]["per_frame_timing"] = False

    pipeline = Pipeline(config, stats_store=GLOBAL_STATS)
    st.session_state["pipeline"] = pipeline
    st.session_state["pipeline_started"] = True

    def _run() -> None:
        try:
            pipeline.run()
        except Exception as exc:  # noqa: BLE001
            GLOBAL_STATS.set_status(f"error: {exc}")

    thread = threading.Thread(target=_run, name="atcc-pipeline", daemon=True)
    st.session_state["pipeline_thread"] = thread
    thread.start()


def main() -> None:
    """Render the live ATCC dashboard."""
    st.set_page_config(page_title="ATCC Live Dashboard", layout="wide")
    st.title("ATCC — Live Traffic Counter")
    st.caption("Phase 3 dashboard · YOLO + ByteTrack · lane line counting")

    config_path, source_override = _parse_streamlit_args()

    with st.sidebar:
        st.header("Controls")
        config_path = st.text_input("Config path", value=config_path)
        source_override = st.text_input(
            "Source override (file path, rtsp://..., or webcam index)",
            value=source_override or "",
        ) or None
        refresh_ms = st.slider("UI refresh (ms)", 200, 2000, 500, 100)
        if st.button("Start / Restart pipeline", type="primary"):
            # Stop previous run if any
            old = st.session_state.get("pipeline")
            if old is not None:
                old.request_stop()
            st.session_state["pipeline_started"] = False
            _ensure_pipeline_thread(config_path, source_override)
            st.success("Pipeline thread started")
        if st.button("Stop pipeline"):
            pipe = st.session_state.get("pipeline")
            if pipe is not None:
                pipe.request_stop()
                st.warning("Stop requested — flushing Excel")

    if not st.session_state.get("pipeline_started"):
        st.info("Click **Start / Restart pipeline** in the sidebar to begin.")
        return

    snap = GLOBAL_STATS.snapshot()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", snap.status)
    c2.metric("Total counts", snap.total_events)
    c3.metric("Processing FPS", f"{snap.fps:.1f}")
    c4.metric("Report", Path(snap.report_path).name if snap.report_path else "—")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Live view")
        if snap.frame_bgr is not None:
            rgb = cv2.cvtColor(snap.frame_bgr, cv2.COLOR_BGR2RGB)
            st.image(rgb, channels="RGB", use_container_width=True)
        else:
            st.write("Waiting for first frame…")

    with right:
        st.subheader("Per-lane counts")
        if snap.lane_counts:
            st.dataframe(
                pd.DataFrame(
                    [{"Lane": k, "Count": v} for k, v in snap.lane_counts.items()]
                ),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.write("No lane counts yet.")

        st.subheader("Per-class (current window)")
        if snap.class_counts:
            st.dataframe(
                pd.DataFrame(
                    [{"Class": k, "Count": v} for k, v in snap.class_counts.items()]
                ),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.write("No class counts yet.")

        st.subheader("Recent events")
        if snap.recent_events:
            st.dataframe(pd.DataFrame(snap.recent_events), hide_index=True, use_container_width=True)
        else:
            st.write("—")

    # Auto-refresh
    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=refresh_ms, key="atcc_refresh")
    except ImportError:
        st.caption(f"Install streamlit-autorefresh for auto UI refresh, or manually rerun (~{refresh_ms} ms).")


if __name__ == "__main__":
    main()
