"""End-to-end ATCC pipeline orchestration."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import supervision as sv

from src.aggregator import Aggregator, IntervalBucket
from src.config_loader import load_config, resolve_path
from src.counter import LaneCounter
from src.detector import create_detector
from src.exporter import ExcelExporter
from src.preprocessing import FramePreprocessor
from src.schemas import Detection
from src.stats_store import GLOBAL_STATS, StatsStore
from src.tracker import Tracker
from src.video_source import FramePacket, create_video_source

logger = logging.getLogger(__name__)


class Pipeline:
    """Wire video → preprocess → detect → track → count → aggregate → export."""

    def __init__(
        self,
        config: dict[str, Any] | str | Path,
        stats_store: StatsStore | None = None,
        on_frame: Callable[[np.ndarray, list[Detection]], None] | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Build all pipeline stages from a config dict or YAML path.

        Args:
            config: Config mapping or path to ``camera_config.yaml``.
            stats_store: Optional live stats publisher (defaults to global store).
            on_frame: Optional callback with annotated frame + tracks each tick.
            on_progress: Optional callback with live progress dict each frame.
        """
        if isinstance(config, (str, Path)):
            self.config = load_config(config)
        else:
            self.config = config

        self.stats = stats_store or GLOBAL_STATS
        self.on_frame = on_frame
        self.on_progress = on_progress
        self._stop = False

        self.video = create_video_source(self.config)
        self.preprocessor = FramePreprocessor(self.config)
        self.detector = create_detector(self.config)
        self.tracker = Tracker(self.config)
        self.exporter = ExcelExporter(self.config)
        self.aggregator = Aggregator(
            self.config,
            on_interval_complete=self._on_interval_complete,
        )
        self.counter = LaneCounter(self.config)

        vis_cfg = self.config.get("visualization", {})
        self.visualize = bool(vis_cfg.get("enabled", True))
        self.publish_stats = bool(vis_cfg.get("publish_stats", True))
        self.vis_dir = resolve_path(str(vis_cfg.get("output_dir", "outputs/annotated")))
        self._writer: cv2.VideoWriter | None = None
        self._timing = bool(self.config.get("logging", {}).get("per_frame_timing", True))
        self._progress_every = int(self.config.get("logging", {}).get("progress_every_n_frames", 1))
        self.total_events = 0
        self._fps_ema = 0.0
        self.frames_total_estimate = 0

    def request_stop(self) -> None:
        """Signal the run loop to exit (used by live stream / dashboard)."""
        self._stop = True
        stop_fn = getattr(self.video, "stop", None)
        if callable(stop_fn):
            stop_fn()

    def _on_interval_complete(self, bucket: IntervalBucket) -> None:
        """Export a completed interval and reset per-lane window mirrors."""
        self.exporter.write_bucket(bucket)
        self.counter.reset_window_counts()

    def _ensure_writer(self, frame: np.ndarray, fps: float) -> None:
        """Lazily open an annotated video writer."""
        if not self.visualize or self._writer is not None:
            return
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.vis_dir / f"annotated_{self.exporter.session_id}.mp4"
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(out_path), fourcc, max(fps, 1.0), (w, h))
        logger.info("Annotated video → %s", out_path)

    def _annotate(
        self,
        frame: np.ndarray,
        detections: list[Detection],
    ) -> np.ndarray:
        """Draw bounding boxes, track IDs, and counting lines."""
        canvas = frame.copy()
        if detections:
            xyxy = np.array([list(d.bbox) for d in detections], dtype=np.float32)
            tracker_id = np.array(
                [d.track_id if d.track_id is not None else -1 for d in detections],
                dtype=int,
            )
            labels = [
                f"#{d.track_id} {d.class_name} {d.confidence:.2f}"
                for d in detections
            ]
            sv_dets = sv.Detections(
                xyxy=xyxy,
                tracker_id=tracker_id,
                class_id=np.array([d.class_id for d in detections], dtype=int),
            )
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            canvas = box_annotator.annotate(canvas, sv_dets)
            canvas = label_annotator.annotate(canvas, sv_dets, labels=labels)

        for lane in self.counter.lanes:
            x1, y1 = map(int, lane.line_start)
            x2, y2 = map(int, lane.line_end)
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
            mid = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.putText(
                canvas,
                f"{lane.label}:{lane.display_count}",
                mid,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            f"counts={self.total_events}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def _publish(self, annotated: np.ndarray, packet: FramePacket, frame_ms: float) -> None:
        """Push live stats for the dashboard."""
        if not self.publish_stats:
            return
        inst_fps = 1000.0 / frame_ms if frame_ms > 0 else 0.0
        self._fps_ema = inst_fps if self._fps_ema == 0 else (0.8 * self._fps_ema + 0.2 * inst_fps)
        lane_counts = {lane.name: lane.display_count for lane in self.counter.lanes}
        class_counts: dict[str, int] = {}
        for lane in self.counter.lanes:
            for cls_name, n in lane.counts.items():
                class_counts[cls_name] = class_counts.get(cls_name, 0) + n
        self.stats.update_frame(
            frame_bgr=annotated,
            total_events=self.total_events,
            lane_counts=lane_counts,
            class_counts=class_counts,
            fps=self._fps_ema,
            timestamp=packet.timestamp,
            report_path=str(self.exporter.workbook_path),
        )

    def _emit_progress(self, packet: FramePacket, frame_ms: float) -> None:
        """Notify listeners + log terminal progress for the current frame."""
        done = packet.frame_index + 1
        total = max(self.frames_total_estimate, done)
        progress = min(1.0, done / float(total)) if total else 0.0
        inst_fps = 1000.0 / frame_ms if frame_ms > 0 else 0.0
        self._fps_ema = inst_fps if self._fps_ema == 0 else (0.8 * self._fps_ema + 0.2 * inst_fps)
        movement_counts = {lane.name: lane.display_count for lane in self.counter.lanes}
        class_counts = self.counter.class_totals()
        payload = {
            "frames_done": done,
            "frames_total": total,
            "progress": progress,
            "total_events": self.total_events,
            "movement_counts": movement_counts,
            "class_counts": class_counts,
            "fps": self._fps_ema,
            "message": f"Frame {done}/{total} · vehicles={self.total_events}",
        }
        if self.on_progress is not None:
            self.on_progress(payload)

        if self._timing or (done == 1 or done % max(1, self._progress_every) == 0):
            logger.info(
                "PROGRESS frame %d/%d (%.0f%%) | vehicles=%d | fps=%.1f | frame=%.0fms",
                done,
                total,
                progress * 100,
                self.total_events,
                self._fps_ema,
                frame_ms,
            )

    def run(self) -> Path:
        """Execute the full pipeline until the video ends or stop is requested.

        Returns:
            Path to the Excel workbook written.
        """
        t_pipeline_start = time.perf_counter()
        self.stats.set_status("running")
        logger.info("Pipeline starting…")

        try:
            with self.video:
                if self.video.width and self.video.height:
                    self.preprocessor.build_roi_mask(self.video.width, self.video.height)

                stride = self.video._sample_stride()
                if self.video.frame_count > 0:
                    self.frames_total_estimate = max(1, int(round(self.video.frame_count / stride)))
                else:
                    self.frames_total_estimate = 0
                logger.info(
                    "Video ready | %dx%d | source_frames=%d | estimated_process_frames=%d",
                    self.video.width,
                    self.video.height,
                    self.video.frame_count,
                    self.frames_total_estimate,
                )
                if self.on_progress is not None:
                    self.on_progress(
                        {
                            "frames_done": 0,
                            "frames_total": self.frames_total_estimate,
                            "progress": 0.0,
                            "total_events": 0,
                            "movement_counts": {},
                            "class_counts": {},
                            "fps": 0.0,
                            "message": "Starting frame processing…",
                        }
                    )

                for packet in self.video:
                    if self._stop:
                        break
                    self._process_frame(packet)

        except Exception:
            self.stats.set_status("error")
            raise
        finally:
            self.aggregator.flush()
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            self.stats.set_status("stopped")

        elapsed = time.perf_counter() - t_pipeline_start
        logger.info(
            "Pipeline finished in %.2fs | total count events=%d | report=%s",
            elapsed,
            self.total_events,
            self.exporter.workbook_path,
        )
        return self.exporter.workbook_path

    def _process_frame(self, packet: FramePacket) -> None:
        """Run one sampled frame through the full stage chain."""
        t0 = time.perf_counter()
        frame = self.preprocessor.apply(packet.frame)

        if self.preprocessor.should_skip_inference(frame):
            self.aggregator.observe_time(packet.timestamp)
            total_ms = (time.perf_counter() - t0) * 1000
            self._emit_progress(packet, total_ms)
            return

        detections = self.detector.predict(frame)
        tracked = self.tracker.update(detections)

        events = self.counter.update(tracked)
        for event in events:
            self.aggregator.add_event(event, packet.timestamp)
            self.total_events += 1
            self.stats.push_event(
                {
                    "track_id": event.track_id,
                    "movement": event.movement_id,
                    "arm": event.arm,
                    "flow": event.flow,
                    "lane": event.lane,
                    "direction": event.direction,
                    "class": event.class_name,
                    "time": packet.timestamp.isoformat(),
                }
            )
        self.aggregator.observe_time(packet.timestamp)

        need_annotate = self.visualize or self.publish_stats or self.on_frame is not None
        annotated = self._annotate(packet.frame, tracked) if need_annotate else packet.frame

        if self.visualize:
            self._ensure_writer(packet.frame, self.video.target_fps)
            if self._writer is not None:
                self._writer.write(annotated)

        if self.on_frame is not None:
            self.on_frame(annotated, tracked)

        total_ms = (time.perf_counter() - t0) * 1000
        self._publish(annotated if need_annotate else packet.frame, packet, total_ms)
        self._emit_progress(packet, total_ms)


def run_from_config(config_path: str | Path) -> Path:
    """Convenience entry: load YAML and run the pipeline.

    Args:
        config_path: Path to camera config YAML.

    Returns:
        Path to the generated Excel report.
    """
    pipeline = Pipeline(config_path)
    return pipeline.run()
