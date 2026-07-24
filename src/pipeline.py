"""End-to-end ATCC pipeline orchestration."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv

from src.aggregator import Aggregator, IntervalBucket
from src.config_loader import load_config, resolve_path
from src.counter import LaneCounter
from src.detector import Detector
from src.exporter import ExcelExporter
from src.preprocessing import FramePreprocessor
from src.schemas import Detection
from src.tracker import Tracker
from src.video_source import FramePacket, VideoSource

logger = logging.getLogger(__name__)


class Pipeline:
    """Wire video → preprocess → detect → track → count → aggregate → export."""

    def __init__(self, config: dict[str, Any] | str | Path) -> None:
        """Build all pipeline stages from a config dict or YAML path.

        Args:
            config: Config mapping or path to ``camera_config.yaml``.
        """
        if isinstance(config, (str, Path)):
            self.config = load_config(config)
        else:
            self.config = config

        self.video = VideoSource(self.config)
        self.preprocessor = FramePreprocessor(self.config)
        self.detector = Detector(self.config)
        self.tracker = Tracker(self.config)
        self.exporter = ExcelExporter(self.config)
        self.aggregator = Aggregator(
            self.config,
            on_interval_complete=self._on_interval_complete,
        )
        self.counter = LaneCounter(self.config)

        vis_cfg = self.config.get("visualization", {})
        self.visualize = bool(vis_cfg.get("enabled", True))
        self.vis_dir = resolve_path(str(vis_cfg.get("output_dir", "outputs/annotated")))
        self._writer: cv2.VideoWriter | None = None
        self._timing = bool(self.config.get("logging", {}).get("per_frame_timing", True))
        self.total_events = 0

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
                f"{lane.name}:{lane.display_count}",
                mid,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
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

    def run(self) -> Path:
        """Execute the full pipeline until the video ends; flush on exit.

        Returns:
            Path to the Excel workbook written.
        """
        t_pipeline_start = time.perf_counter()

        try:
            with self.video:
                if self.video.width and self.video.height:
                    self.preprocessor.build_roi_mask(self.video.width, self.video.height)

                for packet in self.video:
                    self._process_frame(packet)

        finally:
            # Graceful shutdown: flush partial interval to Excel.
            self.aggregator.flush()
            if self._writer is not None:
                self._writer.release()
                self._writer = None

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
            if self._timing:
                logger.debug("frame %d skipped (motion gate)", packet.frame_index)
            return

        detections = self.detector.predict(frame)
        t_det = time.perf_counter()

        tracked = self.tracker.update(detections)
        t_track = time.perf_counter()

        events = self.counter.update(tracked)
        for event in events:
            self.aggregator.add_event(event, packet.timestamp)
            self.total_events += 1
        self.aggregator.observe_time(packet.timestamp)
        t_count = time.perf_counter()

        if self.visualize:
            self._ensure_writer(packet.frame, self.video.target_fps)
            annotated = self._annotate(packet.frame, tracked)
            if self._writer is not None:
                self._writer.write(annotated)

        if self._timing:
            logger.info(
                "frame %d | det=%.1fms track=%.1fms count=%.1fms total=%.1fms | dets=%d tracks=%d events=%d",
                packet.frame_index,
                (t_det - t0) * 1000,
                (t_track - t_det) * 1000,
                (t_count - t_track) * 1000,
                (time.perf_counter() - t0) * 1000,
                len(detections),
                len(tracked),
                len(events),
            )


def run_from_config(config_path: str | Path) -> Path:
    """Convenience entry: load YAML and run the pipeline.

    Args:
        config_path: Path to camera config YAML.

    Returns:
        Path to the generated Excel report.
    """
    pipeline = Pipeline(config_path)
    return pipeline.run()
