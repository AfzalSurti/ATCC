"""ByteTrack wrapper via the supervision library.

Do not hand-roll a tracker — ByteTrack assigns persistent track IDs used for
line-crossing deduplication in ``counter.py``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import supervision as sv

from src.schemas import Detection

logger = logging.getLogger(__name__)


class Tracker:
    """Wrap ``supervision.ByteTrack`` around standardized detections."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Create a ByteTrack instance from config tracking parameters.

        Args:
            config: Full pipeline config.
        """
        track_cfg = config.get("tracking", {})
        frame_sampling = config.get("frame_sampling", {})

        self.byte_track = sv.ByteTrack(
            track_activation_threshold=float(track_cfg.get("track_activation_threshold", 0.25)),
            lost_track_buffer=int(track_cfg.get("lost_track_buffer", 30)),
            minimum_matching_threshold=float(track_cfg.get("minimum_matching_threshold", 0.8)),
            frame_rate=int(track_cfg.get("frame_rate", frame_sampling.get("target_fps", 12))),
        )
        logger.info("ByteTrack initialized (frame_rate=%s)", track_cfg.get("frame_rate", frame_sampling.get("target_fps", 12)))

    @staticmethod
    def _to_sv_detections(detections: list[Detection]) -> sv.Detections:
        """Convert our Detection list to a supervision Detections object."""
        if not detections:
            return sv.Detections.empty()

        xyxy = np.array([list(d.bbox) for d in detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id = np.array([d.class_id for d in detections], dtype=int)
        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Assign persistent ``track_id`` values to the current frame's detections.

        Args:
            detections: Detector output for the current frame.

        Returns:
            Same detections annotated with ``track_id`` (may drop unmatched).
        """
        sv_dets = self._to_sv_detections(detections)
        tracked = self.byte_track.update_with_detections(sv_dets)

        if tracked.tracker_id is None or len(tracked) == 0:
            return []

        # Rebuild Detection list from tracked supervision output.
        # Class names are recovered via class_id → original detection lookup when possible.
        name_by_id = {d.class_id: d.class_name for d in detections}
        out: list[Detection] = []
        for i in range(len(tracked)):
            cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            tid = int(tracked.tracker_id[i])
            x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[i])
            out.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=cls_id,
                    class_name=name_by_id.get(cls_id, str(cls_id)),
                    confidence=conf,
                    track_id=tid,
                )
            )
        return out
