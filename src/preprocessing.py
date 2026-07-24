"""Frame preprocessing: ROI mask, optional CLAHE, motion-skip helper.

Detection itself is YOLO — this module never uses MOG2/KNN background
subtraction as a detection mechanism.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FramePreprocessor:
    """Apply ROI masking and optional lighting normalization to frames."""

    def __init__(self, config: dict[str, Any], frame_size: tuple[int, int] | None = None) -> None:
        """Initialize preprocessor from config.

        Args:
            config: Full pipeline config.
            frame_size: Optional ``(width, height)`` used to build ROI mask early.
        """
        roi_cfg = config.get("roi", {})
        prep_cfg = config.get("preprocessing", {})
        clahe_cfg = prep_cfg.get("clahe", {})
        motion_cfg = prep_cfg.get("motion_skip", {})

        self.roi_enabled: bool = bool(roi_cfg.get("enabled", False))
        self.roi_polygon: list[list[int]] = list(roi_cfg.get("polygon") or [])
        self.imgsz: int = int(prep_cfg.get("imgsz", 640))

        self.clahe_enabled: bool = bool(clahe_cfg.get("enabled", False))
        self.clahe_clip: float = float(clahe_cfg.get("clip_limit", 2.0))
        tile = clahe_cfg.get("tile_grid_size", [8, 8])
        self.clahe_tile: tuple[int, int] = (int(tile[0]), int(tile[1]))

        self.motion_skip_enabled: bool = bool(motion_cfg.get("enabled", False))
        self.min_diff_ratio: float = float(motion_cfg.get("min_diff_ratio", 0.01))

        self._roi_mask: np.ndarray | None = None
        self._prev_gray: np.ndarray | None = None
        self._clahe = (
            cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_tile)
            if self.clahe_enabled
            else None
        )

        if frame_size is not None and self.roi_enabled and self.roi_polygon:
            self.build_roi_mask(frame_size[0], frame_size[1])

    def build_roi_mask(self, width: int, height: int) -> None:
        """Build a binary ROI mask from the configured polygon.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        if not self.roi_polygon:
            logger.warning("ROI enabled but polygon empty — treating as full frame")
            self._roi_mask = None
            return

        mask = np.zeros((height, width), dtype=np.uint8)
        pts = np.array(self.roi_polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        self._roi_mask = mask
        logger.info("ROI mask built for %dx%d with %d vertices", width, height, len(self.roi_polygon))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply ROI mask and optional CLAHE; return processed BGR frame.

        Resize/letterbox to model input is handled by Ultralytics at inference
        time using ``imgsz`` — we keep original resolution for geometry
        (counting lines / track centroids stay in source pixel space).

        Args:
            frame: BGR image from the video source.

        Returns:
            Processed BGR image (same spatial size as input).
        """
        out = frame

        if self.roi_enabled:
            if self._roi_mask is None:
                h, w = frame.shape[:2]
                self.build_roi_mask(w, h)
            if self._roi_mask is not None:
                out = cv2.bitwise_and(out, out, mask=self._roi_mask)

        if self._clahe is not None:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_ch = self._clahe.apply(l_ch)
            out = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

        return out

    def should_skip_inference(self, frame: np.ndarray) -> bool:
        """Optional motion gate: skip YOLO when the scene is nearly static.

        This is an optimization only — not a detector. Separated from YOLO logic.

        Args:
            frame: Current (possibly preprocessed) BGR frame.

        Returns:
            True if inference should be skipped for this frame.
        """
        if not self.motion_skip_enabled:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return False

        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray
        changed = float(np.count_nonzero(diff > 25)) / float(diff.size)
        return changed < self.min_diff_ratio
