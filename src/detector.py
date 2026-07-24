"""YOLO detector wrapper with COCO → target class mapping.

Phase 1 uses a pretrained COCO checkpoint. A clearly labeled mapping function
translates COCO vehicle classes into our report taxonomy. Once a custom
fine-tuned model exists (Phase 2), delete ``map_coco_to_target`` and feed
native class IDs through instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from ultralytics import YOLO

from src.config_loader import resolve_path

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Standardized detection object returned by the detector."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float
    track_id: int | None = None


# ---------------------------------------------------------------------------
# PLACEHOLDER — COCO → Phase-1 target taxonomy
# DELETE this mapping once a custom fine-tuned India-class model exists
# (Phase 2). Full India-specific classes require labeled data; do not invent them.
# ---------------------------------------------------------------------------
COCO_ID_TO_NAME: dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Identity mapping for Phase 1: COCO names == target report names.
# Kept as an explicit function so Phase 2 can replace it in one place.
COCO_TO_TARGET_NAME: dict[str, str] = {
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
}


def map_coco_to_target(coco_class_id: int, coco_class_name: str) -> tuple[int, str]:
    """Map a COCO vehicle class to the Phase-1 target taxonomy.

    PLACEHOLDER until a custom-trained model exists. Trivial to delete/replace
    in Phase 2: return the model's native ``(class_id, class_name)`` instead.

    Args:
        coco_class_id: Ultralytics/COCO class index.
        coco_class_name: COCO class name string.

    Returns:
        ``(target_class_id, target_class_name)`` for counting / Excel export.
    """
    name = COCO_TO_TARGET_NAME.get(coco_class_name, coco_class_name)
    # Stable local IDs for target names (order matches config classes.target).
    target_order = ["bicycle", "motorcycle", "car", "bus", "truck"]
    try:
        target_id = target_order.index(name)
    except ValueError:
        target_id = coco_class_id
    return target_id, name


class Detector:
    """Ultralytics YOLO wrapper returning standardized ``Detection`` objects.

    Structured so a TensorRT / Jetson backend can replace this class later
    without touching tracking or counting (Phase 4).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Load YOLO weights and detection thresholds from config.

        Args:
            config: Full pipeline config.
        """
        det_cfg = config.get("detection", {})
        class_cfg = config.get("classes", {})

        model_path = str(det_cfg.get("model_path", "yolov8n.pt"))
        # Allow bare weight names (ultralytics downloads) or project-relative paths.
        resolved = resolve_path(model_path)
        load_path = str(resolved) if resolved.is_file() else model_path

        self.confidence: float = float(det_cfg.get("confidence_threshold", 0.4))
        self.iou: float = float(det_cfg.get("iou_threshold", 0.5))
        self.device: str = str(det_cfg.get("device", "") or "")
        self.imgsz: int = int(config.get("preprocessing", {}).get("imgsz", 640))
        self.coco_vehicle_ids: set[int] = set(class_cfg.get("coco_vehicle_ids", [1, 2, 3, 5, 7]))
        self.target_classes: list[str] = list(
            class_cfg.get("target", ["bicycle", "motorcycle", "car", "bus", "truck"])
        )

        logger.info("Loading YOLO model from %s", load_path)
        self.model = YOLO(load_path)
        if self.device:
            self.model.to(self.device)

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Run inference and return filtered, mapped detections.

        Args:
            frame: BGR image (ROI-masked if preprocessing applied).

        Returns:
            List of ``Detection`` with target class names (COCO-mapped).
        """
        kwargs: dict[str, Any] = {
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "verbose": False,
            "classes": sorted(self.coco_vehicle_ids),
        }
        if self.device:
            kwargs["device"] = self.device

        results = self.model.predict(frame, **kwargs)
        if not results:
            return []

        result = results[0]
        detections: list[Detection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names or COCO_ID_TO_NAME

        for bbox, conf, cls_id in zip(boxes, confs, cls_ids):
            if int(cls_id) not in self.coco_vehicle_ids:
                continue
            coco_name = str(names.get(int(cls_id), COCO_ID_TO_NAME.get(int(cls_id), str(cls_id))))
            target_id, target_name = map_coco_to_target(int(cls_id), coco_name)
            x1, y1, x2, y2 = (float(v) for v in bbox)
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=target_id,
                    class_name=target_name,
                    confidence=float(conf),
                )
            )

        return detections


# ---------------------------------------------------------------------------
# Phase 4 stub — edge / TensorRT backend swap
# ---------------------------------------------------------------------------
class TensorRTDetector(Detector):
    """Phase 4 stub: TensorRT-backed detector with the same ``predict`` interface.

    TODO(Phase 4): Load an engine exported from ONNX and implement ``predict``
    without changing Tracker / LaneCounter / Aggregator.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Stub — not implemented in Phase 1."""
        raise NotImplementedError(
            "TensorRTDetector is a Phase 4 stub. Export ONNX/TensorRT later."
        )
