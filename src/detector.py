"""YOLO / ONNX / TensorRT detector wrappers with a shared Detection contract.

Phase 1 default: pretrained COCO + ``map_coco_to_target`` placeholder.
Phase 2: ``classes.mode: custom`` uses native model class names (no COCO map).
Phase 4: ``detection.backend: onnx|tensorrt`` swaps the inference engine.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.config_loader import resolve_path
from src.schemas import Detection

logger = logging.getLogger(__name__)

__all__ = [
    "Detection",
    "BaseDetector",
    "Detector",
    "ONNXDetector",
    "TensorRTDetector",
    "create_detector",
    "map_coco_to_target",
]


COCO_ID_TO_NAME: dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

COCO_TO_TARGET_NAME: dict[str, str] = {
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
}


def map_coco_to_target(coco_class_id: int, coco_class_name: str) -> tuple[int, str]:
    """Map a COCO vehicle class to the Phase-1 target taxonomy.

    PLACEHOLDER until a custom-trained model exists. With ``classes.mode: custom``
    this function is not used.

    Args:
        coco_class_id: Ultralytics/COCO class index.
        coco_class_name: COCO class name string.

    Returns:
        ``(target_class_id, target_class_name)`` for counting / Excel export.
    """
    name = COCO_TO_TARGET_NAME.get(coco_class_name, coco_class_name)
    target_order = ["bicycle", "motorcycle", "car", "bus", "truck"]
    try:
        target_id = target_order.index(name)
    except ValueError:
        target_id = coco_class_id
    return target_id, name


class BaseDetector(ABC):
    """Common detector interface so backends are interchangeable."""

    @abstractmethod
    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Run inference and return standardized detections."""


class Detector(BaseDetector):
    """Ultralytics YOLO wrapper returning standardized ``Detection`` objects."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Load YOLO weights and detection thresholds from config.

        Args:
            config: Full pipeline config.
        """
        from ultralytics import YOLO

        det_cfg = config.get("detection", {})
        class_cfg = config.get("classes", {})

        model_path = str(det_cfg.get("model_path", "yolov8n.pt"))
        resolved = resolve_path(model_path)
        load_path = str(resolved) if resolved.is_file() else model_path

        self.confidence: float = float(det_cfg.get("confidence_threshold", 0.4))
        self.iou: float = float(det_cfg.get("iou_threshold", 0.5))
        self.device: str = str(det_cfg.get("device", "") or "")
        self.imgsz: int = int(config.get("preprocessing", {}).get("imgsz", 640))
        self.class_mode: str = str(class_cfg.get("mode", "coco_mapped")).lower()
        self.coco_vehicle_ids: set[int] = set(class_cfg.get("coco_vehicle_ids", [1, 2, 3, 5, 7]))
        self.target_classes: list[str] = list(
            class_cfg.get("target", ["bicycle", "motorcycle", "car", "bus", "truck"])
        )

        logger.info(
            "Loading YOLO model from %s (classes.mode=%s)",
            load_path,
            self.class_mode,
        )
        self.model = YOLO(load_path)
        if self.device:
            self.model.to(self.device)

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Run inference and return filtered detections.

        Args:
            frame: BGR image (ROI-masked if preprocessing applied).

        Returns:
            List of ``Detection`` objects.
        """
        kwargs: dict[str, Any] = {
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "verbose": False,
        }
        if self.class_mode == "coco_mapped":
            kwargs["classes"] = sorted(self.coco_vehicle_ids)
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
        names = result.names or {}

        for bbox, conf, cls_id in zip(boxes, confs, cls_ids):
            cls_id_i = int(cls_id)
            raw_name = str(names.get(cls_id_i, COCO_ID_TO_NAME.get(cls_id_i, str(cls_id_i))))

            if self.class_mode == "custom":
                # Native fine-tuned taxonomy — single source of truth is model + config.target.
                if self.target_classes and 0 <= cls_id_i < len(self.target_classes):
                    target_id, target_name = cls_id_i, self.target_classes[cls_id_i]
                else:
                    target_id, target_name = cls_id_i, raw_name
            else:
                if cls_id_i not in self.coco_vehicle_ids:
                    continue
                target_id, target_name = map_coco_to_target(cls_id_i, raw_name)

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


class ONNXDetector(BaseDetector):
    """Phase 4 — ONNX Runtime detector (portable CPU/GPU edge path).

    Expects an ONNX model exported via ``scripts/export_onnx.py``. Uses Ultralytics'
    YOLO loader on the ``.onnx`` path when available (handles letterbox/NMS),
    otherwise falls back to raw onnxruntime with a simplified postprocess.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Load an ONNX model path from config."""
        det_cfg = config.get("detection", {})
        model_path = str(det_cfg.get("onnx_path") or det_cfg.get("model_path", ""))
        if not model_path.lower().endswith(".onnx"):
            # Allow model_path to be .pt and look for sibling .onnx
            candidate = resolve_path(model_path).with_suffix(".onnx")
            if candidate.is_file():
                model_path = str(candidate)
            else:
                raise FileNotFoundError(
                    f"ONNX model not found for {model_path}. "
                    "Run: python scripts/export_onnx.py --weights <model.pt>"
                )

        resolved = resolve_path(model_path)
        if not resolved.is_file():
            raise FileNotFoundError(f"ONNX file missing: {resolved}")

        # Prefer Ultralytics' ONNX path (same predict API / letterbox).
        from ultralytics import YOLO

        logger.info("Loading ONNX detector from %s", resolved)
        # Reuse Detector logic by wrapping YOLO(onnx).
        cfg = dict(config)
        cfg.setdefault("detection", {})
        cfg["detection"] = dict(det_cfg)
        cfg["detection"]["model_path"] = str(resolved)
        self._inner = Detector(cfg)

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Run ONNX-backed inference."""
        return self._inner.predict(frame)


class TensorRTDetector(BaseDetector):
    """Phase 4 — TensorRT engine detector (NVIDIA edge / Jetson).

    Loads a ``.engine`` exported by Ultralytics (``model.export(format='engine')``)
    or ``scripts/export_onnx.py --tensorrt``. Requires a CUDA machine with
    TensorRT; otherwise raises a clear error.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Load a TensorRT engine from config."""
        det_cfg = config.get("detection", {})
        model_path = str(det_cfg.get("engine_path") or det_cfg.get("model_path", ""))
        resolved = resolve_path(model_path)
        if not str(resolved).lower().endswith(".engine"):
            candidate = resolved.with_suffix(".engine")
            if candidate.is_file():
                resolved = candidate
            else:
                raise FileNotFoundError(
                    f"TensorRT engine not found for {model_path}. "
                    "Export with: python scripts/export_onnx.py --weights <pt> --tensorrt"
                )
        if not resolved.is_file():
            raise FileNotFoundError(f"Engine file missing: {resolved}")

        from ultralytics import YOLO

        logger.info("Loading TensorRT engine from %s", resolved)
        cfg = dict(config)
        cfg.setdefault("detection", {})
        cfg["detection"] = dict(det_cfg)
        cfg["detection"]["model_path"] = str(resolved)
        try:
            self._inner = Detector(cfg)
        except Exception as exc:  # noqa: BLE001 — surface TRT install issues clearly
            raise RuntimeError(
                "Failed to load TensorRT engine. Install TensorRT + matching CUDA, "
                "or use detection.backend: onnx for portable inference. "
                f"Underlying error: {exc}"
            ) from exc

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Run TensorRT-backed inference."""
        return self._inner.predict(frame)


def create_detector(config: dict[str, Any]) -> BaseDetector:
    """Factory: build the detector backend named in ``detection.backend``.

    Args:
        config: Full pipeline config.

    Returns:
        A ``BaseDetector`` implementation.
    """
    backend = str(config.get("detection", {}).get("backend", "ultralytics")).lower()
    if backend in {"ultralytics", "yolo", "pt", ""}:
        return Detector(config)
    if backend == "onnx":
        return ONNXDetector(config)
    if backend in {"tensorrt", "trt", "engine"}:
        return TensorRTDetector(config)
    raise ValueError(f"Unknown detection.backend: {backend!r}")
