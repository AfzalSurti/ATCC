"""Tests for Phase 2/4 detector helpers (no GPU / weights required)."""

from __future__ import annotations

import pytest

from src.detector import create_detector, map_coco_to_target


def test_map_coco_to_target_known_classes() -> None:
    """COCO placeholder mapping should be stable for Phase 1 classes."""
    assert map_coco_to_target(2, "car") == (2, "car")
    assert map_coco_to_target(3, "motorcycle") == (1, "motorcycle")
    assert map_coco_to_target(7, "truck") == (4, "truck")


def test_create_detector_rejects_unknown_backend() -> None:
    """Factory must reject unsupported backends early."""
    with pytest.raises(ValueError, match="Unknown detection.backend"):
        create_detector({"detection": {"backend": "magic"}, "classes": {}})


def test_create_video_source_factory() -> None:
    """Factory should return LiveVideoSource for rtsp type without opening yet."""
    from src.video_source import LiveVideoSource, VideoSource, create_video_source

    file_src = create_video_source(
        {"source": {"type": "file", "path": "data/sample_videos/x.mp4"}, "frame_sampling": {"target_fps": 10}}
    )
    assert isinstance(file_src, VideoSource)
    assert not isinstance(file_src, LiveVideoSource)

    live = create_video_source(
        {"source": {"type": "rtsp", "path": "rtsp://example/stream"}, "frame_sampling": {"target_fps": 10}}
    )
    assert isinstance(live, LiveVideoSource)
