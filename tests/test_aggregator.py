"""Unit tests for time-bucketed Aggregator rollover and flush."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.aggregator import Aggregator, IntervalBucket
from src.counter import CountEvent


def _event(class_name: str = "car", lane: str = "lane_1") -> CountEvent:
    return CountEvent(
        track_id=1,
        lane=lane,
        direction="north",
        class_name=class_name,
        class_id=0,
    )


def test_interval_rollover_invokes_callback() -> None:
    """Crossing the interval boundary should complete the prior bucket."""
    completed: list[IntervalBucket] = []
    start = datetime(2026, 7, 24, 10, 0, 0, tzinfo=timezone.utc)
    agg = Aggregator(
        {"export": {"interval_minutes": 15}},
        on_interval_complete=completed.append,
        session_start=start,
    )

    agg.add_event(_event("car"), start + timedelta(minutes=1))
    agg.add_event(_event("bus"), start + timedelta(minutes=14))
    assert completed == []

    # Push past 15 minutes → first window closes.
    agg.observe_time(start + timedelta(minutes=15, seconds=1))
    assert len(completed) == 1
    assert completed[0].counts[("lane_1", "north", "car")] == 1
    assert completed[0].counts[("lane_1", "north", "bus")] == 1
    assert completed[0].interval_start == start
    assert completed[0].interval_end == start + timedelta(minutes=15)


def test_flush_exports_partial_window() -> None:
    """Graceful shutdown must flush the in-progress interval."""
    completed: list[IntervalBucket] = []
    start = datetime(2026, 7, 24, 12, 0, 0, tzinfo=timezone.utc)
    agg = Aggregator(
        {"export": {"interval_minutes": 15}},
        on_interval_complete=completed.append,
        session_start=start,
    )
    agg.add_event(_event("truck"), start + timedelta(minutes=3))
    flushed = agg.flush()
    assert flushed is not None
    assert len(completed) == 1
    assert completed[0].counts[("lane_1", "north", "truck")] == 1
    assert agg.flush() is None  # second flush is a no-op
