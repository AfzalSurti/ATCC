"""Time-bucketed count aggregation (default 15-minute windows)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

from src.schemas import CountEvent

logger = logging.getLogger(__name__)


@dataclass
class IntervalBucket:
    """Counts for one completed (or in-progress) time interval."""

    interval_start: datetime
    interval_end: datetime
    # (lane, direction, class_name) -> count
    counts: dict[tuple[str, str, str], int] = field(default_factory=dict)

    def add(self, event: CountEvent) -> None:
        """Increment the count for an event's lane/direction/class."""
        key = (event.lane, event.direction, event.class_name)
        self.counts[key] = self.counts.get(key, 0) + 1

    def is_empty(self) -> bool:
        """Return True if no vehicles were counted in this interval."""
        return not self.counts or all(v == 0 for v in self.counts.values())


class Aggregator:
    """Bucket count events into configurable time windows.

    On interval rollover, hands the completed bucket to a callback (typically
    the Excel exporter) and starts a fresh bucket.
    """

    def __init__(
        self,
        config: dict[str, Any],
        on_interval_complete: Callable[[IntervalBucket], None] | None = None,
        session_start: datetime | None = None,
    ) -> None:
        """Create an aggregator from export interval settings.

        Args:
            config: Full pipeline config.
            on_interval_complete: Called with each completed ``IntervalBucket``.
            session_start: Optional start timestamp (defaults to now when first event arrives).
        """
        export_cfg = config.get("export", {})
        self.interval = timedelta(minutes=float(export_cfg.get("interval_minutes", 15)))
        self.on_interval_complete = on_interval_complete
        self._session_start = session_start
        self._current: IntervalBucket | None = None
        self.completed: list[IntervalBucket] = []

    def _ensure_bucket(self, ts: datetime) -> IntervalBucket:
        """Create the current bucket aligned to the interval grid if needed."""
        if self._current is not None:
            return self._current

        start = self._session_start or ts
        # Align to wall interval optionally; for MVP use session_start as first window start.
        self._current = IntervalBucket(
            interval_start=start,
            interval_end=start + self.interval,
        )
        logger.info(
            "Aggregator window opened: %s → %s",
            self._current.interval_start.isoformat(),
            self._current.interval_end.isoformat(),
        )
        return self._current

    def _roll(self, ts: datetime) -> None:
        """Complete current bucket(s) until ``ts`` falls inside the open window."""
        while self._current is not None and ts >= self._current.interval_end:
            done = self._current
            self.completed.append(done)
            logger.info(
                "Interval complete %s → %s | %d class-lane keys",
                done.interval_start.isoformat(),
                done.interval_end.isoformat(),
                len(done.counts),
            )
            if self.on_interval_complete is not None:
                self.on_interval_complete(done)

            next_start = done.interval_end
            self._current = IntervalBucket(
                interval_start=next_start,
                interval_end=next_start + self.interval,
            )

    def observe_time(self, ts: datetime) -> None:
        """Advance the clock (call each frame even if no counts).

        Args:
            ts: Current frame timestamp.
        """
        self._ensure_bucket(ts)
        self._roll(ts)

    def add_event(self, event: CountEvent, ts: datetime) -> None:
        """Add a count event into the bucket covering ``ts``.

        Args:
            event: Counted vehicle crossing.
            ts: Timestamp of the frame where the crossing was detected.
        """
        self.observe_time(ts)
        assert self._current is not None
        self._current.add(event)

    def flush(self) -> IntervalBucket | None:
        """Flush the in-progress interval (e.g. on graceful shutdown).

        Returns:
            The flushed bucket, or None if nothing was open / empty and caller
            still wants a record — empty buckets are still exported so the
            report shows zero-activity windows when desired. Always exports
            if a bucket exists.
        """
        if self._current is None:
            return None

        done = self._current
        self._current = None
        self.completed.append(done)
        logger.info(
            "Flushed partial interval %s → %s",
            done.interval_start.isoformat(),
            done.interval_end.isoformat(),
        )
        if self.on_interval_complete is not None:
            self.on_interval_complete(done)
        return done
