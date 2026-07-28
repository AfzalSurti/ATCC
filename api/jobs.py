"""DB-backed job registry — processing continues on the server after the tab closes."""

from __future__ import annotations

import copy
import logging
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy.orm import Session, joinedload

from api.db import get_session_factory
from api.models import Job, JobVideo
from src.config_loader import load_config, project_root, resolve_path
from src.junction import (
    expected_movements,
    junction_to_lane_configs,
    normalize_junction_type,
)

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Lifecycle states for an upload/process job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobManager:
    """Create upload jobs, persist them in Postgres, process in background threads."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        """Load base camera config used for every upload job."""
        root = project_root()
        self.config_path = resolve_path(config_path or "config/camera_config.yaml")
        self.upload_root = root / "data" / "uploads"
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._pipelines: dict[str, Any] = {}
        self._cancel_flags: set[str] = set()

    def _session(self) -> Session:
        return get_session_factory()()

    def create_job(
        self,
        user_id: int,
        saved_files: list[tuple[str, Path]],
        junction_type: str = "four_way",
    ) -> Job:
        """Register a new batch from saved upload paths for a user."""
        jtype = normalize_junction_type(junction_type)
        job_id = uuid.uuid4().hex[:12]
        db = self._session()
        try:
            job = Job(
                id=job_id,
                user_id=user_id,
                status=JobStatus.QUEUED.value,
                junction_type=jtype,
                progress=0.0,
            )
            db.add(job)
            for name, path in saved_files:
                db.add(
                    JobVideo(
                        id=uuid.uuid4().hex[:8],
                        job_id=job_id,
                        filename=name,
                        path=str(path),
                        status=JobStatus.QUEUED.value,
                        movement_counts={},
                        class_counts={},
                    )
                )
            db.commit()
            return self.get_job(job_id, user_id=user_id)  # type: ignore[return-value]
        finally:
            db.close()

    def get_job(self, job_id: str, user_id: int | None = None) -> Job | None:
        """Return a job by id (optionally scoped to user), or None."""
        db = self._session()
        try:
            q = (
                db.query(Job)
                .options(joinedload(Job.videos))
                .filter(Job.id == job_id)
            )
            if user_id is not None:
                q = q.filter(Job.user_id == user_id)
            job = q.first()
            if job is None:
                return None
            # Detach with videos loaded
            db.expunge(job)
            for v in job.videos:
                db.expunge(v)
            return job
        finally:
            db.close()

    def list_jobs(self, user_id: int) -> list[Job]:
        """Return the user's jobs newest-first."""
        db = self._session()
        try:
            jobs = (
                db.query(Job)
                .options(joinedload(Job.videos))
                .filter(Job.user_id == user_id)
                .order_by(Job.created_at.desc())
                .all()
            )
            for job in jobs:
                db.expunge(job)
                for v in job.videos:
                    db.expunge(v)
            return jobs
        finally:
            db.close()

    def start_job(self, job_id: str) -> Job:
        """Spawn a background thread to process all videos in the job."""
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status == JobStatus.RUNNING.value:
            return job

        with self._lock:
            self._cancel_flags.discard(job_id)

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id,),
            name=f"atcc-job-{job_id}",
            daemon=True,
        )
        thread.start()
        return job

    def cancel_job(self, job_id: str, user_id: int) -> Job:
        """Request stop on the active pipeline for a job owned by the user."""
        job = self.get_job(job_id, user_id=user_id)
        if job is None:
            raise KeyError(job_id)
        with self._lock:
            self._cancel_flags.add(job_id)
            pipe = self._pipelines.get(job_id)
        if pipe is not None:
            pipe.request_stop()
        self._update_job(job_id, status=JobStatus.CANCELLED.value)
        out = self.get_job(job_id, user_id=user_id)
        if out is None:
            raise KeyError(job_id)
        return out

    def mark_stale_running_as_failed(self) -> int:
        """On API startup, mark orphaned running jobs as failed (process restarted)."""
        db = self._session()
        try:
            rows = (
                db.query(Job)
                .filter(Job.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value]))
                .all()
            )
            count = 0
            for job in rows:
                # Only mark RUNNING as interrupted; leave QUEUED so we could resume later
                if job.status == JobStatus.RUNNING.value:
                    job.status = JobStatus.FAILED.value
                    job.error = "Server restarted while this job was running. Please re-upload."
                    for v in job.videos:
                        if v.status in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
                            v.status = JobStatus.FAILED.value
                            v.error = "Interrupted by server restart"
                            v.message = "Interrupted by server restart"
                    count += 1
            db.commit()
            return count
        finally:
            db.close()

    def job_to_dict(self, job: Job) -> dict[str, Any]:
        """Serialize a job for JSON responses."""
        created = job.created_at
        if isinstance(created, datetime):
            created_s = created.astimezone(timezone.utc).isoformat()
        else:
            created_s = str(created)
        return {
            "job_id": job.id,
            "created_at": created_s,
            "status": job.status,
            "progress": float(job.progress or 0.0),
            "error": job.error,
            "junction_type": job.junction_type,
            "expected_movements": expected_movements(job.junction_type),
            "videos": [
                {
                    "video_id": v.id,
                    "filename": v.filename,
                    "status": v.status,
                    "total_events": v.total_events,
                    "report_path": v.report_path,
                    "annotated_path": v.annotated_path,
                    "error": v.error,
                    "lane_counts": dict(v.movement_counts or {}),
                    "movement_counts": dict(v.movement_counts or {}),
                    "class_counts": dict(v.class_counts or {}),
                    "frames_done": v.frames_done,
                    "frames_total": v.frames_total,
                    "video_progress": float(v.video_progress or 0.0),
                    "fps": float(v.fps or 0.0),
                    "message": v.message or "",
                    "report_url": (
                        f"/api/jobs/{job.id}/videos/{v.id}/report" if v.report_path else None
                    ),
                }
                for v in job.videos
            ],
        }

    def _update_job(self, job_id: str, **fields: Any) -> None:
        db = self._session()
        try:
            job = db.get(Job, job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)
            job.updated_at = datetime.now(timezone.utc)
            db.commit()
        finally:
            db.close()

    def _update_video(self, video_id: str, **fields: Any) -> None:
        db = self._session()
        try:
            video = db.get(JobVideo, video_id)
            if video is None:
                return
            for key, value in fields.items():
                setattr(video, key, value)
            db.commit()
        finally:
            db.close()

    def update_video_path(self, video_id: str, path: str) -> None:
        """Update stored path after moving upload directory."""
        self._update_video(video_id, path=path)

    def _load_junction_preset(self, junction_type: str) -> dict[str, Any]:
        """Load arms from configs_examples/junction_*.yaml for the selected type."""
        jtype = normalize_junction_type(junction_type)
        preset_path = project_root() / "configs_examples" / f"junction_{jtype}.yaml"
        if not preset_path.is_file():
            return {}
        with preset_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return dict(data.get("junction") or {})

    def build_config_for_job(self, junction_type: str) -> dict[str, Any]:
        """Merge base camera config with the selected junction type/preset arms."""
        cfg = copy.deepcopy(load_config(self.config_path))
        jtype = normalize_junction_type(junction_type)
        base_j = dict(cfg.get("junction") or {})
        preset = self._load_junction_preset(jtype)

        if normalize_junction_type(str(base_j.get("type", jtype))) == jtype and base_j.get("arms"):
            try:
                junction_to_lane_configs({"junction": {**base_j, "type": jtype}})
                cfg["junction"] = {**base_j, "type": jtype}
                return cfg
            except ValueError:
                logger.warning("Base junction arms invalid for %s — using preset", jtype)

        merged = {**preset, "type": jtype}
        if not merged.get("arms") and not merged.get("movements"):
            raise ValueError(f"No junction arms configured for type {jtype}")
        junction_to_lane_configs({"junction": merged})
        cfg["junction"] = merged
        cfg.pop("lanes", None)
        return cfg

    def _is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancel_flags

    def _run_job(self, job_id: str) -> None:
        """Process each video sequentially; progress is written to Postgres."""
        job = self.get_job(job_id)
        if job is None:
            return

        self._update_job(job_id, status=JobStatus.RUNNING.value, error=None)
        try:
            base_cfg = self.build_config_for_job(job.junction_type)
        except Exception as exc:  # noqa: BLE001
            self._update_job(job_id, status=JobStatus.FAILED.value, error=str(exc))
            return

        videos = list(job.videos)
        total = max(len(videos), 1)

        for idx, video in enumerate(videos):
            if self._is_cancelled(job_id):
                break

            self._update_video(
                video.id,
                status=JobStatus.RUNNING.value,
                message="Queued for processing…",
                error=None,
            )
            self._update_job(job_id, progress=idx / total)

            try:
                cfg = copy.deepcopy(base_cfg)
                cfg.setdefault("source", {})
                cfg["source"]["type"] = "file"
                cfg["source"]["path"] = video.path
                cfg.setdefault("export", {})
                cfg["export"]["mode"] = "session"
                cfg["export"]["filename_prefix"] = f"atcc_{job_id}_{video.id}"
                cfg.setdefault("visualization", {})
                cfg["visualization"]["enabled"] = False
                cfg["visualization"]["publish_stats"] = False
                cfg.setdefault("logging", {})
                cfg["logging"]["per_frame_timing"] = False
                cfg["logging"]["progress_every_n_frames"] = 1
                cfg["logging"]["level"] = "INFO"

                from src.pipeline import Pipeline
                from src.stats_store import StatsStore

                _last_write = {"t": 0.0}

                def _on_progress(
                    info: dict[str, Any],
                    _video_id: str = video.id,
                    _idx: int = idx,
                    _total: int = total,
                ) -> None:
                    import time

                    now = time.monotonic()
                    # Throttle DB writes (~2/sec) so Neon isn't hammered
                    if now - _last_write["t"] < 0.5 and float(info.get("progress", 0)) < 0.999:
                        return
                    _last_write["t"] = now
                    vp = float(info.get("progress", 0.0))
                    self._update_video(
                        _video_id,
                        frames_done=int(info.get("frames_done", 0)),
                        frames_total=int(info.get("frames_total", 0)),
                        video_progress=vp,
                        total_events=int(info.get("total_events", 0)),
                        movement_counts=dict(info.get("movement_counts") or {}),
                        class_counts=dict(info.get("class_counts") or {}),
                        fps=float(info.get("fps", 0.0)),
                        message=str(info.get("message") or ""),
                    )
                    self._update_job(job_id, progress=(_idx + vp) / _total)

                self._update_video(
                    video.id,
                    message="Loading YOLO model (first run may download weights)…",
                )
                logger.info("Job %s | loading model for %s", job_id, video.filename)

                store = StatsStore()
                pipeline = Pipeline(cfg, stats_store=store, on_progress=_on_progress)
                with self._lock:
                    self._pipelines[job_id] = pipeline

                report = pipeline.run()
                self._update_video(
                    video.id,
                    total_events=pipeline.total_events,
                    report_path=str(report),
                    movement_counts=pipeline.counter.movement_totals(),
                    class_counts=pipeline.counter.class_totals(),
                    video_progress=1.0,
                    frames_done=max(video.frames_done, video.frames_total),
                    message=f"Done · {pipeline.total_events} vehicles",
                    status=JobStatus.COMPLETED.value,
                )
                logger.info(
                    "Job %s video %s done — %d counts → %s",
                    job_id,
                    video.filename,
                    pipeline.total_events,
                    report,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Job %s video %s failed", job_id, video.filename)
                self._update_video(
                    video.id,
                    status=JobStatus.FAILED.value,
                    error=str(exc),
                    message=f"Failed: {exc}",
                )
            finally:
                with self._lock:
                    self._pipelines.pop(job_id, None)

            self._update_job(job_id, progress=(idx + 1) / total)

        if self._is_cancelled(job_id):
            self._update_job(job_id, status=JobStatus.CANCELLED.value)
            return

        refreshed = self.get_job(job_id)
        if refreshed is None:
            return
        if all(v.status == JobStatus.FAILED.value for v in refreshed.videos):
            self._update_job(job_id, status=JobStatus.FAILED.value, progress=1.0)
        else:
            self._update_job(job_id, status=JobStatus.COMPLETED.value, progress=1.0)


MANAGER = JobManager()
