"""In-memory job registry for uploaded video processing."""

from __future__ import annotations

import copy
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from src.config_loader import load_config, project_root, resolve_path
from src.junction import (
    describe_junction_types,
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


@dataclass
class VideoJobItem:
    """One uploaded video inside a batch job."""

    video_id: str
    filename: str
    path: str
    status: JobStatus = JobStatus.QUEUED
    total_events: int = 0
    report_path: str | None = None
    annotated_path: str | None = None
    error: str | None = None
    lane_counts: dict[str, int] = field(default_factory=dict)
    class_counts: dict[str, int] = field(default_factory=dict)
    movement_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class BatchJob:
    """A user upload batch (one or many videos)."""

    job_id: str
    created_at: str
    status: JobStatus = JobStatus.QUEUED
    videos: list[VideoJobItem] = field(default_factory=list)
    error: str | None = None
    progress: float = 0.0
    junction_type: str = "four_way"


class JobManager:
    """Create upload jobs and process them with the ATCC pipeline."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        """Load base camera config used for every upload job."""
        root = project_root()
        self.config_path = resolve_path(config_path or "config/camera_config.yaml")
        self.upload_root = root / "data" / "uploads"
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, BatchJob] = {}
        self._lock = threading.Lock()
        self._pipelines: dict[str, Any] = {}

    def create_job(
        self,
        saved_files: list[tuple[str, Path]],
        junction_type: str = "four_way",
    ) -> BatchJob:
        """Register a new batch from saved upload paths."""
        jtype = normalize_junction_type(junction_type)
        job_id = uuid.uuid4().hex[:12]
        videos = [
            VideoJobItem(
                video_id=uuid.uuid4().hex[:8],
                filename=name,
                path=str(path),
            )
            for name, path in saved_files
        ]
        job = BatchJob(
            job_id=job_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            status=JobStatus.QUEUED,
            videos=videos,
            junction_type=jtype,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> BatchJob | None:
        """Return a job by id, or None."""
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[BatchJob]:
        """Return all jobs newest-first."""
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def start_job(self, job_id: str) -> BatchJob:
        """Spawn a background thread to process all videos in the job."""
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status == JobStatus.RUNNING:
            return job

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id,),
            name=f"atcc-job-{job_id}",
            daemon=True,
        )
        thread.start()
        return job

    def cancel_job(self, job_id: str) -> BatchJob:
        """Request stop on the active pipeline for a job."""
        with self._lock:
            job = self._jobs.get(job_id)
            pipe = self._pipelines.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if pipe is not None:
            pipe.request_stop()
        job.status = JobStatus.CANCELLED
        return job

    def job_to_dict(self, job: BatchJob) -> dict[str, Any]:
        """Serialize a job for JSON responses."""
        return {
            "job_id": job.job_id,
            "created_at": job.created_at,
            "status": job.status.value,
            "progress": job.progress,
            "error": job.error,
            "junction_type": job.junction_type,
            "expected_movements": expected_movements(job.junction_type),
            "videos": [
                {
                    "video_id": v.video_id,
                    "filename": v.filename,
                    "status": v.status.value,
                    "total_events": v.total_events,
                    "report_path": v.report_path,
                    "annotated_path": v.annotated_path,
                    "error": v.error,
                    "lane_counts": v.lane_counts,
                    "movement_counts": v.movement_counts or v.lane_counts,
                    "class_counts": v.class_counts,
                    "report_url": (
                        f"/api/jobs/{job.job_id}/videos/{v.video_id}/report"
                        if v.report_path
                        else None
                    ),
                }
                for v in job.videos
            ],
        }

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

        # Prefer calibrated arms from camera_config when type already matches.
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
        # Remove legacy lanes so junction takes effect exclusively
        cfg.pop("lanes", None)
        return cfg

    def _run_job(self, job_id: str) -> None:
        """Process each video sequentially."""
        job = self.get_job(job_id)
        if job is None:
            return

        job.status = JobStatus.RUNNING
        try:
            base_cfg = self.build_config_for_job(job.junction_type)
        except Exception as exc:  # noqa: BLE001
            job.status = JobStatus.FAILED
            job.error = str(exc)
            return

        total = max(len(job.videos), 1)

        for idx, video in enumerate(job.videos):
            if job.status == JobStatus.CANCELLED:
                break

            video.status = JobStatus.RUNNING
            job.progress = idx / total
            try:
                cfg = copy.deepcopy(base_cfg)
                cfg.setdefault("source", {})
                cfg["source"]["type"] = "file"
                cfg["source"]["path"] = video.path
                cfg.setdefault("export", {})
                cfg["export"]["mode"] = "session"
                cfg["export"]["filename_prefix"] = f"atcc_{job_id}_{video.video_id}"
                cfg.setdefault("visualization", {})
                cfg["visualization"]["enabled"] = True
                cfg["visualization"]["publish_stats"] = False
                cfg.setdefault("logging", {})
                cfg["logging"]["per_frame_timing"] = False

                from src.pipeline import Pipeline
                from src.stats_store import StatsStore

                store = StatsStore()
                pipeline = Pipeline(cfg, stats_store=store)
                with self._lock:
                    self._pipelines[job_id] = pipeline

                report = pipeline.run()
                video.total_events = pipeline.total_events
                video.report_path = str(report)
                video.movement_counts = pipeline.counter.movement_totals()
                video.lane_counts = dict(video.movement_counts)
                video.class_counts = pipeline.counter.class_totals()
                vis_dir = resolve_path(str(cfg["visualization"].get("output_dir", "outputs/annotated")))
                annotated = sorted(vis_dir.glob(f"annotated_{pipeline.exporter.session_id}.*"))
                video.annotated_path = str(annotated[-1]) if annotated else None
                video.status = JobStatus.COMPLETED
                logger.info(
                    "Job %s video %s done — %d counts → %s",
                    job_id,
                    video.filename,
                    video.total_events,
                    video.report_path,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Job %s video %s failed", job_id, video.filename)
                video.status = JobStatus.FAILED
                video.error = str(exc)
            finally:
                with self._lock:
                    self._pipelines.pop(job_id, None)

            job.progress = (idx + 1) / total

        if job.status != JobStatus.CANCELLED:
            if all(v.status == JobStatus.FAILED for v in job.videos):
                job.status = JobStatus.FAILED
            else:
                job.status = JobStatus.COMPLETED
        job.progress = 1.0 if job.status != JobStatus.CANCELLED else job.progress


MANAGER = JobManager()
