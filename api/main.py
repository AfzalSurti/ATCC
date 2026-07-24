"""FastAPI backend for the ATCC React dashboard.

Endpoints:
  POST /api/upload          — upload one or many videos, start processing
  GET  /api/jobs            — list jobs
  GET  /api/jobs/{id}       — job status / results
  POST /api/jobs/{id}/cancel
  GET  /api/jobs/{id}/videos/{vid}/report — download Excel
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.jobs import MANAGER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("atcc.api")

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

app = FastAPI(title="ATCC API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe_name(name: str) -> str:
    """Sanitize an upload filename."""
    base = Path(name).name
    base = re.sub(r"[^\w.\- ]+", "_", base).strip() or "video.mp4"
    return base


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_videos(files: list[UploadFile] = File(...)) -> dict:
    """Accept one or more video uploads and start a processing job."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    job_dir = MANAGER.upload_root / f"pending_{id(files)}"
    # Create job id folder after we know job id — save under temp then move.
    saved: list[tuple[str, Path]] = []
    staging = MANAGER.upload_root / f"tmp_{Path(files[0].filename or 'x').stem}_{len(files)}"
    staging.mkdir(parents=True, exist_ok=True)

    try:
        for upload in files:
            original = upload.filename or "video.mp4"
            ext = Path(original).suffix.lower()
            if ext not in ALLOWED_EXT:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type '{ext}' for {original}. Allowed: {sorted(ALLOWED_EXT)}",
                )
            safe = _safe_name(original)
            dest = staging / safe
            # Avoid overwrite collisions within the same batch
            if dest.exists():
                dest = staging / f"{dest.stem}_{len(saved)}{dest.suffix}"
            data = await upload.read()
            if not data:
                raise HTTPException(status_code=400, detail=f"Empty file: {original}")
            dest.write_bytes(data)
            saved.append((original, dest))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    job = MANAGER.create_job(saved)
    # Move staging folder under job id for clarity
    final_dir = MANAGER.upload_root / job.job_id
    if staging.resolve() != final_dir.resolve():
        staging.rename(final_dir)
        for video in job.videos:
            video.path = str(final_dir / Path(video.path).name)

    MANAGER.start_job(job.job_id)
    logger.info("Started job %s with %d video(s)", job.job_id, len(job.videos))
    return MANAGER.job_to_dict(job)


@app.get("/api/jobs")
def list_jobs() -> dict:
    """List all processing jobs."""
    return {"jobs": [MANAGER.job_to_dict(j) for j in MANAGER.list_jobs()]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """Get one job's status and per-video results."""
    job = MANAGER.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return MANAGER.job_to_dict(job)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """Cancel a running job."""
    try:
        job = MANAGER.cancel_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return MANAGER.job_to_dict(job)


@app.get("/api/jobs/{job_id}/videos/{video_id}/report")
def download_report(job_id: str, video_id: str) -> FileResponse:
    """Download the Excel report for one processed video."""
    job = MANAGER.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    video = next((v for v in job.videos if v.video_id == video_id), None)
    if video is None or not video.report_path:
        raise HTTPException(status_code=404, detail="Report not ready")
    path = Path(video.report_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Report file missing")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=path.name,
    )


# Optional: serve built React app from frontend/dist when present
_dist = ROOT / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")


def create_app() -> FastAPI:
    """App factory for uvicorn."""
    return app
