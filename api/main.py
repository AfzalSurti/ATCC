"""FastAPI backend for the ATCC React dashboard.

Auth + Neon Postgres so jobs keep running after the browser tab closes.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from api.auth import (  # noqa: E402
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from api.db import get_db, init_db  # noqa: E402
from api.jobs import MANAGER  # noqa: E402
from api.models import User  # noqa: E402
from src.junction import describe_junction_types, expected_movements  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("atcc.api")

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _cors_origins() -> list[str]:
    """Local defaults + optional CORS_ORIGINS env (comma-separated).

    Trailing slashes are stripped — `https://app.vercel.app/` must match
    the browser Origin header without a slash.
    """
    defaults = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://atcc-ten.vercel.app",
    ]
    extra = os.getenv("CORS_ORIGINS", "").strip()
    if not extra:
        return defaults
    parsed = [o.strip().rstrip("/") for o in extra.split(",") if o.strip()]
    return list(dict.fromkeys(defaults + parsed))


app = FastAPI(title="ATCC API", version="0.4.0")

_origins = _cors_origins()
# Bearer JWT auth does not need cookie credentials — allow * when requested,
# and always allow Vercel preview/prod hosts via regex.
_allow_all = "*" in _origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allow_all else _origins,
    allow_origin_regex=r"https://.*\.vercel\.app" if not _allow_all else None,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
logger.info("CORS origins: %s (allow_all=%s)", _origins, _allow_all)


class SignupBody(BaseModel):
    """Signup request."""

    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class LoginBody(BaseModel):
    """Login request."""

    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    """Token + user payload."""

    access_token: str
    token_type: str = "bearer"
    email: str
    user_id: int


def _safe_name(name: str) -> str:
    """Sanitize an upload filename."""
    base = Path(name).name
    base = re.sub(r"[^\w.\- ]+", "_", base).strip() or "video.mp4"
    return base


@app.on_event("startup")
def on_startup() -> None:
    """Connect to Neon, create tables, clean up interrupted jobs."""
    try:
        init_db()
        n = MANAGER.mark_stale_running_as_failed()
        if n:
            logger.warning("Marked %d interrupted job(s) as failed after restart", n)
        logger.info("Auth + Postgres ready")
    except Exception:
        logger.exception("Database init failed — set DATABASE_URL")
        raise


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/api/junction-types")
def junction_types() -> dict:
    """Describe 2-way / 3-way / 4-way movement models for the UI."""
    return {"types": describe_junction_types()}


@app.post("/api/auth/signup", response_model=AuthResponse)
def signup(body: SignupBody, db: Session = Depends(get_db)) -> AuthResponse:
    """Create an account and return a JWT."""
    email = body.email.lower().strip()
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=email, password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(user.id, user.email)
    return AuthResponse(access_token=token, email=user.email, user_id=user.id)


@app.post("/api/auth/login", response_model=AuthResponse)
def login(body: LoginBody, db: Session = Depends(get_db)) -> AuthResponse:
    """Log in and return a JWT."""
    email = body.email.lower().strip()
    user = db.query(User).filter(User.email == email).first()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(user.id, user.email)
    return AuthResponse(access_token=token, email=user.email, user_id=user.id)


@app.get("/api/auth/me")
def me(user: User = Depends(get_current_user)) -> dict:
    """Return the current authenticated user."""
    return {"user_id": user.id, "email": user.email}


@app.post("/api/upload")
async def upload_videos(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    junction_type: str = Form("four_way"),
    user: User = Depends(get_current_user),
) -> dict:
    """Accept video uploads for the signed-in user and start processing on the server."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

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
            if dest.exists():
                dest = staging / f"{dest.stem}_{len(saved)}{dest.suffix}"
            # Stream to disk — avoid loading whole video into RAM (Render OOM → bare 500 / fake CORS)
            size = 0
            with dest.open("wb") as out:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    out.write(chunk)
            if size == 0:
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail=f"Empty file: {original}")
            saved.append((original, dest))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        job = MANAGER.create_job(user.id, saved, junction_type=junction_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("create_job failed (check DATABASE_URL)")
        raise HTTPException(
            status_code=500,
            detail=f"Could not create job (database?): {exc}",
        ) from exc

    final_dir = MANAGER.upload_root / job.id
    if staging.resolve() != final_dir.resolve():
        if final_dir.exists():
            raise HTTPException(status_code=500, detail="Job upload folder already exists")
        staging.rename(final_dir)
        for video in job.videos:
            new_path = str(final_dir / Path(video.path).name)
            MANAGER.update_video_path(video.id, new_path)
            video.path = new_path

    # Start AFTER the HTTP response is sent — loading YOLO during the request can OOM
    # Render free tier and return a bare 500 with no CORS headers (looks like a CORS error).
    background_tasks.add_task(MANAGER.start_job, job.id)
    logger.info(
        "Queued job %s for user %s (%s, %d movements) with %d video(s)",
        job.id,
        user.email,
        job.junction_type,
        expected_movements(job.junction_type),
        len(job.videos),
    )
    refreshed = MANAGER.get_job(job.id, user_id=user.id)
    return MANAGER.job_to_dict(refreshed or job)


@app.get("/api/jobs")
def list_jobs(user: User = Depends(get_current_user)) -> dict:
    """List jobs for the signed-in user (survive tab close)."""
    return {"jobs": [MANAGER.job_to_dict(j) for j in MANAGER.list_jobs(user.id)]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, user: User = Depends(get_current_user)) -> dict:
    """Get one job owned by the signed-in user."""
    job = MANAGER.get_job(job_id, user_id=user.id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return MANAGER.job_to_dict(job)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, user: User = Depends(get_current_user)) -> dict:
    """Cancel a running job owned by the user."""
    try:
        job = MANAGER.cancel_job(job_id, user_id=user.id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return MANAGER.job_to_dict(job)


@app.get("/api/jobs/{job_id}/videos/{video_id}/report")
def download_report(
    job_id: str,
    video_id: str,
    user: User = Depends(get_current_user),
) -> FileResponse:
    """Download the Excel report for one processed video."""
    job = MANAGER.get_job(job_id, user_id=user.id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    video = next((v for v in job.videos if v.id == video_id), None)
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


_dist = ROOT / "frontend" / "dist"
if os.getenv("SERVE_FRONTEND", "").strip() in {"1", "true", "yes"} and _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")


def create_app() -> FastAPI:
    """App factory for uvicorn."""
    return app
