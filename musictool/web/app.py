"""FastAPI web app: upload audio, watch progress, preview and download MIDI."""

import io
import re
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..drums import adtof_available
from ..gpu import device
from ..pipeline import TranscriptionOptions, TranscriptionResult, resolve_engines, transcribe
from ..pitch import crepe_available
from ..separation import NEURAL_MODELS, neural_available

STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff", ".m4a"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
JOB_TTL_SECONDS = 2 * 60 * 60

app = FastAPI(title="MusicTool", description="AI electronic music to MIDI transcription")
executor = ThreadPoolExecutor(max_workers=1)  # models are CPU heavy; run one job at a time


@dataclass
class Job:
    id: str
    filename: str
    workdir: Path
    status: str = "queued"  # queued | running | done | error
    stage: str = "Queued"
    progress: float = 0.0
    error: str | None = None
    result: TranscriptionResult | None = None
    created: float = field(default_factory=time.time)


jobs: dict[str, Job] = {}
jobs_lock = threading.Lock()


def _cleanup_old_jobs() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    with jobs_lock:
        for job_id in [j.id for j in jobs.values() if j.created < cutoff and j.status in ("done", "error")]:
            shutil.rmtree(jobs.pop(job_id).workdir, ignore_errors=True)


def _run(job: Job, audio_path: Path, options: TranscriptionOptions) -> None:
    job.status = "running"

    def progress(stage: str, frac: float) -> None:
        job.stage, job.progress = stage, frac

    try:
        job.result = transcribe(audio_path, options, progress)
        job.status = "done"
    except Exception as exc:  # surface any failure to the UI
        job.status, job.error = "error", str(exc)
    finally:
        audio_path.unlink(missing_ok=True)


@app.get("/api/info")
def info() -> dict:
    return {
        "neural_separation": neural_available(),
        "separation_models": {k: label for k, (_, label) in NEURAL_MODELS.items()},
        "adtof": adtof_available(),
        "crepe": crepe_available(),
        "gpu": device() == "cuda",
        "defaults": asdict(resolve_engines(TranscriptionOptions())),
    }


@app.post("/api/transcribe")
async def create_job(
    file: UploadFile = File(...),
    separation: str = Form("auto"),
    drum_engine: str = Form("auto"),
    mono_engine: str = Form("auto"),
    bpm: float | None = Form(None),
    quantize: int = Form(16),
    drums: bool = Form(True),
    bass: bool = Form(True),
    melodic: bool = Form(True),
    onset_threshold: float = Form(0.5),
    frame_threshold: float = Form(0.3),
    min_note_ms: float = Form(80.0),
    drum_sensitivity: float = Form(0.5),
) -> dict:
    _cleanup_old_jobs()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Use one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    if separation not in ("auto", "hpss", *NEURAL_MODELS):
        raise HTTPException(400, f"separation must be one of: auto, hpss, {', '.join(NEURAL_MODELS)}")
    if separation in NEURAL_MODELS and not neural_available():
        raise HTTPException(400, "Neural separation is not installed on the server (run install.bat with GPU support)")
    if drum_engine not in ("auto", "adtof", "dsp"):
        raise HTTPException(400, "drum_engine must be 'auto', 'adtof' or 'dsp'")
    if drum_engine == "adtof" and not adtof_available():
        raise HTTPException(400, "ADTOF is not installed on the server")
    if mono_engine not in ("auto", "crepe", "pyin", "basic-pitch"):
        raise HTTPException(400, "mono_engine must be 'auto', 'crepe', 'pyin' or 'basic-pitch'")
    if mono_engine == "crepe" and not crepe_available():
        raise HTTPException(400, "CREPE is not installed on the server")

    job_id = uuid.uuid4().hex
    workdir = Path(tempfile.mkdtemp(prefix=f"musictool-{job_id[:8]}-"))
    audio_path = workdir / f"input{suffix}"
    size = 0
    with audio_path.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                shutil.rmtree(workdir, ignore_errors=True)
                raise HTTPException(413, "File too large (max 200 MB)")
            out.write(chunk)

    options = TranscriptionOptions(
        separation=separation,
        bpm=bpm or None,
        quantize=quantize or None,
        drums=drums,
        bass=bass,
        melodic=melodic,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        min_note_ms=min_note_ms,
        drum_sensitivity=drum_sensitivity,
        drum_engine=drum_engine,
        mono_engine=mono_engine,
        stems_dir=workdir / "stems",
    )
    safe_name = re.sub(r"[^\w\-. ]", "_", Path(file.filename).stem)[:100] or "transcription"
    job = Job(id=job_id, filename=safe_name, workdir=workdir)
    with jobs_lock:
        jobs[job_id] = job
    executor.submit(_run, job, audio_path, options)
    return {"id": job_id}


def _get_job(job_id: str) -> Job:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = _get_job(job_id)
    body = {
        "id": job.id,
        "filename": job.filename,
        "status": job.status,
        "stage": job.stage,
        "progress": round(job.progress, 3),
        "error": job.error,
    }
    if job.result is not None:
        body["result"] = job.result.to_dict()
        body["stems"] = sorted(p.stem.split("_", 1)[1] for p in (job.workdir / "stems").glob("*.wav"))
    return body


@app.get("/api/jobs/{job_id}/stems/{stem}")
def download_stem(job_id: str, stem: str) -> FileResponse:
    job = _get_job(job_id)
    path = job.workdir / "stems" / f"input_{stem}.wav"
    if not re.fullmatch(r"[a-z]+", stem) or not path.exists():
        raise HTTPException(404, f"No stem named {stem!r}")
    return FileResponse(path, media_type="audio/wav", filename=f"{job.filename}_{stem}.wav")


@app.get("/api/jobs/{job_id}/midi")
def download_midi(job_id: str, track: str | None = None) -> StreamingResponse:
    job = _get_job(job_id)
    if job.result is None:
        raise HTTPException(409, "Transcription not finished")
    names = None
    suffix = ""
    if track:
        if track not in [t.name for t in job.result.tracks]:
            raise HTTPException(404, f"No track named {track!r}")
        names, suffix = [track], f"_{track.lower()}"
    buf = io.BytesIO()
    job.result.to_midi(names).write(buf)
    buf.seek(0)
    filename = f"{job.filename}{suffix}.mid"
    return StreamingResponse(
        buf,
        media_type="audio/midi",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
