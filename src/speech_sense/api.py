"""FastAPI service — REST wrapper around SpeakerVerifier.

Endpoints:

    GET  /                      -> the static web dashboard
    GET  /healthz               -> liveness + backend info (never authenticated)
    GET  /readyz                -> readiness: is the encoder actually loaded?
    GET  /prompts               -> recording prompt sentences
    GET  /speakers              -> list enrolled speakers
    GET  /analytics             -> dashboard analytics data
    GET  /employees             -> employee audio directory listing
    POST /enroll                -> multipart upload: name + one-or-more clips
    POST /identify              -> multipart upload: one clip -> best speaker
    POST /verify                -> multipart upload: one clip + claimed name
    DELETE /speakers/{name}     -> remove a speaker
    POST /reload                -> reload the database from disk
    POST /employees/enroll-all  -> batch enroll the employee audio folder
    POST /batch/identify        -> identify many uploads in one request
    POST /batch/enroll          -> enroll one speaker from many uploads

Operational design, in order of how often it bites:

* **Nothing CPU-bound runs on the event loop.** Speaker embedding takes tens to
  hundreds of milliseconds. Doing that inside an `async def` handler stalls
  *every* connection for its duration, so all inference is dispatched with
  `run_in_threadpool`.
* **Every input is bounded.** Per-file byte cap, per-request file cap, and a
  decoded-duration cap (bytes alone don't bound the work — compressed audio
  expands).
* **Startup failure degrades, it doesn't crash-loop.** If the encoder can't be
  built, the process still serves `/healthz`, reports `not ready` on `/readyz`,
  and answers 503 with the real reason everywhere else.
* **No traceback ever reaches a client.** A global handler logs the exception
  with a request id and returns that id instead.
* **Optional API key.** Setting `SPEECH_SENSE_API_KEY` requires an `X-API-Key`
  header on everything that mutates state, costs CPU, or discloses the roster.

The verifier is constructed once at app startup so the encoder isn't reloaded
per request. The database is persisted after every mutation so a restart
doesn't lose enrollments.
"""

from __future__ import annotations

import hmac
import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Callable, List, Optional

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .audio import load_audio, normalize, resample, sanitize
from .config import DEFAULT, Config
from .database import SpeakerDatabase, clean_speaker_name
from .embedding import SpeakerEncoder
from .verifier import SpeakerVerifier

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent / "web"
UPLOAD_CHUNK = 64 * 1024
MAX_RECENT_EVENTS = 50

# Temp files whose deletion lost a race with the decoder. librosa's audioread
# fallback can still hold the handle when it raises, and on Windows deleting an
# open file fails outright — so the unlink is retried on the next decode instead
# of leaking a file per bad upload for the lifetime of the process.
_PENDING_UNLINK: set[str] = set()


def _discard_temp(path: str) -> None:
    """Delete a scratch file, deferring to the next attempt if it is still locked."""
    try:
        Path(path).unlink(missing_ok=True)
        _PENDING_UNLINK.discard(path)
    except OSError as exc:
        log.debug("temp file %s not yet removable (%s); will retry", path, exc)
        _PENDING_UNLINK.add(path)


def _sweep_temp_files() -> None:
    for path in list(_PENDING_UNLINK):
        _discard_temp(path)

# Recording prompt sentences — phonetically balanced for voice enrollment.
RECORDING_PROMPTS = [
    {
        "id": 1,
        "category": "Security",
        "title": "Security Passphrase",
        "text": "My voice is my password, verify me for secure system access.",
        "duration_hint": "4-6 seconds",
    },
    {
        "id": 2,
        "category": "Phonetic",
        "title": "Phonetically Balanced",
        "text": "The quick brown fox jumps over the lazy dog near the quiet river bank.",
        "duration_hint": "5-7 seconds",
    },
    {
        "id": 3,
        "category": "Identity",
        "title": "Employee Identification",
        "text": "Authorization request for employee biometric voice recognition system.",
        "duration_hint": "4-6 seconds",
    },
    {
        "id": 4,
        "category": "Biometrics",
        "title": "Voice Pattern Sample",
        "text": "Voice biometrics provide robust identity authentication across all secure sessions.",
        "duration_hint": "5-7 seconds",
    },
    {
        "id": 5,
        "category": "Access",
        "title": "Access Verification",
        "text": "Access granted to authorized personnel only after voice pattern matching.",
        "duration_hint": "4-6 seconds",
    },
    {
        "id": 6,
        "category": "General",
        "title": "Natural Speech Sample",
        "text": "Please speak naturally and clearly to capture your unique vocal characteristics.",
        "duration_hint": "4-6 seconds",
    },
]


def configure_logging(level: str = DEFAULT.log_level) -> None:
    """Attach a console handler once, without stomping on an existing setup."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        )
    else:
        root.setLevel(getattr(logging, level.upper(), logging.INFO))


def create_app(
    database_path: str | Path = DEFAULT.database_path,
    backend: str = "auto",
    use_vad: bool = True,
    config: Config = DEFAULT,
) -> FastAPI:
    """Application factory — tests instantiate their own app with a temp DB path."""
    configure_logging(config.log_level)

    state: dict = {
        "database_path": str(database_path),
        "backend": backend,
        "use_vad": use_vad,
        "config": config,
        "verifier": None,
        "startup_error": "",
        "batch_enroll_running": False,
        "analytics": {
            "total_identifications": 0,
            "total_enrollments": 0,
            "total_verifications": 0,
            "successful_matches": 0,
            "failed_matches": 0,
            "confidence_sum": 0.0,
            "recent_events": [],
            "start_time": time.time(),
        },
    }
    # Guards the analytics counters and the batch-enroll flag. The database has
    # its own lock; this one exists because `dict[key] += 1` is a read-modify-
    # write that interleaves badly across thread-pool workers.
    stats_lock = Lock()

    def _build_verifier() -> SpeakerVerifier:
        encoder = SpeakerEncoder.load(state["backend"])  # type: ignore[arg-type]
        database = SpeakerDatabase.load(state["database_path"])
        log.info(
            "verifier ready: backend=%s, %d enrolled speaker(s), database=%s",
            encoder.name, len(database), state["database_path"],
        )
        return SpeakerVerifier(
            encoder=encoder, database=database,
            use_vad=state["use_vad"], config=state["config"],
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            state["verifier"] = await run_in_threadpool(_build_verifier)
            state["startup_error"] = ""
        except Exception as exc:  # noqa: BLE001
            # A model that won't load must not turn into a crash loop: come up
            # degraded so /healthz answers, an operator can read the reason,
            # and an orchestrator's restart backoff isn't the only diagnostic.
            state["startup_error"] = f"{type(exc).__name__}: {exc}"
            log.exception("startup failed; serving in degraded mode")
        yield
        state["verifier"] = None

    app = FastAPI(
        title="SpeechSense",
        version="1.0.0",
        description="Robust Voice Recognition — speaker verification dashboard by Ishtiaq Ahmed.",
        lifespan=lifespan,
    )

    if config.allowed_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.allowed_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["X-API-Key", "Content-Type"],
        )

    # ──────────────────────────────────────────────────────────────────────
    # Cross-cutting middleware
    # ──────────────────────────────────────────────────────────────────────

    @app.middleware("http")
    async def request_guard(request: Request, call_next: Callable):
        """Access log, hard body-size ceiling, and a traceback firewall."""
        request_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()

        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                # One request may carry several files plus form fields, so the
                # whole-body ceiling is generous relative to the per-file one.
                ceiling = config.max_upload_bytes * config.max_files_per_request + (1 << 20)
                if int(declared) > ceiling:
                    return JSONResponse(
                        {"detail": f"request body exceeds {ceiling} bytes", "request_id": request_id},
                        status_code=413,
                    )
            except ValueError:
                return JSONResponse(
                    {"detail": "invalid Content-Length header", "request_id": request_id},
                    status_code=400,
                )

        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001
            # Anything reaching here is a bug. Log it in full server-side and
            # hand the client only a correlation id — stack traces disclose
            # paths, versions, and sometimes data.
            log.exception("unhandled error [%s] %s %s", request_id, request.method, request.url.path)
            return JSONResponse(
                {
                    "detail": "internal server error",
                    "request_id": request_id,
                },
                status_code=500,
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        log.info(
            "%s %s -> %d (%.1f ms) [%s]",
            request.method, request.url.path, response.status_code, elapsed_ms, request_id,
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    # ──────────────────────────────────────────────────────────────────────
    # Dependencies
    # ──────────────────────────────────────────────────────────────────────

    def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
        """Gate on `SPEECH_SENSE_API_KEY` when one is configured.

        No key configured = open service, which is the right default for the
        local demo. Comparison is constant-time so the key can't be recovered
        one byte at a time from response timings.
        """
        if not config.auth_enabled:
            return
        if not x_api_key or not hmac.compare_digest(x_api_key, config.api_key):
            raise HTTPException(status_code=401, detail="missing or invalid X-API-Key header")

    def get_verifier() -> SpeakerVerifier:
        verifier = state.get("verifier")
        if verifier is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"speaker engine unavailable: {state['startup_error']}"
                    if state["startup_error"]
                    else "speaker engine is still starting up"
                ),
            )
        return verifier

    Verifier = Depends(get_verifier)
    Auth = Depends(require_api_key)

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _persist(verifier: SpeakerVerifier) -> None:
        try:
            verifier.database.save(state["database_path"])
        except OSError as exc:
            # The in-memory enrollment succeeded; only durability failed. Say
            # so precisely instead of implying the whole operation was lost.
            log.error("could not persist database to %s: %s", state["database_path"], exc)
            raise HTTPException(
                status_code=507,
                detail=f"speaker was accepted but could not be saved to disk: {exc}",
            ) from exc

    def _log_event(event_type: str, details: str, success: bool = True) -> None:
        with stats_lock:
            events = state["analytics"]["recent_events"]
            events.insert(0, {
                "type": event_type,
                "details": details,
                "success": success,
                "timestamp": time.time(),
            })
            del events[MAX_RECENT_EVENTS:]

    def _bump(field: str, amount: float = 1) -> None:
        with stats_lock:
            state["analytics"][field] += amount

    def _employees_root() -> Path:
        """Resolved employee-corpus directory.

        Read from the environment per call rather than captured at import, so
        an operator (and the test suite) can repoint it without a restart.
        """
        raw = os.environ.get("SPEECH_SENSE_EMPLOYEES_AUDIO") or config.employees_audio
        return Path(raw).expanduser()

    async def _read_upload(upload: UploadFile) -> bytes:
        """Read one upload, refusing to buffer more than the configured cap."""
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await upload.read(UPLOAD_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > config.max_upload_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"{upload.filename or 'file'} exceeds the "
                        f"{config.max_upload_bytes} byte limit"
                    ),
                )
            chunks.append(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail=f"{upload.filename or 'file'} is empty")
        return b"".join(chunks)

    def _decode(raw: bytes, filename: str | None) -> np.ndarray:
        """Decode uploaded bytes to mono float32 at the configured sample rate.

        Blocking: always call via `run_in_threadpool`.
        """
        _sweep_temp_files()
        try:
            audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
        except Exception:  # noqa: BLE001 — libsndfile can't do MP3/WebM
            import tempfile

            suffix = Path(filename or "clip").suffix or ".dat"
            tmp_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(raw)
                    tmp_path = tmp.name
                audio = load_audio(tmp_path, target_sr=config.sample_rate)
                sr = config.sample_rate
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"could not decode audio ({filename or 'upload'}): {exc}. "
                        f"Supported without extra tools: WAV, FLAC, OGG, AIFF."
                    ),
                ) from exc
            finally:
                # Cleanup must never mask the decode error above — a locked
                # temp file used to turn a clean 400 into a 500.
                if tmp_path is not None:
                    _discard_temp(tmp_path)

        audio = sanitize(audio)
        if audio.size == 0:
            raise HTTPException(status_code=400, detail="audio is empty")
        if sr <= 0:
            raise HTTPException(status_code=400, detail=f"audio declares an invalid sample rate ({sr})")
        if audio.size / float(sr) > config.max_audio_seconds:
            # A few MiB of compressed audio can decode to hours of PCM; the
            # byte cap alone doesn't bound the work this would create.
            raise HTTPException(
                status_code=413,
                detail=(
                    f"audio is longer than the {config.max_audio_seconds:g}s limit "
                    f"({audio.size / float(sr):.0f}s decoded)"
                ),
            )
        if sr != config.sample_rate:
            audio = resample(audio, int(sr), config.sample_rate)
        return normalize(audio)

    async def _read_clip(upload: UploadFile) -> np.ndarray:
        raw = await _read_upload(upload)
        return await run_in_threadpool(_decode, raw, upload.filename)

    def _check_file_count(files: List[UploadFile]) -> None:
        if not files:
            raise HTTPException(status_code=400, detail="at least one audio file is required")
        if len(files) > config.max_files_per_request:
            raise HTTPException(
                status_code=413,
                detail=f"at most {config.max_files_per_request} files per request",
            )

    def _clean_name(raw: str) -> str:
        try:
            return clean_speaker_name(raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ──────────────────────────────────────────────────────────────────────
    # Health / Status  (deliberately unauthenticated — probes must always work)
    # ──────────────────────────────────────────────────────────────────────

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        verifier = state.get("verifier")
        if verifier is None:
            return JSONResponse(
                {
                    "status": "degraded",
                    "detail": state["startup_error"] or "starting up",
                },
                status_code=503,
            )
        return JSONResponse({
            "status": "ok",
            "backend": verifier.encoder.name,
            "sample_rate": verifier.config.sample_rate,
            "n_speakers": len(verifier.database),
            "similarity_threshold": verifier.config.similarity_threshold,
            "use_vad": verifier.use_vad,
            "auth_required": config.auth_enabled,
        })

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        ready = state.get("verifier") is not None
        return JSONResponse(
            {"ready": ready, "detail": state["startup_error"] or "ok"},
            status_code=200 if ready else 503,
        )

    @app.get("/prompts")
    def get_prompts() -> dict:
        return {"prompts": RECORDING_PROMPTS, "count": len(RECORDING_PROMPTS)}

    # ──────────────────────────────────────────────────────────────────────
    # Analytics Dashboard Data
    # ──────────────────────────────────────────────────────────────────────

    @app.get("/analytics", dependencies=[Auth])
    def get_analytics(verifier: SpeakerVerifier = Verifier) -> dict:
        with stats_lock:
            a = dict(state["analytics"])
            a["recent_events"] = list(a["recent_events"])[:20]
        total_ops = a["total_identifications"] + a["total_verifications"]
        return {
            "speakers_enrolled": len(verifier.database),
            "speaker_names": verifier.database.names(),
            "total_identifications": a["total_identifications"],
            "total_enrollments": a["total_enrollments"],
            "total_verifications": a["total_verifications"],
            "successful_matches": a["successful_matches"],
            "failed_matches": a["failed_matches"],
            "match_rate": round(a["successful_matches"] / total_ops * 100, 1) if total_ops else 0.0,
            "avg_confidence": round(a["confidence_sum"] / total_ops, 4) if total_ops else 0.0,
            "total_operations": total_ops + a["total_enrollments"],
            "recent_events": a["recent_events"],
            "uptime_seconds": round(time.time() - a["start_time"], 0),
            "server_time": time.time(),
            "backend": verifier.encoder.name,
            "config": {
                "sample_rate": verifier.config.sample_rate,
                "similarity_threshold": verifier.config.similarity_threshold,
                "known_speaker_margin": verifier.config.known_speaker_margin,
                "vad_enabled": verifier.use_vad,
            },
        }

    # ──────────────────────────────────────────────────────────────────────
    # Speaker Management
    # ──────────────────────────────────────────────────────────────────────

    @app.get("/speakers", dependencies=[Auth])
    def list_speakers(verifier: SpeakerVerifier = Verifier) -> dict:
        names = verifier.database.names()
        return {"speakers": names, "count": len(names)}

    @app.delete("/speakers/{name}", dependencies=[Auth])
    def delete_speaker(name: str, verifier: SpeakerVerifier = Verifier) -> dict:
        name = _clean_name(name)
        if not verifier.database.remove(name):
            raise HTTPException(status_code=404, detail=f"speaker {name!r} not found")
        _persist(verifier)
        _log_event("delete", f"Removed speaker: {name}")
        return {"deleted": name, "remaining": verifier.database.names()}

    @app.post("/reload", dependencies=[Auth])
    def reload_db(verifier: SpeakerVerifier = Verifier) -> dict:
        verifier.database = SpeakerDatabase.load(state["database_path"])
        _log_event("reload", f"Reloaded {len(verifier.database)} speaker(s) from disk")
        return {"reloaded": True, "n_speakers": len(verifier.database)}

    # ──────────────────────────────────────────────────────────────────────
    # Enrollment
    # ──────────────────────────────────────────────────────────────────────

    @app.post("/enroll", dependencies=[Auth])
    async def enroll(
        name: str = Form(...),
        files: List[UploadFile] = File(...),
        verifier: SpeakerVerifier = Verifier,
    ) -> dict:
        name = _clean_name(name)
        _check_file_count(files)
        clips = [await _read_clip(f) for f in files]
        try:
            # Embedding is CPU-bound; on the event loop it would freeze every
            # other connection for the duration.
            await run_in_threadpool(verifier.enroll_from_audio, name, clips)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _persist(verifier)
        _bump("total_enrollments")
        _log_event("enroll", f"Enrolled {name} with {len(clips)} clips")
        return {
            "enrolled": name,
            "clips": len(clips),
            "backend": verifier.encoder.name,
            "speakers": verifier.database.names(),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Identification / Verification
    # ──────────────────────────────────────────────────────────────────────

    @app.post("/identify", dependencies=[Auth])
    async def identify(
        file: UploadFile = File(...),
        verifier: SpeakerVerifier = Verifier,
    ) -> dict:
        audio = await _read_clip(file)
        result = await run_in_threadpool(verifier.identify, audio)
        with stats_lock:
            a = state["analytics"]
            a["total_identifications"] += 1
            a["confidence_sum"] += result.confidence
            a["successful_matches" if result.is_known else "failed_matches"] += 1
        if result.is_known:
            _log_event("identify", f"Matched: {result.speaker} ({result.confidence:.0%})")
        else:
            _log_event("identify", f"No match found ({result.reason})", success=False)
        return result.to_dict()

    @app.post("/verify", dependencies=[Auth])
    async def verify(
        name: str = Form(...),
        file: UploadFile = File(...),
        verifier: SpeakerVerifier = Verifier,
    ) -> dict:
        name = _clean_name(name)
        audio = await _read_clip(file)
        result = await run_in_threadpool(verifier.verify, audio, name)
        with stats_lock:
            a = state["analytics"]
            a["total_verifications"] += 1
            a["confidence_sum"] += result.confidence
            a["successful_matches" if result.is_known else "failed_matches"] += 1
        if result.is_known:
            _log_event("verify", f"Verified: {name}")
        else:
            _log_event("verify", f"Verification failed for {name} ({result.reason})", success=False)
        return result.to_dict()

    # ──────────────────────────────────────────────────────────────────────
    # Employee Audio Corpora
    # ──────────────────────────────────────────────────────────────────────

    @app.get("/employees", dependencies=[Auth])
    async def list_employees() -> dict:
        from .batch import iter_audio_files

        root = _employees_root()

        def scan() -> dict:
            if not root.is_dir():
                return {"root": str(root), "exists": False, "employees": []}
            employees = [
                {"name": sp.name, "clips": sum(1 for _ in iter_audio_files(sp))}
                for sp in sorted(p for p in root.iterdir() if p.is_dir())
            ]
            return {"root": str(root), "exists": True, "employees": employees}

        # Walking a corpus directory is disk-bound and can be slow on a
        # network mount — keep it off the event loop.
        return await run_in_threadpool(scan)

    @app.post("/employees/enroll-all", dependencies=[Auth])
    async def enroll_all_employees(
        workers: Optional[int] = Query(default=None, ge=1, le=64),
        verifier: SpeakerVerifier = Verifier,
    ) -> dict:
        from .batch import batch_enroll_directory

        root = _employees_root()
        if not root.is_dir():
            raise HTTPException(
                status_code=404,
                detail=f"employees audio directory not found at {root}",
            )

        with stats_lock:
            if state["batch_enroll_running"]:
                # Two concurrent full-corpus enrolments would fight over the
                # same speakers and multiply the CPU cost for no benefit.
                raise HTTPException(
                    status_code=409,
                    detail="a batch enrollment is already running; retry when it finishes",
                )
            state["batch_enroll_running"] = True
        try:
            report = await run_in_threadpool(
                batch_enroll_directory,
                verifier, root,
                workers=workers, backend=state["backend"], config=verifier.config,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            with stats_lock:
                state["batch_enroll_running"] = False

        _persist(verifier)
        n_enrolled = len(report.enrolled)
        _bump("total_enrollments", n_enrolled)
        _log_event("batch_enroll", f"Enrolled {n_enrolled} employees from disk")
        return report.to_dict()

    # ──────────────────────────────────────────────────────────────────────
    # Batch Endpoints
    # ──────────────────────────────────────────────────────────────────────

    @app.post("/batch/identify", dependencies=[Auth])
    async def batch_identify_endpoint(
        files: List[UploadFile] = File(...),
        verifier: SpeakerVerifier = Verifier,
    ) -> dict:
        _check_file_count(files)
        results = []
        for f in files:
            try:
                audio = await _read_clip(f)
            except HTTPException as exc:
                results.append({"filename": f.filename, "error": exc.detail})
                continue
            result = await run_in_threadpool(verifier.identify, audio)
            results.append({"filename": f.filename, **result.to_dict()})
        return {"count": len(results), "results": results}

    @app.post("/batch/enroll", dependencies=[Auth])
    async def batch_enroll_endpoint(
        name: str = Form(...),
        files: List[UploadFile] = File(...),
        verifier: SpeakerVerifier = Verifier,
    ) -> dict:
        name = _clean_name(name)
        _check_file_count(files)
        good_clips: list[np.ndarray] = []
        skipped: list[dict] = []
        for f in files:
            try:
                good_clips.append(await _read_clip(f))
            except HTTPException as exc:
                skipped.append({"filename": f.filename, "error": exc.detail})
        if not good_clips:
            raise HTTPException(
                status_code=400,
                detail=f"no readable clips; skipped: {skipped}",
            )
        try:
            await run_in_threadpool(verifier.enroll_from_audio, name, good_clips)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _persist(verifier)
        _bump("total_enrollments")
        _log_event("enroll", f"Enrolled {name} with {len(good_clips)} clips")
        return {
            "enrolled": name,
            "clips_used": len(good_clips),
            "skipped": skipped,
            "backend": verifier.encoder.name,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Static Web Dashboard
    # ──────────────────────────────────────────────────────────────────────

    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

        @app.get("/")
        def root():
            index = WEB_DIR / "index.html"
            if not index.is_file():
                return JSONResponse({"detail": "web dashboard not installed"}, status_code=404)
            return FileResponse(index)

    return app


# `uvicorn speech_sense.api:app` friendly module-level instance
app = create_app()
