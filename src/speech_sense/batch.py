"""Batch / parallel enrollment and identification.

Designed for the "GBs of audio" case: point it at a directory tree of audio and
it enrolls or identifies everything using a process pool. The pool size
defaults to `Config.workers` (all-but-one CPU) and is capped at
`SPEECH_SENSE_WORKERS` when set.

Layout convention for `batch_enroll_directory`:

    dataset/
        alice/
            take1.wav
            take2.wav
        bob/
            ...

Each top-level subdirectory is one speaker; all audio beneath contributes clips.
Any speaker with fewer than one valid clip is skipped with a warning, not a
hard error — we don't want a single unreadable file to blow up a 10k-file run.

All three public entry points share `_embed_many`, which owns every failure
mode a long batch run actually hits: an unreadable file, a worker killed by the
OOM reaper (`BrokenProcessPool`), and an environment where a process pool
cannot be created at all. The first two degrade to a per-file error string; the
last degrades to serial execution. A batch job that dies 80% of the way through
a million clips is worse than a slow one.

`ponytail`: uses `multiprocessing.get_context("spawn")` explicitly so the code
behaves the same on Windows / macOS / Linux. `ProcessPoolExecutor` would be
one line shorter but its Windows semantics are less obvious.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np

from .audio import load_wav
from .config import DEFAULT, Config
from .embedding import Backend, SpeakerEncoder
from .verifier import SpeakerVerifier, VerificationResult

log = logging.getLogger(__name__)

AUDIO_SUFFIXES = {
    ".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".m4a",
    ".aac", ".aiff", ".aif", ".wma", ".webm",
}

# One task: (path, target_sample_rate).
Task = tuple[str, int]
# One result: (index_into_tasks, path, embedding_or_None, error_message).
EmbedResult = tuple[int, str, "np.ndarray | None", str]


@dataclass
class BatchIdentifyRow:
    path: str
    result: VerificationResult

    def to_dict(self) -> dict:
        return {"path": self.path, **self.result.to_dict()}


@dataclass
class BatchEnrollReport:
    enrolled: dict[str, int]  # speaker -> num clips ingested
    skipped: dict[str, list[str]]  # speaker -> list of failed clip paths
    n_speakers: int
    n_clips: int

    def to_dict(self) -> dict:
        return {
            "enrolled": self.enrolled,
            "skipped": self.skipped,
            "n_speakers": self.n_speakers,
            "n_clips": self.n_clips,
        }


# ---------- Worker functions (must be top-level for spawn pickling) ---------- #

_WORKER_ENCODER: SpeakerEncoder | None = None
_WORKER_BACKEND: str | None = None


def _init_worker(backend: str) -> None:
    """Load the encoder once per process.

    Also called on the serial path, which `batch_identify_stream` re-enters
    once per chunk — without the guard that reloads the model (and, for
    resemblyzer, re-reads its weights) every 32 files.
    """
    global _WORKER_ENCODER, _WORKER_BACKEND
    if _WORKER_ENCODER is not None and _WORKER_BACKEND == backend:
        return
    _WORKER_ENCODER = SpeakerEncoder.load(backend)  # type: ignore[arg-type]
    _WORKER_BACKEND = backend


def _embed_one(task: Task) -> tuple[str, "np.ndarray | None", str]:
    """Load a file and embed it. Returns (path, embedding_or_None, error_msg)."""
    path, sr = task
    try:
        if _WORKER_ENCODER is None:  # pragma: no cover — defensive
            raise RuntimeError("worker encoder was not initialised")
        audio = load_wav(path, sr)
        return path, _WORKER_ENCODER.embed(audio, sr), ""
    except Exception as exc:  # noqa: BLE001 — batch jobs should not die on one bad file
        return path, None, f"{type(exc).__name__}: {exc}"


def _embed_one_indexed(indexed: tuple[int, Task]) -> EmbedResult:
    i, task = indexed
    path, emb, err = _embed_one(task)
    return i, path, emb, err


# ---------- Public API ---------- #

def iter_audio_files(root: str | Path) -> Iterator[Path]:
    """Yield every audio file under `root`, sorted for determinism."""
    root = Path(root)
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        return
    for p in sorted(root.rglob("*")):
        try:
            if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES:
                yield p
        except OSError as exc:  # broken symlink, permission denied, too-long path
            log.warning("skipping %s: %s", p, exc)


def _embed_many(
    tasks: Sequence[Task],
    workers: int | None,
    backend: Backend,
    config: Config,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> Iterator[EmbedResult]:
    """Embed every task, yielding results as they land (completion order).

    Never raises on a per-file problem and never raises because the pool died —
    both surface as an error string on the affected task.
    """
    total = len(tasks)
    if total == 0:
        return

    n_workers = min(workers or config.workers, total)
    done = 0
    emitted: set[int] = set()

    def _serial(subset: Sequence[tuple[int, Task]]) -> Iterator[EmbedResult]:
        nonlocal done
        _init_worker(backend)
        for i, task in subset:
            result = _embed_one_indexed((i, task))
            done += 1
            emitted.add(i)
            if on_progress:
                on_progress(result[1], done, total)
            yield result

    indexed = list(enumerate(tasks))
    if n_workers <= 1 or total == 1:
        # Skip the pool overhead for tiny jobs.
        yield from _serial(indexed)
        return

    try:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(backend,),
        ) as pool:
            futures = {pool.submit(_embed_one_indexed, item): item for item in indexed}
            for future in as_completed(futures):
                i, task = futures[future]
                try:
                    result = future.result()
                except BrokenExecutor:
                    # A worker died (usually the OOM reaper), which poisons
                    # every outstanding future. Bail out to the serial
                    # fallback below rather than marking thousands of
                    # perfectly good files as failed.
                    raise
                except Exception as exc:  # noqa: BLE001
                    result = (i, task[0], None, f"{type(exc).__name__}: {exc}")
                done += 1
                emitted.add(i)
                if on_progress:
                    on_progress(result[1], done, total)
                yield result
    except (BrokenExecutor, OSError, ValueError) as exc:
        # Either the pool could not be created (sandbox with no spawn,
        # exhausted descriptors, no /dev/shm) or it collapsed mid-run. One
        # process uses less memory than N, so retrying serially is a real
        # recovery, not just a slower repeat of the same failure.
        log.warning(
            "process pool unusable (%s: %s); finishing %d remaining file(s) serially",
            type(exc).__name__, exc, total - len(emitted),
        )
        yield from _serial([item for item in indexed if item[0] not in emitted])


def batch_enroll_directory(
    verifier: SpeakerVerifier,
    dataset_root: str | Path,
    workers: int | None = None,
    backend: Backend = "auto",
    config: Config = DEFAULT,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> BatchEnrollReport:
    """Enroll every speaker under `dataset_root/<speaker>/*` in parallel.

    Any prior enrollment for a given speaker is overwritten (last write wins).
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise ValueError(f"dataset root is not a directory: {dataset_root}")

    speakers: dict[str, list[Path]] = {}
    for sp_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        clips = list(iter_audio_files(sp_dir))
        if clips:
            speakers[sp_dir.name] = clips

    if not speakers:
        return BatchEnrollReport(enrolled={}, skipped={}, n_speakers=0, n_clips=0)

    tasks: list[Task] = []
    task_to_speaker: list[str] = []
    for name, clips in speakers.items():
        for c in clips:
            tasks.append((str(c), config.sample_rate))
            task_to_speaker.append(name)

    embeddings_by_speaker: dict[str, list[np.ndarray]] = {n: [] for n in speakers}
    skipped: dict[str, list[str]] = {n: [] for n in speakers}

    for i, path, emb, err in _embed_many(tasks, workers, backend, config, on_progress):
        name = task_to_speaker[i]
        if emb is not None:
            embeddings_by_speaker[name].append(emb)
        else:
            skipped[name].append(f"{path}: {err}")

    n_ingested = 0
    enrolled_counts: dict[str, int] = {}
    for name, embs in embeddings_by_speaker.items():
        mean = _mean_unit_vector(embs)
        if mean is None:
            if not skipped[name]:
                skipped[name].append("no usable audio produced an embedding")
            continue
        try:
            verifier.database.add(
                name, mean,
                meta={"n_clips": len(embs), "backend": verifier.encoder.name, "batch_enrolled": True},
            )
        except ValueError as exc:
            # A bad name or a dimension clash must not abandon the other speakers.
            log.error("could not enrol %r: %s", name, exc)
            skipped[name].append(str(exc))
            continue
        enrolled_counts[name] = len(embs)
        n_ingested += len(embs)

    return BatchEnrollReport(
        enrolled=enrolled_counts,
        skipped={k: v for k, v in skipped.items() if v},
        n_speakers=len(enrolled_counts),
        n_clips=n_ingested,
    )


def _mean_unit_vector(embeddings: list[np.ndarray]) -> np.ndarray | None:
    """Average and L2-normalise, or None when the result is unusable."""
    if not embeddings:
        return None
    mean = np.mean(embeddings, axis=0).astype(np.float32)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm < 1e-9:
        return None
    return (mean / norm).astype(np.float32)


def batch_identify(
    verifier: SpeakerVerifier,
    audio_paths: Iterable[str | Path],
    workers: int | None = None,
    backend: Backend = "auto",
    config: Config = DEFAULT,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> list[BatchIdentifyRow]:
    """Identify a batch of audio files in parallel; return one row per input.

    Rows preserve input order.
    """
    paths = [str(p) for p in audio_paths]
    if not paths:
        return []

    tasks: list[Task] = [(p, config.sample_rate) for p in paths]
    # Preallocate so results come back in input order regardless of completion
    # order from the pool.
    embeddings: list[np.ndarray | None] = [None] * len(tasks)
    errors: list[str] = [""] * len(tasks)

    for i, _path, emb, err in _embed_many(tasks, workers, backend, config, on_progress):
        embeddings[i] = emb
        errors[i] = err

    names, mat = verifier.database.matrix()
    rows: list[BatchIdentifyRow] = []
    for path, emb, err in zip(paths, embeddings, errors):
        rows.append(BatchIdentifyRow(path=path, result=_score_row(verifier, names, mat, emb, err)))
    return rows


def _score_row(
    verifier: SpeakerVerifier,
    names: list[str],
    mat: np.ndarray,
    emb: np.ndarray | None,
    err: str,
) -> VerificationResult:
    """Turn one embedding into a VerificationResult, mirroring SpeakerVerifier.identify."""
    def failure(reason: str, contains_speech: bool) -> VerificationResult:
        return VerificationResult(
            speaker=None, is_known=False,
            similarity=0.0, margin=0.0, confidence=0.0,
            contains_speech=contains_speech, backend=verifier.encoder.name,
            reason=reason,
        )

    if emb is None:
        return failure(f"could not read audio: {err}", contains_speech=False)
    if not names:
        return failure("database is empty", contains_speech=True)

    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(emb))
    if not np.isfinite(norm) or norm < 1e-9:
        return failure("audio produced a degenerate embedding", contains_speech=False)
    if mat.shape[1] != emb.shape[0]:
        return failure(
            f"enrolled voice-prints are {mat.shape[1]}-D but this backend produces "
            f"{emb.shape[0]}-D vectors; re-enrol every speaker with this backend",
            contains_speech=True,
        )

    sims = np.asarray(mat @ (emb / norm), dtype=np.float64)
    order = np.argsort(-sims)
    best = int(order[0])
    best_sim = float(sims[best])
    second = float(sims[order[1]]) if len(order) > 1 else 0.0
    margin = best_sim - second
    is_known = (
        best_sim >= verifier.config.similarity_threshold
        and margin >= verifier.config.known_speaker_margin
    )
    k = max(1, min(verifier.config.top_k_scores, len(names)))
    return VerificationResult(
        speaker=names[best] if is_known else None,
        is_known=is_known,
        similarity=best_sim, margin=margin,
        confidence=SpeakerVerifier._confidence(best_sim, margin),
        scores={names[int(i)]: float(sims[int(i)]) for i in order[:k]},
        contains_speech=True, backend=verifier.encoder.name,
        reason="ok" if is_known else "below similarity threshold or ambiguous",
    )


def batch_identify_directory(
    verifier: SpeakerVerifier,
    directory: str | Path,
    **kwargs,
) -> list[BatchIdentifyRow]:
    """Convenience: identify every audio file under `directory`."""
    return batch_identify(verifier, list(iter_audio_files(directory)), **kwargs)


def batch_identify_stream(
    verifier: SpeakerVerifier,
    audio_paths: Iterable[str | Path],
    workers: int | None = None,
    backend: Backend = "auto",
    config: Config = DEFAULT,
    chunk_size: int | None = None,
) -> Iterator[BatchIdentifyRow]:
    """Generator variant of batch_identify — yields results as they land.

    Use for datasets too large to hold every embedding in memory at once
    (millions of clips). Processes files in fixed-size chunks so memory stays
    bounded.
    """
    paths = [str(p) for p in audio_paths]
    if not paths:
        return
    chunk = chunk_size or max(config.batch_size, 32)
    for i in range(0, len(paths), chunk):
        yield from batch_identify(
            verifier, paths[i : i + chunk],
            workers=workers, backend=backend, config=config,
        )


def batch_enroll_streaming(
    verifier: SpeakerVerifier,
    dataset_root: str | Path,
    workers: int | None = None,
    backend: Backend = "auto",
    config: Config = DEFAULT,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> Iterator[tuple[str, int, dict]]:
    """Streaming variant of batch_enroll_directory — enrolls speakers one at a
    time and yields (speaker_name, n_clips_used, meta) after each.

    Suitable for corpora with tens of thousands of speakers where holding
    every embedding across every speaker in memory would be wasteful. The
    caller is expected to persist the database periodically so a crash mid-run
    doesn't lose progress.
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise ValueError(f"dataset root is not a directory: {dataset_root}")

    speaker_dirs = sorted(p for p in dataset_root.iterdir() if p.is_dir())
    total = len(speaker_dirs)

    for done, sp_dir in enumerate(speaker_dirs, start=1):
        clips = list(iter_audio_files(sp_dir))
        if not clips:
            continue
        tasks: list[Task] = [(str(c), config.sample_rate) for c in clips]
        embeddings = [
            emb for _i, _p, emb, _e in _embed_many(tasks, workers, backend, config)
            if emb is not None
        ]

        mean = _mean_unit_vector(embeddings)
        if mean is None:
            log.warning("no usable audio for speaker %r; skipping", sp_dir.name)
            continue
        meta = {"n_clips": len(embeddings), "backend": verifier.encoder.name, "streaming": True}
        try:
            verifier.database.add(sp_dir.name, mean, meta=meta)
        except ValueError as exc:
            log.error("could not enrol %r: %s", sp_dir.name, exc)
            continue
        if on_progress:
            on_progress(sp_dir.name, done, total)
        yield sp_dir.name, len(embeddings), meta
