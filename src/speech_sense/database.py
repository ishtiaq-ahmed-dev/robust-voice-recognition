"""Speaker database: name -> mean embedding, persisted as .npz.

The file format is NumPy's `.npz` read with `allow_pickle=False`, so opening a
database can never execute code. (An earlier version accepted `pickle` files
for migration; that path was arbitrary code execution against anyone who
pointed `--database` at an untrusted file and has been removed.)

Three durability properties this module is responsible for:

    * **Atomic writes.** `save()` writes a sibling temp file and `os.replace`s
      it over the target, so a crash or power loss mid-write leaves the
      previous database intact rather than a truncated one.
    * **Corruption never blocks startup.** A damaged file is moved aside to
      `<path>.corrupt-<timestamp>` and an empty database is returned, so the
      service boots, the operator gets a loud error, and nothing is destroyed.
    * **Rectangular embeddings.** Every vector in one database shares a
      dimension, so `matrix()` can never raise on a ragged stack.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import DEFAULT

log = logging.getLogger(__name__)

MAX_NAME_LEN = DEFAULT.max_speaker_name_len


class SpeakerDatabaseError(RuntimeError):
    """Raised when a database file exists but cannot be used."""


def clean_speaker_name(name: str) -> str:
    """Validate and normalise a speaker name.

    Names arrive from HTTP forms, CLI flags, and directory listings, so this is
    a trust boundary. NFC normalisation stops two visually identical names from
    becoming two separate database entries, and two Unicode categories are
    refused:

        Cc  control characters — they corrupt logs and terminal output.
        Cf  format characters — these include the bidi overrides (U+202E and
            friends) and zero-width joiners. In a display name they let stored
            text render as something else entirely, which in an identity system
            is a spoofing tool, not a typographic nicety.
    """
    if not isinstance(name, str):
        raise ValueError(f"speaker name must be a string, got {type(name).__name__}")
    cleaned = unicodedata.normalize("NFC", name).strip()
    if not cleaned:
        raise ValueError("speaker name must not be empty")
    if len(cleaned) > MAX_NAME_LEN:
        raise ValueError(f"speaker name must be at most {MAX_NAME_LEN} characters")
    if any(unicodedata.category(ch) in {"Cc", "Cf"} for ch in cleaned):
        raise ValueError(
            "speaker name must not contain control or text-direction characters"
        )
    return cleaned


@dataclass
class SpeakerDatabase:
    """In-memory speaker -> embedding store with save/load."""

    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, dict] = field(default_factory=dict)
    # Cached (names, matrix) so identify() doesn't restack the whole roster on
    # every request. Invalidated by add()/remove().
    _matrix_cache: tuple[list[str], np.ndarray] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    # The REST API serves requests from a thread pool, so one database is read
    # and written concurrently. Without this, a `np.stack` over the roster can
    # run while another request inserts a speaker and raise "dictionary changed
    # size during iteration" mid-identification. Re-entrant because add()
    # touches the cache that matrix() also guards.
    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False, compare=False
    )

    def __len__(self) -> int:
        return len(self.embeddings)

    def __contains__(self, name: str) -> bool:
        return name in self.embeddings

    def __iter__(self) -> Iterator[str]:
        return iter(self.embeddings)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self.embeddings)

    @property
    def dim(self) -> int | None:
        """Embedding dimension held by this database, or None when empty."""
        for emb in self.embeddings.values():
            return int(emb.shape[0])
        return None

    def add(self, name: str, embedding: np.ndarray, meta: dict | None = None) -> None:
        """Store `embedding` under `name`, replacing any previous entry.

        Rejects non-finite vectors (they poison every later cosine score and
        serialise as invalid JSON) and vectors whose dimension disagrees with
        the rest of the database — that happens when the embedding backend
        changes under an existing database, and silently mixing the two makes
        every similarity meaningless.
        """
        name = clean_speaker_name(name)
        emb = np.asarray(embedding, dtype=np.float32)
        if emb.ndim != 1:
            raise ValueError(f"embedding must be 1-D, got shape {emb.shape}")
        if emb.size == 0:
            raise ValueError("embedding must not be empty")
        if not np.all(np.isfinite(emb)):
            raise ValueError(f"embedding for {name!r} contains NaN or infinity")

        with self._lock:
            # Compare against the *other* entries: re-enrolling the only speaker
            # in a database may change its dimension, mixing two may not.
            others = [e for n, e in self.embeddings.items() if n != name]
            if others and emb.shape[0] != others[0].shape[0]:
                raise ValueError(
                    f"embedding dimension {emb.shape[0]} does not match the "
                    f"{others[0].shape[0]}-D vectors already in this database. This "
                    f"usually means the embedding backend changed; re-enrol every "
                    f"speaker with the new backend or use a separate database file."
                )

            self.embeddings[name] = emb
            self.metadata[name] = dict(meta or {})
            self._matrix_cache = None

    def remove(self, name: str) -> bool:
        with self._lock:
            if name not in self.embeddings:
                return False
            del self.embeddings[name]
            self.metadata.pop(name, None)
            self._matrix_cache = None
            return True

    def matrix(self) -> tuple[list[str], np.ndarray]:
        """Return (names_sorted, matrix of shape (n_speakers, dim)) for vectorised scoring.

        Cached: identify() would otherwise restack the entire roster on every
        request. The returned tuple is a consistent snapshot — a concurrent
        add() replaces the cache rather than mutating this array.
        """
        with self._lock:
            if self._matrix_cache is not None:
                return self._matrix_cache
            names = sorted(self.embeddings)
            if not names:
                result: tuple[list[str], np.ndarray] = ([], np.zeros((0, 0), dtype=np.float32))
            else:
                result = (names, np.stack([self.embeddings[n] for n in names]).astype(np.float32))
            self._matrix_cache = result
            return result

    # ---------- Persistence ---------- #

    def save(self, path: str | Path = DEFAULT.database_path) -> None:
        """Persist atomically: write a sibling temp file, then replace the target.

        A half-written database is never observable — readers see either the
        old file or the new one.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            if self.embeddings:
                names = list(self.embeddings)
                arrays: dict[str, np.ndarray] = {
                    f"emb__{i}": self.embeddings[n] for i, n in enumerate(names)
                }
                arrays["__names__"] = np.array(names)
                arrays["__meta__"] = np.array(json.dumps(self.metadata))
            else:
                # np.savez refuses to write nothing; a stub header keeps load() happy.
                arrays = {"__names__": np.array([]), "__meta__": np.array(json.dumps({}))}

        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(tmp_fd, "wb") as fh:
                # Hand np.savez a file object, not a path: given a path it
                # silently appends ".npz", which would write somewhere other
                # than where load() looks.
                np.savez(fh, **arrays)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str | Path = DEFAULT.database_path) -> "SpeakerDatabase":
        """Load a database, tolerating a missing or damaged file.

        A missing file is an empty database. A damaged one is quarantined to
        `<path>.corrupt-<timestamp>` and reported at ERROR — the service still
        starts, and the unreadable bytes are preserved for recovery instead of
        being overwritten by the next save.
        """
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            return cls._load_npz(path)
        except Exception as exc:  # noqa: BLE001 — any decode failure is corruption
            quarantine = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
            try:
                os.replace(path, quarantine)
                moved = f"moved to {quarantine}"
            except OSError as move_exc:
                moved = f"could not quarantine it ({move_exc})"
            log.error(
                "speaker database %s is unreadable (%s: %s); %s. "
                "Starting with an empty database.",
                path, type(exc).__name__, exc, moved,
            )
            return cls()

    @classmethod
    def _load_npz(cls, path: Path) -> "SpeakerDatabase":
        db = cls()
        # Open the file ourselves rather than handing np.load a path: when
        # np.load raises on a damaged file it leaves the descriptor open, and
        # on Windows that makes the quarantine rename fail with "file in use".
        with open(path, "rb") as fh, np.load(fh, allow_pickle=False) as data:
            if "__names__" not in data:
                raise SpeakerDatabaseError("missing __names__ array — not a speaker database")
            names = [str(n) for n in data["__names__"]]
            try:
                meta_raw = data["__meta__"].item() if "__meta__" in data else "{}"
                meta = json.loads(meta_raw)
                if not isinstance(meta, dict):
                    raise ValueError("metadata is not an object")
            except (ValueError, TypeError) as exc:
                # Metadata is descriptive only — losing it must not lose voices.
                log.warning("discarding unreadable metadata in %s: %s", path, exc)
                meta = {}

            loaded: dict[str, np.ndarray] = {}
            for i, name in enumerate(names):
                key = f"emb__{i}"
                if key not in data:
                    log.warning("speaker %r in %s has no embedding; skipping", name, path)
                    continue
                emb = np.asarray(data[key], dtype=np.float32).reshape(-1)
                if emb.size == 0 or not np.all(np.isfinite(emb)):
                    log.warning("speaker %r in %s has an invalid embedding; skipping", name, path)
                    continue
                loaded[name] = emb

        # A database written before/after a backend swap can hold two different
        # embedding sizes. Keep the majority and drop the rest rather than
        # letting a ragged stack crash every identify() call.
        dims = Counter(e.shape[0] for e in loaded.values())
        if len(dims) > 1:
            keep_dim = dims.most_common(1)[0][0]
            dropped = [n for n, e in loaded.items() if e.shape[0] != keep_dim]
            log.error(
                "%s mixes embedding dimensions %s; keeping the %d-D majority and "
                "dropping %d speaker(s): %s. Re-enrol them with the current backend.",
                path, sorted(dims), keep_dim, len(dropped), ", ".join(sorted(dropped)),
            )
            loaded = {n: e for n, e in loaded.items() if e.shape[0] == keep_dim}

        for name, emb in loaded.items():
            db.embeddings[name] = emb
            entry = meta.get(name, {})
            db.metadata[name] = entry if isinstance(entry, dict) else {}
        return db
