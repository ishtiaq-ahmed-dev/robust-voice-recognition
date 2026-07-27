"""Speaker verifier — orchestrates VAD, embedding, database and scoring.

Confidence semantics (deliberate, documented):

    * `similarity`     — raw cosine (-1..1) between the query and the best speaker.
    * `margin`         — similarity gap between best and second-best speaker.
    * `confidence`     — 0..1, calibrated from similarity + margin so callers can
                         treat it as a probability-like score. A prediction with
                         high similarity but tiny margin is *not* confident, and
                         this is what the confidence score encodes.

`identify()` returns a `VerificationResult` including all three; downstream
threshold decisions read `confidence` (with `is_known` as the boolean shortcut).

Every path that cannot produce a decision — empty database, silence, a backend
whose embedding size disagrees with the enrolled vectors — returns a populated
`VerificationResult` with `is_known=False` and a `reason`, never an exception.
Failing closed is the only safe default for an authentication primitive.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from . import vad as vad_mod
from .audio import load_wav, sanitize
from .config import DEFAULT, Config
from .database import SpeakerDatabase, clean_speaker_name
from .embedding import SpeakerEncoder

log = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Structured verifier output — safe to serialize as JSON."""

    speaker: Optional[str]
    is_known: bool
    similarity: float
    margin: float
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    contains_speech: bool = True
    backend: str = ""
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "is_known": self.is_known,
            "similarity": round(self.similarity, 4),
            "margin": round(self.margin, 4),
            "confidence": round(self.confidence, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "contains_speech": self.contains_speech,
            "backend": self.backend,
            "reason": self.reason,
        }


class SpeakerVerifier:
    """High-level API — hides embedding backend, VAD, and DB from callers."""

    def __init__(
        self,
        encoder: SpeakerEncoder | None = None,
        database: SpeakerDatabase | None = None,
        config: Config = DEFAULT,
        use_vad: bool = True,
    ) -> None:
        self.config = config
        self.encoder = encoder or SpeakerEncoder.load("auto")
        self.database = database or SpeakerDatabase()
        self.use_vad = use_vad

    # ---------- Enrollment ---------- #

    def enroll_from_audio(self, name: str, clips: list[np.ndarray]) -> np.ndarray:
        """Average per-clip embeddings and store under `name`.

        Raises ValueError on an invalid name, empty input, or when every clip
        is silence.
        """
        name = clean_speaker_name(name)
        if not clips:
            raise ValueError("at least one audio clip is required")

        embeddings = []
        for clip in clips:
            processed = self._preprocess(clip)
            if processed is None or processed.size == 0:
                # Silence-only clip — skip but don't fail the whole enrollment.
                continue
            emb = sanitize(self.encoder.embed(processed, self.config.sample_rate))
            if emb.size == 0 or not np.any(emb):
                continue
            embeddings.append(emb)

        if not embeddings:
            raise ValueError(
                f"all {len(clips)} enrollment clips were silence — try again in a quieter room"
            )

        mean_emb = np.mean(embeddings, axis=0).astype(np.float32)
        norm = float(np.linalg.norm(mean_emb))
        if not math.isfinite(norm) or norm < 1e-9:
            # Cancelling embeddings would otherwise be "normalised" by dividing
            # by ~1e-9 and stored as a vector of infinities.
            raise ValueError(
                "enrollment clips produced a degenerate voice-print; record longer, clearer samples"
            )
        # L2 normalise so cosine similarity is well-defined.
        mean_emb = (mean_emb / norm).astype(np.float32)
        self.database.add(
            name,
            mean_emb,
            meta={"n_clips": len(embeddings), "backend": self.encoder.name},
        )
        return mean_emb

    def enroll_from_files(self, name: str, paths: list[str]) -> np.ndarray:
        clips = [load_wav(p, self.config.sample_rate) for p in paths]
        return self.enroll_from_audio(name, clips)

    # ---------- Identification ---------- #

    def identify(self, audio: np.ndarray) -> VerificationResult:
        """Return the most likely speaker plus confidence."""
        ranked = self._rank(audio)
        if isinstance(ranked, VerificationResult):
            return ranked
        names, sims = ranked

        order = np.argsort(-sims)
        best_idx = int(order[0])
        best_sim = float(sims[best_idx])
        second_sim = float(sims[order[1]]) if len(order) > 1 else 0.0
        margin = best_sim - second_sim

        confidence = self._confidence(best_sim, margin)
        is_known = (
            best_sim >= self.config.similarity_threshold
            and margin >= self.config.known_speaker_margin
        )

        return VerificationResult(
            speaker=names[best_idx] if is_known else None,
            is_known=is_known,
            similarity=best_sim,
            margin=margin,
            confidence=confidence,
            scores=self._top_scores(names, sims, order),
            contains_speech=True,
            backend=self.encoder.name,
            reason="ok" if is_known else "below similarity threshold or ambiguous",
        )

    def identify_file(self, path: str) -> VerificationResult:
        audio = load_wav(path, self.config.sample_rate)
        return self.identify(audio)

    def verify(self, audio: np.ndarray, claimed: str) -> VerificationResult:
        """1-vs-1 verification: is `audio` the person we claim it is?"""
        try:
            claimed = clean_speaker_name(claimed)
        except ValueError as exc:
            return self._failure(f"invalid claimed speaker name: {exc}")
        if claimed not in self.database:
            return self._failure(f"claimed speaker {claimed!r} is not enrolled")

        ranked = self._rank(audio)
        if isinstance(ranked, VerificationResult):
            # Silence / empty DB / dimension mismatch — keep the real reason
            # rather than overwriting it with a threshold message.
            return ranked
        names, sims = ranked

        by_name = dict(zip(names, (float(s) for s in sims)))
        similarity = by_name[claimed]
        rest = [s for n, s in by_name.items() if n != claimed]
        margin = similarity - (max(rest) if rest else 0.0)
        is_known = similarity >= self.config.similarity_threshold

        order = np.argsort(-sims)
        return VerificationResult(
            speaker=claimed if is_known else None,
            is_known=is_known,
            similarity=similarity,
            margin=margin,
            confidence=self._confidence(similarity, margin),
            scores=self._top_scores(names, sims, order, always_include=claimed),
            contains_speech=True,
            backend=self.encoder.name,
            reason="ok" if is_known else "similarity below threshold",
        )

    # ---------- Internals ---------- #

    def _rank(self, audio: np.ndarray) -> Union[tuple[list[str], np.ndarray], VerificationResult]:
        """Score `audio` against every enrolled speaker.

        Returns `(names, similarities)` on success, or a finished
        `VerificationResult` explaining why scoring was not possible.
        """
        if not self.database:
            return self._failure(
                "database is empty; enroll at least one speaker first",
                contains_speech=False,
            )

        processed = self._preprocess(audio)
        if processed is None or processed.size == 0:
            return self._failure("no speech detected in the audio", contains_speech=False)

        query = sanitize(self.encoder.embed(processed, self.config.sample_rate))
        norm = float(np.linalg.norm(query))
        if not math.isfinite(norm) or norm < 1e-9:
            return self._failure("audio produced a degenerate embedding", contains_speech=False)
        query = query / norm

        names, mat = self.database.matrix()
        if mat.shape[1] != query.shape[0]:
            # Enrolling under one backend and querying under another is a
            # configuration error, not a match failure. Fail closed, loudly.
            msg = (
                f"enrolled voice-prints are {mat.shape[1]}-D but the "
                f"{self.encoder.name} backend produces {query.shape[0]}-D vectors; "
                f"re-enrol every speaker with this backend"
            )
            log.error("%s", msg)
            return self._failure(msg)

        # Enrolled embeddings are already unit-normalised on ingest, so a dot
        # product gives cosine similarity directly.
        return names, np.asarray(mat @ query, dtype=np.float64)

    def _failure(self, reason: str, contains_speech: bool = True) -> VerificationResult:
        return VerificationResult(
            speaker=None, is_known=False,
            similarity=0.0, margin=0.0, confidence=0.0,
            contains_speech=contains_speech, backend=self.encoder.name,
            reason=reason,
        )

    def _top_scores(
        self,
        names: list[str],
        sims: np.ndarray,
        order: np.ndarray,
        always_include: str | None = None,
    ) -> dict[str, float]:
        """The `top_k_scores` highest-scoring speakers.

        The full table is deliberately not returned: it lets any caller
        enumerate the entire enrolled roster from a single request, and it
        makes the response grow linearly with the number of speakers.
        """
        k = max(1, min(self.config.top_k_scores, len(names)))
        selected = {names[int(i)]: float(sims[int(i)]) for i in order[:k]}
        if always_include is not None and always_include not in selected:
            idx = names.index(always_include)
            selected[always_include] = float(sims[idx])
        return selected

    def _preprocess(self, audio: np.ndarray) -> np.ndarray | None:
        """Trimmed speech, or None when the clip has none.

        One VAD pass: `extract_speech` returns None precisely when no speech
        segment was confirmed, which is the signal `_rank` needs.
        """
        audio = sanitize(audio)
        if audio.size == 0:
            return None
        if not self.use_vad:
            return audio
        return vad_mod.extract_speech(audio, self.config.sample_rate)

    @staticmethod
    def _confidence(similarity: float, margin: float) -> float:
        # Sigmoid over similarity (centred at 0.7) with margin as a multiplicative
        # boost. Chosen to make high-margin/high-sim -> ~0.95+, low-margin/high-sim
        # -> ~0.6, so identical twins do not get flagged as confident.
        if not (math.isfinite(similarity) and math.isfinite(margin)):
            return 0.0
        sim_component = 1.0 / (1.0 + math.exp(-12.0 * (similarity - 0.7)))
        margin_component = 1.0 / (1.0 + math.exp(-30.0 * (margin - 0.05)))
        combined = sim_component * (0.6 + 0.4 * margin_component)
        return float(max(0.0, min(1.0, combined)))
