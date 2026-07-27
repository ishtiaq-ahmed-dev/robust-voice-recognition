"""Speaker embedding backends.

Two backends live behind a common `SpeakerEncoder` interface:

    * ResemblyzerBackend — GE2E-trained CNN, produces 256-D unit vectors and
      is the model actually used at runtime.
    * MfccBackend       — pooled MFCC statistics, 78-D. Only used when
      Resemblyzer isn't installed (CI, tests, environments without model
      downloads). Reasonable-but-mediocre; documented as such.

Callers select via `SpeakerEncoder.load(backend="auto")`. `auto` picks
Resemblyzer if it can actually be constructed, otherwise the MFCC fallback
with a warning so the degraded mode is never silent.
"""

from __future__ import annotations

import logging
from typing import Literal, Protocol

import numpy as np

from .audio import sanitize
from .config import DEFAULT

log = logging.getLogger(__name__)

Backend = Literal["resemblyzer", "mfcc", "auto"]


class SpeakerEncoderBackend(Protocol):
    """Strategy-shape protocol for anything that turns audio -> embedding."""

    name: str
    dim: int

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray: ...


class MfccBackend:
    """Fallback backend: pooled MFCC + delta statistics as a speaker embedding.

    This is NOT a real speaker embedding — it will discriminate strongly-
    different voices but confuse similar ones. It exists so the pipeline runs
    end-to-end in test environments that can't fetch the Resemblyzer weights.
    """

    name = "mfcc"
    dim = 39 * 2  # (13 MFCC + 13 delta + 13 delta-delta) mean and std

    def __init__(self, n_mfcc: int = 13, hop_length: int = 512) -> None:
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import librosa

        audio = sanitize(audio)
        if audio.size == 0:
            return np.zeros(self.dim, dtype=np.float32)
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length
        )
        d1, d2 = self._deltas(mfcc)
        stacked = np.concatenate([mfcc, d1, d2], axis=0)  # (39, T)
        pooled = np.concatenate([stacked.mean(axis=1), stacked.std(axis=1)])
        pooled = sanitize(pooled)
        # L2 normalise so cosine similarity behaves consistently across backends.
        n = float(np.linalg.norm(pooled))
        if n < 1e-9:
            return np.zeros(self.dim, dtype=np.float32)
        return (pooled / n).astype(np.float32)

    @staticmethod
    def _deltas(mfcc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Delta and delta-delta, tolerating clips too short for the default window.

        `librosa.feature.delta` raises when its Savitzky-Golay width exceeds
        the frame count, so a sub-second clip would crash the encoder outright.
        Narrow the window to fit, and fall back to zeros when even that is
        impossible (fewer than 3 frames).
        """
        import librosa

        n_frames = mfcc.shape[1]
        width = min(9, n_frames if n_frames % 2 else n_frames - 1)
        if width < 3:
            zeros = np.zeros_like(mfcc)
            return zeros, zeros.copy()
        return (
            librosa.feature.delta(mfcc, width=width),
            librosa.feature.delta(mfcc, order=2, width=width),
        )


class ResemblyzerBackend:
    """Wraps the pretrained Resemblyzer VoiceEncoder (256-D speaker vectors)."""

    name = "resemblyzer"
    dim = 256

    def __init__(self) -> None:
        from resemblyzer import VoiceEncoder  # deferred to avoid heavy import cost

        self._encoder = VoiceEncoder(verbose=False)

    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        from resemblyzer import preprocess_wav

        audio = sanitize(audio)
        if audio.size == 0:
            return np.zeros(self.dim, dtype=np.float32)
        wav = preprocess_wav(audio, source_sr=sr)
        if np.asarray(wav).size == 0:
            # Resemblyzer runs its own webrtcvad pass and can trim a marginal
            # clip to nothing; embed_utterance would then raise on an empty
            # array. An all-zero vector is rejected upstream with a clear
            # "degenerate embedding" reason instead.
            log.debug("resemblyzer preprocessing removed the entire clip")
            return np.zeros(self.dim, dtype=np.float32)
        return sanitize(self._encoder.embed_utterance(wav))


class SpeakerEncoder:
    """Thin adapter over a backend so verifier/API code doesn't care which model runs."""

    def __init__(self, backend: SpeakerEncoderBackend) -> None:
        self.backend = backend

    @property
    def name(self) -> str:
        return self.backend.name

    @property
    def dim(self) -> int:
        return self.backend.dim

    def embed(self, audio: np.ndarray, sr: int = DEFAULT.sample_rate) -> np.ndarray:
        return self.backend.embed(audio, sr)

    def embed_batch(self, clips: list[np.ndarray], sr: int = DEFAULT.sample_rate) -> np.ndarray:
        # Kept simple — Resemblyzer's own batching is limited and the fallback
        # is CPU-only. If this ever bottlenecks, switch to embed_utterances().
        return np.stack([self.embed(c, sr) for c in clips]) if clips else np.zeros((0, self.dim), dtype=np.float32)

    @classmethod
    def load(cls, backend: Backend = "auto") -> "SpeakerEncoder":
        if backend == "auto":
            try:
                return cls(ResemblyzerBackend())
            except Exception as exc:  # noqa: BLE001
                # Not just ImportError: a missing weights download, an
                # incompatible torch build, or no writable cache directory all
                # fail here, and every one of them should degrade to the
                # fallback rather than take the service down.
                log.warning(
                    "resemblyzer unavailable (%s: %s); falling back to the MFCC "
                    "embedding, which is materially less accurate. Install "
                    "`speech_sense[model]` for production accuracy.",
                    type(exc).__name__, exc,
                )
                return cls(MfccBackend())
        if backend == "resemblyzer":
            return cls(ResemblyzerBackend())
        if backend == "mfcc":
            return cls(MfccBackend())
        raise ValueError(f"unknown backend: {backend!r} (expected auto, resemblyzer, or mfcc)")
