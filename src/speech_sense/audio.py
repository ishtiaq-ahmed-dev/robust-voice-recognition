"""Audio I/O, resampling, and noise-injection utilities.

Kept free of any speaker-model dependency so that unit tests and the API can
exercise the preprocessing pipeline without pulling in Resemblyzer/torch.

Everything that returns a waveform routes through `sanitize()`. Decoders hand
back NaN and infinity more often than you'd like (truncated float WAVs,
audioread fallbacks, division inside a resampler), and a single NaN sample
propagates all the way to a NaN cosine score, which then serialises as the
bare token `NaN` — invalid JSON that breaks strict clients.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf

from .config import DEFAULT

log = logging.getLogger(__name__)

PathLike = Union[str, Path]


def sanitize(audio: np.ndarray) -> np.ndarray:
    """Coerce to a 1-D float32 waveform with no NaN or infinity.

    Never writes through to the caller's array — `np.asarray` on an array that
    is already float32 hands back the same buffer, so an in-place fix here
    would silently mutate whatever the caller passed in.
    """
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size and not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def load_audio(path: PathLike, target_sr: int = DEFAULT.sample_rate) -> np.ndarray:
    """Load any human-recordable audio format as mono float32 in [-1, 1] at target_sr.

    Tries libsndfile first (WAV / FLAC / OGG / AIFF — fast, native, no ffmpeg).
    Falls back to librosa for anything libsndfile can't handle (MP3, M4A, WMA,
    WebM, MP4, AAC, …). librosa's audioread backend calls out to ffmpeg or
    GStreamer when installed. Raises a friendly error naming the file when both
    fail — a corrupt input should never bubble a mysterious IO error.
    """
    path = str(path)
    if not Path(path).is_file():
        # Otherwise a typo'd path is reported as a decode failure with advice
        # about installing ffmpeg, which is not the user's problem.
        raise FileNotFoundError(f"no such audio file: {path!r}")
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        # soundfile lays channels out as (samples, channels).
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
    except Exception as sf_exc:  # noqa: BLE001 — any libsndfile failure means "try librosa"
        try:
            import librosa

            audio, sr = librosa.load(path, sr=None, mono=True)
            audio = np.asarray(audio, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"could not decode audio {path!r}. "
                f"Install ffmpeg for MP3/M4A/WebM support "
                f"(libsndfile: {sf_exc}; librosa: {exc})"
            ) from exc

    audio = sanitize(audio)
    if audio.size == 0:
        raise ValueError(f"audio file {path!r} decoded to zero samples")
    if sr != target_sr:
        audio = resample(audio, int(sr), target_sr)
    return normalize(audio)


# Backwards-compatible alias — existing callers use this name.
load_wav = load_audio


def save_wav(path: PathLike, audio: np.ndarray, sr: int = DEFAULT.sample_rate) -> None:
    """Save a mono float32 waveform to a WAV file."""
    if sr <= 0:
        raise ValueError(f"sample rate must be positive, got {sr}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), sanitize(audio), sr)


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using scipy.signal.resample_poly (avoids a librosa hard dep at import)."""
    if orig_sr <= 0 or target_sr <= 0:
        # A decoder reporting sr=0 would otherwise reach gcd()/floor-division
        # and raise ZeroDivisionError deep inside the request handler.
        raise ValueError(f"sample rates must be positive, got {orig_sr} -> {target_sr}")
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if orig_sr == target_sr or audio.size == 0:
        return audio
    from math import gcd

    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    from scipy.signal import resample_poly  # local import keeps top import cheap

    return sanitize(resample_poly(audio, up, down))


def normalize(audio: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Peak-normalize; returns zeros unchanged instead of dividing by zero."""
    audio = sanitize(audio)
    m = float(np.max(np.abs(audio))) if audio.size else 0.0
    if m < 1e-9:
        return audio
    return (audio * (peak / m)).astype(np.float32)


def add_white_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add Gaussian noise to reach a given signal-to-noise ratio (dB)."""
    rng = rng or np.random.default_rng(0)
    audio = sanitize(audio)
    if audio.size == 0:
        return audio
    signal_power = float(np.mean(audio.astype(np.float64) ** 2))
    if signal_power < 1e-12:
        return audio
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
    return sanitize(audio + noise)


def rms(audio: np.ndarray) -> float:
    """Root-mean-square energy of a waveform."""
    audio = sanitize(audio)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def frame_signal(audio: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """Slice the signal into overlapping frames of shape (n_frames, frame_len).

    Any trailing samples that don't fit a full frame are dropped — VAD downstream
    tolerates missing tail samples but cannot handle a ragged final frame.
    """
    if frame_len <= 0 or hop_len <= 0:
        raise ValueError("frame_len and hop_len must be positive")
    audio = np.asarray(audio)
    n = audio.shape[0] if audio.ndim else 0
    if n < frame_len:
        return np.empty((0, frame_len), dtype=audio.dtype)
    n_frames = 1 + (n - frame_len) // hop_len
    idx = (
        np.arange(frame_len)[None, :]
        + hop_len * np.arange(n_frames)[:, None]
    )
    return audio[idx]
