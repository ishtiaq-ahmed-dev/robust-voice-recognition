"""Energy + zero-crossing rate Voice Activity Detection.

No new dependencies — uses only numpy. Silero/webrtcvad are strictly better
for adversarial acoustics, but they're heavyweight installs and this VAD is
tuned to be conservative (favour rejecting silence over rejecting speech).

The public API is deliberately small:

    is_speech(audio, sr)           -> bool
    trim_silence(audio, sr)        -> np.ndarray
    speech_segments(audio, sr)     -> list[tuple[int, int]]   # sample indices
    extract_speech(audio, sr)      -> np.ndarray | None       # single-pass

**Threshold model.** A frame is speech when its RMS clears a fraction of the
*loudest* frame in the clip, and also clears a small absolute floor. An earlier
version compared each frame against `1.5 * percentile(energies, 30)`, which has
a nasty degenerate case: in a clip recorded at a steady level the 30th
percentile is itself speech energy, nothing clears 1.5x it, and a perfectly
good recording is reported as pure silence — enrolment then fails with "all
clips were silence". Peak-relative gating has no such fixed point: a uniform
clip is entirely speech, and a clip with real pauses still drops them.

`ponytail`: this VAD is coarse — its ceiling is that it is purely energetic,
so sustained non-speech noise at speech-like level and low zero-crossing rate
(a hum, an engine) reads as speech. Upgrade path is to swap `frame_decisions`
for Silero's neural VAD probabilities; every function below is expressed in
terms of that one array, so nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import frame_signal, sanitize
from .config import DEFAULT


@dataclass(frozen=True)
class VADParams:
    frame_ms: int = DEFAULT.vad_frame_ms
    min_speech_ms: int = DEFAULT.vad_min_speech_ms
    max_silence_ms: int = DEFAULT.vad_max_silence_ms
    # Frames quieter than this fraction of the loudest frame are non-speech
    # (0.05 ≈ 26 dB below peak).
    speech_floor_ratio: float = DEFAULT.vad_speech_floor_ratio
    # …and regardless of the peak, a frame this quiet is digital silence.
    abs_energy_floor: float = DEFAULT.vad_abs_energy_floor
    # Frames with a zero-crossing rate above this look like broadband noise.
    zcr_max: float = DEFAULT.vad_zcr_max


def _frame_energy(frames: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))


def _frame_zcr(frames: np.ndarray) -> np.ndarray:
    signs = np.sign(frames)
    signs[signs == 0] = 1.0
    return np.mean(np.abs(np.diff(signs, axis=1)) > 0, axis=1)


def frame_decisions(
    audio: np.ndarray,
    sr: int = DEFAULT.sample_rate,
    params: VADParams | None = None,
) -> np.ndarray:
    """Return a bool array marking each frame as speech (True) or non-speech."""
    if sr <= 0:
        raise ValueError(f"sample rate must be positive, got {sr}")
    p = params or VADParams()
    audio = sanitize(audio)
    frame_len = max(1, int(sr * p.frame_ms / 1000))
    frames = frame_signal(audio, frame_len, frame_len)  # non-overlapping — cheap and adequate
    if frames.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    energies = _frame_energy(frames)
    zcrs = _frame_zcr(frames)

    peak = float(energies.max())
    threshold = max(peak * p.speech_floor_ratio, p.abs_energy_floor)
    # `>=` matters: in a clip of perfectly constant level every frame *is* the
    # peak, and a strict `>` would reject all of them.
    return (energies >= threshold) & (zcrs < p.zcr_max)


def speech_segments(
    audio: np.ndarray,
    sr: int = DEFAULT.sample_rate,
    params: VADParams | None = None,
) -> list[tuple[int, int]]:
    """Merge frame-level decisions into (start_sample, end_sample) speech runs."""
    p = params or VADParams()
    decisions = frame_decisions(audio, sr, p)
    if decisions.size == 0:
        return []

    frame_len = max(1, int(sr * p.frame_ms / 1000))
    min_speech_frames = max(1, p.min_speech_ms // p.frame_ms)
    max_silence_frames = max(1, p.max_silence_ms // p.frame_ms)

    segments: list[tuple[int, int]] = []
    in_speech = False
    seg_start = 0
    silence_run = 0

    for i, is_speech_frame in enumerate(decisions):
        if is_speech_frame:
            if not in_speech:
                seg_start = i
                in_speech = True
            silence_run = 0
        elif in_speech:
            silence_run += 1
            if silence_run >= max_silence_frames:
                seg_end = i - silence_run + 1
                if seg_end - seg_start >= min_speech_frames:
                    segments.append((seg_start * frame_len, seg_end * frame_len))
                in_speech = False
                silence_run = 0

    if in_speech:
        seg_end = len(decisions) - silence_run
        if seg_end - seg_start >= min_speech_frames:
            segments.append((seg_start * frame_len, seg_end * frame_len))

    return segments


def is_speech(
    audio: np.ndarray,
    sr: int = DEFAULT.sample_rate,
    params: VADParams | None = None,
) -> bool:
    """True if the clip contains at least one confirmed speech segment."""
    return len(speech_segments(audio, sr, params)) > 0


def _join_segments(
    audio: np.ndarray,
    segments: list[tuple[int, int]],
    sr: int,
    pad_ms: int,
) -> np.ndarray:
    pad = int(sr * pad_ms / 1000)
    n = len(audio)
    pieces = [audio[max(0, s - pad) : min(n, e + pad)] for s, e in segments]
    return np.concatenate(pieces).astype(np.float32)


def trim_silence(
    audio: np.ndarray,
    sr: int = DEFAULT.sample_rate,
    params: VADParams | None = None,
    pad_ms: int = 100,
) -> np.ndarray:
    """Return the concatenation of detected speech segments with small padding.

    If no speech is detected, returns the original waveform unchanged — the
    caller should decide whether that empty-input case is a hard error (in the
    verifier it is; in enrollment we surface it as a friendlier message).
    """
    audio = sanitize(audio)
    segments = speech_segments(audio, sr, params)
    if not segments:
        return audio
    return _join_segments(audio, segments, sr, pad_ms)


def extract_speech(
    audio: np.ndarray,
    sr: int = DEFAULT.sample_rate,
    params: VADParams | None = None,
    pad_ms: int = 100,
) -> np.ndarray | None:
    """Trimmed speech, or None when the clip contains none.

    One pass. `trim_silence` followed by `is_speech` computes the same frame
    decisions twice and still can't distinguish "returned the whole clip
    because it is all speech" from "returned the whole clip because there was
    no speech to trim" — this collapses both problems.
    """
    audio = sanitize(audio)
    segments = speech_segments(audio, sr, params)
    if not segments:
        return None
    return _join_segments(audio, segments, sr, pad_ms)


def speech_ratio(
    audio: np.ndarray,
    sr: int = DEFAULT.sample_rate,
    params: VADParams | None = None,
) -> float:
    """Fraction of the input classified as speech (0.0-1.0). Handy for logging."""
    decisions = frame_decisions(audio, sr, params)
    if decisions.size == 0:
        return 0.0
    return float(decisions.mean())
