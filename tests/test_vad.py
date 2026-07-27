"""VAD behavioural tests."""

from __future__ import annotations

import numpy as np

from speech_sense.vad import (
    frame_decisions,
    is_speech,
    speech_ratio,
    speech_segments,
    trim_silence,
)


def test_silence_returns_no_speech(silence):
    assert not is_speech(silence)
    assert speech_segments(silence) == []
    assert speech_ratio(silence) == 0.0


def test_speech_detected_in_synthetic_speech(synth_speech):
    x = synth_speech(seconds=1.0)
    assert is_speech(x)
    assert speech_ratio(x) > 0.4


def test_trim_silence_removes_leading_silence(synth_speech, sr):
    speech = synth_speech(seconds=1.0)
    padded = np.concatenate([np.zeros(sr, dtype=np.float32), speech, np.zeros(sr, dtype=np.float32)])
    trimmed = trim_silence(padded, sr)
    # Trimmed version must be shorter than 3s (originally) but at least 0.8s of speech.
    assert len(trimmed) < len(padded)
    assert len(trimmed) > 0.5 * sr


def test_trim_silence_on_silence_returns_original(silence, sr):
    result = trim_silence(silence, sr)
    # If no speech is detected we deliberately return the original signal
    # (the higher-level verifier decides what to do about it).
    assert np.array_equal(result, silence)


def test_frame_decisions_length(synth_speech):
    x = synth_speech(seconds=1.0)
    d = frame_decisions(x)
    # 1s @ 16k in 30 ms non-overlapping frames -> 33 frames.
    assert 30 <= d.size <= 35


def test_speech_segments_disallow_tiny_bursts():
    # A single high-energy sample surrounded by silence must not become a segment.
    audio = np.zeros(16000, dtype=np.float32)
    audio[8000:8005] = 1.0
    assert speech_segments(audio) == []
