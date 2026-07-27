"""Streaming identifier: buffering, gating, event emission."""

from __future__ import annotations

import numpy as np

from speech_sense.config import DEFAULT
from speech_sense.streaming import StreamingIdentifier, iter_file_chunks


def test_stream_emits_events_on_speech(enrolled_verifier, synth_speech):
    stream = StreamingIdentifier(verifier=enrolled_verifier, config=DEFAULT)
    audio = synth_speech(f0=220.0, seconds=3.0, seed=42, speaker="alice")
    chunks = iter_file_chunks(audio, DEFAULT.sample_rate // 2)  # 500 ms
    events = stream.process_iterable(chunks)
    assert len(events) >= 1
    assert any(e.result.speaker == "alice" for e in events)


def test_stream_silence_emits_no_events(enrolled_verifier):
    stream = StreamingIdentifier(verifier=enrolled_verifier, config=DEFAULT)
    silence = np.zeros(int(DEFAULT.sample_rate * 3), dtype=np.float32)
    events = stream.process_iterable(iter_file_chunks(silence, DEFAULT.sample_rate // 2))
    assert events == []


def test_stream_window_bounded(enrolled_verifier, synth_speech):
    stream = StreamingIdentifier(verifier=enrolled_verifier, config=DEFAULT)
    audio = synth_speech(seconds=10.0, seed=1)
    stream.process_iterable(iter_file_chunks(audio, DEFAULT.sample_rate))
    assert stream._buffer_samples <= stream._window_samples + DEFAULT.sample_rate


def test_stream_reset_clears_state(enrolled_verifier, synth_speech):
    stream = StreamingIdentifier(verifier=enrolled_verifier, config=DEFAULT)
    stream.process_iterable(iter_file_chunks(synth_speech(seconds=2.0), 8000))
    stream.reset()
    assert stream._buffer_samples == 0
    assert stream._elapsed_samples == 0
