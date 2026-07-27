"""Near-real-time streaming identification.

The `StreamingIdentifier` maintains a rolling window of audio; every time
`process_chunk` is called it appends the new samples, drops anything past the
window length, and — if the window contains enough speech and enough time has
passed since the last inference — runs identification.

Two use-cases share this class:

    * Wire it to sounddevice.InputStream for live microphone input.
    * Feed it fixed-size chunks from a pre-recorded file for offline evaluation
      of streaming behaviour.

`ponytail`: this is intentionally single-threaded. If throughput ever matters,
put the encoder call in a worker thread — the buffer maths stays the same.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np

from . import vad as vad_mod
from .config import DEFAULT, Config
from .verifier import SpeakerVerifier, VerificationResult


@dataclass
class StreamEvent:
    """A single identification event emitted by the streamer."""

    t_seconds: float
    result: VerificationResult


class StreamingIdentifier:
    def __init__(
        self,
        verifier: SpeakerVerifier,
        config: Config = DEFAULT,
        on_event: Optional[Callable[[StreamEvent], None]] = None,
    ) -> None:
        self.verifier = verifier
        self.config = config
        self.on_event = on_event
        self._buffer: deque[np.ndarray] = deque()
        self._buffer_samples: int = 0
        self._elapsed_samples: int = 0
        self._samples_since_infer: int = 0
        self._infer_stride = int(config.sample_rate * config.stream_chunk_ms / 1000)
        self._window_samples = int(config.sample_rate * config.stream_window_s)

    def reset(self) -> None:
        self._buffer.clear()
        self._buffer_samples = 0
        self._elapsed_samples = 0
        self._samples_since_infer = 0

    def _buffered_audio(self) -> np.ndarray:
        if not self._buffer:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(list(self._buffer))

    def process_chunk(self, chunk: np.ndarray) -> Optional[StreamEvent]:
        """Push a new audio chunk; may return an identification event or None."""
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return None

        self._buffer.append(chunk)
        self._buffer_samples += chunk.size
        self._elapsed_samples += chunk.size
        self._samples_since_infer += chunk.size

        # Drop oldest chunks until we fit inside the window.
        while self._buffer_samples > self._window_samples and len(self._buffer) > 1:
            head = self._buffer.popleft()
            self._buffer_samples -= head.size

        if self._samples_since_infer < self._infer_stride:
            return None
        self._samples_since_infer = 0

        window = self._buffered_audio()
        if not vad_mod.is_speech(window, self.config.sample_rate):
            return None

        result = self.verifier.identify(window)
        event = StreamEvent(
            t_seconds=self._elapsed_samples / self.config.sample_rate,
            result=result,
        )
        if self.on_event is not None:
            self.on_event(event)
        return event

    def process_iterable(self, chunks: Iterable[np.ndarray]) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        for c in chunks:
            e = self.process_chunk(c)
            if e is not None:
                events.append(e)
        return events


def iter_file_chunks(audio: np.ndarray, chunk_samples: int) -> Iterable[np.ndarray]:
    """Slice a waveform into fixed-size chunks (last chunk may be shorter)."""
    for start in range(0, len(audio), chunk_samples):
        yield audio[start : start + chunk_samples]
