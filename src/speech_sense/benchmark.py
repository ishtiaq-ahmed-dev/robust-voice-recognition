"""Latency + size benchmarking for the speaker encoder.

Runs `iterations` embeddings over `seconds` of synthetic speech, reports median
+ p95 latency and derived Real-Time Factor (RTF). Model size is inferred from
the imported backend's params.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import numpy as np

from .config import DEFAULT
from .embedding import SpeakerEncoder


def _synth_speech(seconds: float, sr: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    audio = np.sin(2 * np.pi * 180 * t) * 0.5
    audio += 0.3 * np.sin(2 * np.pi * 720 * t)
    audio += 0.1 * rng.normal(size=audio.shape)
    return audio.astype(np.float32)


def _backend_size_mb(encoder: SpeakerEncoder) -> float | None:
    """Best-effort estimate of the loaded backend's on-disk / in-memory size."""
    try:
        import torch

        model = getattr(encoder.backend, "_encoder", None)
        if model is None:
            return None
        params = sum(p.numel() * p.element_size() for p in model.parameters() if isinstance(p, torch.Tensor))
        buffers = sum(b.numel() * b.element_size() for b in model.buffers() if isinstance(b, torch.Tensor))
        return round((params + buffers) / 1_048_576, 2)
    except Exception:
        return None


def run_benchmark(
    backend: str = "auto",
    seconds: float = 3.0,
    iterations: int = 20,
    sample_rate: int = DEFAULT.sample_rate,
) -> dict[str, Any]:
    """Return a dict of latency + throughput measurements."""
    encoder = SpeakerEncoder.load(backend)  # type: ignore[arg-type]
    audio = _synth_speech(seconds, sample_rate)

    # Warm up (JIT / caching) — 2 untimed runs.
    for _ in range(2):
        encoder.embed(audio, sample_rate)

    times: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        encoder.embed(audio, sample_rate)
        times.append(time.perf_counter() - t0)

    median_s = statistics.median(times)
    p95_s = sorted(times)[int(0.95 * (len(times) - 1))]
    return {
        "backend": encoder.name,
        "embedding_dim": encoder.dim,
        "audio_seconds": seconds,
        "iterations": iterations,
        "latency_median_ms": round(median_s * 1000, 2),
        "latency_p95_ms": round(p95_s * 1000, 2),
        "throughput_clips_per_s": round(1.0 / median_s, 2),
        "real_time_factor": round(median_s / seconds, 4),
        "model_size_mb": _backend_size_mb(encoder),
    }
