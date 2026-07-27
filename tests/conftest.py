"""Shared pytest fixtures: synthetic audio, forced-MFCC verifier, temp DB path."""

from __future__ import annotations

import sys
import zlib
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

# Make src/ importable without an install step.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_sense.config import DEFAULT, Config  # noqa: E402
from speech_sense.database import SpeakerDatabase  # noqa: E402
from speech_sense.embedding import MfccBackend, SpeakerEncoder  # noqa: E402
from speech_sense.verifier import SpeakerVerifier  # noqa: E402

# The MFCC fallback produces very tight cosine similarities on synthetic tones
# (all in the 0.99+ range). Real backends have far wider margins. We relax the
# test-time thresholds so behavioural tests can still validate the pipeline.
TEST_CONFIG = Config(similarity_threshold=0.5, known_speaker_margin=0.001)


@pytest.fixture(scope="session")
def sr() -> int:
    return DEFAULT.sample_rate


@pytest.fixture(scope="session")
def synth_speech() -> Callable[..., np.ndarray]:
    """Deterministic tone-with-formants generator used as a stand-in for speech.

    The `speaker` parameter deterministically shifts formants and adds a
    per-speaker spectral fingerprint. Without this, MFCC (our test-time
    fallback backend) can't tell speakers apart since MFCC deliberately
    discards pitch information.
    """
    def make(
        f0: float = 180.0,
        seconds: float = 2.0,
        sr: int = DEFAULT.sample_rate,
        seed: int = 0,
        speaker: str = "default",
        modulate: bool = True,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        # crc32, not hash(): Python salts str hashing per process, so hash()
        # here would give every speaker a different voice on every run and make
        # the whole suite pass or fail depending on PYTHONHASHSEED.
        speaker_rng = np.random.default_rng(zlib.crc32(speaker.encode()))
        # Big per-speaker formant shifts so MFCC has real discriminative signal.
        formant_shift = speaker_rng.uniform(-350, 350, size=3)
        formant_amps = speaker_rng.uniform(0.15, 0.5, size=3)
        # Extra per-speaker high-frequency resonance for richer discrimination.
        extra_resonance = speaker_rng.uniform(3200, 4800)
        extra_amp = speaker_rng.uniform(0.05, 0.2)

        t = np.linspace(0.0, seconds, int(seconds * sr), endpoint=False)
        wave = np.zeros_like(t)
        for k, freq in enumerate([f0, 2 * f0, 3 * f0]):
            wave += (1.0 / (k + 1)) * np.sin(2 * np.pi * freq * t)
        for freq, amp in zip(np.array([700, 1220, 2600]) + formant_shift, formant_amps):
            wave += amp * np.sin(2 * np.pi * freq * t + speaker_rng.uniform(0, 2 * np.pi))
        wave += extra_amp * np.sin(2 * np.pi * extra_resonance * t)
        if modulate:
            wave *= 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
        wave += rng.normal(0, 0.01, size=wave.shape)
        wave = wave / (np.max(np.abs(wave)) + 1e-9) * 0.9
        return wave.astype(np.float32)
    return make


@pytest.fixture
def silence() -> np.ndarray:
    return np.zeros(DEFAULT.sample_rate, dtype=np.float32)


@pytest.fixture
def encoder_mfcc() -> SpeakerEncoder:
    """Always use the MFCC fallback so tests never depend on model downloads."""
    return SpeakerEncoder(MfccBackend())


@pytest.fixture
def verifier(encoder_mfcc: SpeakerEncoder) -> SpeakerVerifier:
    # use_vad=False keeps tests deterministic — synthetic tones aren't real speech.
    return SpeakerVerifier(
        encoder=encoder_mfcc, database=SpeakerDatabase(),
        config=TEST_CONFIG, use_vad=False,
    )


@pytest.fixture
def enrolled_verifier(verifier: SpeakerVerifier, synth_speech) -> SpeakerVerifier:
    """Pre-enroll three well-separated synthetic speakers."""
    for name, f0 in [("alice", 220.0), ("bob", 110.0), ("carol", 175.0)]:
        clips = [synth_speech(f0=f0, seconds=1.5, seed=i, speaker=name) for i in range(3)]
        verifier.enroll_from_audio(name, clips)
    return verifier


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> str:
    return str(tmp_path / "speakers.npz")
