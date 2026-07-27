"""Preprocessing pipeline: I/O, resampling, normalisation, noise injection."""

from __future__ import annotations

import numpy as np
import pytest

from speech_sense.audio import (
    add_white_noise,
    frame_signal,
    load_wav,
    normalize,
    resample,
    rms,
    save_wav,
)


def test_normalize_peak(synth_speech):
    x = synth_speech() * 0.01
    y = normalize(x, peak=0.9)
    assert abs(np.max(np.abs(y)) - 0.9) < 1e-6


def test_normalize_zero_stays_zero():
    z = np.zeros(1024, dtype=np.float32)
    assert np.array_equal(normalize(z), z)


def test_resample_length(synth_speech):
    x = synth_speech(seconds=1.0)  # 16000 samples @ 16k
    y = resample(x, 16000, 8000)
    assert abs(len(y) - 8000) <= 1


def test_resample_noop_returns_same_dtype(synth_speech):
    x = synth_speech(seconds=0.5)
    y = resample(x, 16000, 16000)
    assert y.dtype == np.float32


def test_add_white_noise_snr_close_to_target(synth_speech):
    x = synth_speech(seconds=1.0)
    signal_power = np.mean(x.astype(np.float64) ** 2)
    y = add_white_noise(x, snr_db=20.0)
    noise = y - x
    noise_power = np.mean(noise.astype(np.float64) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    assert 19.0 < snr < 21.0


def test_add_white_noise_empty_returns_empty():
    z = np.zeros(0, dtype=np.float32)
    assert add_white_noise(z, snr_db=10.0).size == 0


def test_frame_signal_shapes():
    audio = np.arange(1000, dtype=np.float32)
    frames = frame_signal(audio, frame_len=100, hop_len=50)
    assert frames.shape == (19, 100)
    # First frame is [0..99], second [50..149]
    np.testing.assert_array_equal(frames[0], audio[:100])
    np.testing.assert_array_equal(frames[1], audio[50:150])


def test_frame_signal_short_signal_returns_empty():
    audio = np.arange(50, dtype=np.float32)
    frames = frame_signal(audio, frame_len=100, hop_len=50)
    assert frames.shape == (0, 100)


def test_frame_signal_rejects_zero_hop():
    with pytest.raises(ValueError):
        frame_signal(np.zeros(100), 10, 0)


def test_rms_positive(synth_speech):
    x = synth_speech()
    assert rms(x) > 0


def test_wav_roundtrip(tmp_path, synth_speech):
    x = synth_speech(seconds=1.0)
    p = tmp_path / "clip.wav"
    save_wav(p, x)
    y = load_wav(p, target_sr=16000)
    # Not bit-exact due to PCM16 quantisation, but must be close in magnitude.
    assert len(y) == len(x)
    assert np.corrcoef(x, y)[0, 1] > 0.99


def test_load_audio_accepts_flac(tmp_path, synth_speech):
    """libsndfile handles FLAC out of the box — no ffmpeg needed."""
    import soundfile as sf
    from speech_sense.audio import load_audio

    x = synth_speech(seconds=1.0)
    p = tmp_path / "clip.flac"
    sf.write(p, x, 16000, format="FLAC")
    y = load_audio(p, target_sr=16000)
    assert len(y) == len(x)
    assert np.corrcoef(x, y)[0, 1] > 0.99


def test_load_audio_accepts_ogg(tmp_path, synth_speech):
    import soundfile as sf
    from speech_sense.audio import load_audio

    x = synth_speech(seconds=1.0)
    p = tmp_path / "clip.ogg"
    try:
        sf.write(p, x, 16000, format="OGG", subtype="VORBIS")
    except Exception:
        pytest.skip("libsndfile lacks OGG/Vorbis on this platform")
    y = load_audio(p, target_sr=16000)
    assert len(y) > 0


def test_load_audio_rejects_garbage_with_helpful_error(tmp_path):
    from speech_sense.audio import load_audio

    p = tmp_path / "not_audio.mp3"
    p.write_bytes(b"definitely not audio")
    with pytest.raises(ValueError, match="could not decode"):
        load_audio(p)
