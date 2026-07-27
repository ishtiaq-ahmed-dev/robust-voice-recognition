"""Regression tests for the failure modes that make a service fall over.

Each test here pins a specific bug that was found and fixed. They are grouped
by the symptom an operator would actually see, not by module.
"""

from __future__ import annotations

import io
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from speech_sense.api import create_app
from speech_sense.audio import load_audio, resample, sanitize
from speech_sense.config import Config
from speech_sense.database import SpeakerDatabase
from speech_sense.embedding import MfccBackend, SpeakerEncoder
from speech_sense.vad import is_speech, speech_ratio
from speech_sense.verifier import SpeakerVerifier
from tests.conftest import TEST_CONFIG


def _wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


# ── VAD: the "all clips were silence" false negative ────────────────────────

def test_steady_level_speech_is_not_reported_as_silence(sr):
    """The bug: a clip recorded at a constant level was classified as silence.

    The old rule was `energy > 1.5 * percentile(energies, 30)`. When every
    frame carries the same energy the 30th percentile *is* that energy, so
    nothing cleared 1.5x it, every frame was rejected, and enrolment failed
    with "all clips were silence" on a perfectly good recording.
    """
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
    steady = (0.5 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)  # no envelope at all

    assert is_speech(steady, sr), "constant-level speech must be detected"
    assert speech_ratio(steady, sr) > 0.95


def test_digital_silence_is_still_rejected(sr):
    assert not is_speech(np.zeros(2 * sr, dtype=np.float32), sr)


def test_quiet_but_real_speech_is_detected(sr):
    """Peak-relative gating must not depend on absolute loudness."""
    t = np.linspace(0, 1.5, int(1.5 * sr), endpoint=False)
    quiet = (0.02 * np.sin(2 * np.pi * 200 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    assert is_speech(quiet, sr)


# ── Verifier: backend/database dimension mismatch ───────────────────────────

def test_backend_dimension_mismatch_fails_closed_instead_of_crashing(encoder_mfcc, synth_speech):
    """A database enrolled under one backend, queried under another.

    `mat @ query` raised ValueError, so every single /identify returned 500
    until someone deleted the database. It must fail closed with a reason.
    """
    verifier = SpeakerVerifier(encoder=encoder_mfcc, config=TEST_CONFIG, use_vad=False)
    # 256-D vectors, as resemblyzer would have written; the encoder is 78-D.
    verifier.database.add("alice", np.ones(256, dtype=np.float32) / 16.0)

    result = verifier.identify(synth_speech())
    assert result.is_known is False
    assert result.speaker is None
    assert "re-enrol" in result.reason
    assert "256-D" in result.reason


# ── Verifier: unbounded score disclosure ────────────────────────────────────

def test_scores_are_capped_at_top_k(encoder_mfcc, synth_speech):
    """Returning every enrolled speaker leaks the roster and grows unboundedly."""
    config = Config(similarity_threshold=0.5, known_speaker_margin=0.001, top_k_scores=3)
    verifier = SpeakerVerifier(encoder=encoder_mfcc, config=config, use_vad=False)
    for i in range(10):
        verifier.enroll_from_audio(f"person_{i}", [synth_speech(f0=100 + 15 * i, speaker=f"p{i}")])

    result = verifier.identify(synth_speech(f0=100, speaker="p0"))
    assert len(result.scores) == 3
    # The cap must keep the *highest* scores, not an arbitrary three.
    assert result.scores == dict(sorted(result.scores.items(), key=lambda kv: -kv[1]))


def test_verify_reports_the_claimed_speaker_even_outside_the_top_k(encoder_mfcc, synth_speech):
    """Truncating scores must not make a legitimate claim unscoreable."""
    config = Config(similarity_threshold=0.99, known_speaker_margin=0.001, top_k_scores=1)
    verifier = SpeakerVerifier(encoder=encoder_mfcc, config=config, use_vad=False)
    for i in range(5):
        verifier.enroll_from_audio(f"person_{i}", [synth_speech(f0=100 + 20 * i, speaker=f"p{i}")])

    result = verifier.verify(synth_speech(f0=180, speaker="p4"), claimed="person_0")
    assert "person_0" in result.scores
    assert isinstance(result.similarity, float)


def test_verify_preserves_the_no_speech_reason(encoder_mfcc, silence):
    """Silence used to be reported as 'similarity below threshold'."""
    verifier = SpeakerVerifier(encoder=encoder_mfcc, config=TEST_CONFIG, use_vad=True)
    verifier.database.add("alice", np.ones(encoder_mfcc.dim, dtype=np.float32) / 8.0)
    result = verifier.verify(silence, claimed="alice")
    assert result.contains_speech is False
    assert "no speech" in result.reason


# ── NaN propagation ─────────────────────────────────────────────────────────

def test_non_finite_audio_never_reaches_the_json_response(encoder_mfcc, synth_speech):
    """A single NaN sample used to serialise as the bare token `NaN`.

    That is not valid JSON and breaks any strict client parsing the response.
    """
    import json

    verifier = SpeakerVerifier(encoder=encoder_mfcc, config=TEST_CONFIG, use_vad=False)
    verifier.enroll_from_audio("alice", [synth_speech(speaker="alice")])

    poisoned = synth_speech(speaker="alice").copy()
    poisoned[100:110] = np.nan
    poisoned[200] = np.inf

    payload = json.dumps(verifier.identify(poisoned).to_dict())
    assert "NaN" not in payload and "Infinity" not in payload
    assert json.loads(payload) is not None


def test_sanitize_does_not_mutate_the_callers_array():
    original = np.array([1.0, np.nan, -2.0], dtype=np.float32)
    copy = original.copy()
    sanitize(original)
    np.testing.assert_array_equal(original, copy, "sanitize must not write through")


# ── Audio guards ────────────────────────────────────────────────────────────

def test_resample_rejects_a_zero_sample_rate():
    """A decoder reporting sr=0 used to reach gcd() and raise ZeroDivisionError."""
    with pytest.raises(ValueError, match="positive"):
        resample(np.zeros(100, dtype=np.float32), 0, 16000)


def test_missing_file_says_so_rather_than_blaming_ffmpeg(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such audio file"):
        load_audio(tmp_path / "typo.wav")


# ── Embedding: short clips ──────────────────────────────────────────────────

def test_mfcc_backend_survives_a_clip_shorter_than_the_delta_window(sr):
    """librosa.feature.delta raises when its window exceeds the frame count.

    A sub-second upload used to crash the encoder outright.
    """
    encoder = SpeakerEncoder(MfccBackend())
    for duration in (0.05, 0.1, 0.2, 0.5):
        clip = np.sin(np.linspace(0, 50, int(sr * duration))).astype(np.float32)
        emb = encoder.embed(clip, sr)
        assert emb.shape == (encoder.dim,)
        assert np.all(np.isfinite(emb))


# ── API: request limits ─────────────────────────────────────────────────────

@pytest.fixture
def limited_client(tmp_path):
    config = Config(
        similarity_threshold=0.5, known_speaker_margin=0.001,
        max_upload_bytes=2048, max_files_per_request=2, max_audio_seconds=1.0,
    )
    app = create_app(
        database_path=str(tmp_path / "db.npz"),
        backend="mfcc", use_vad=False, config=config,
    )
    with TestClient(app) as client:
        yield client


def test_oversize_upload_is_refused(limited_client, synth_speech):
    big = _wav_bytes(synth_speech(seconds=3.0))
    response = limited_client.post("/identify", files={"file": ("big.wav", big, "audio/wav")})
    assert response.status_code == 413
    assert "limit" in response.json()["detail"]


def test_too_many_files_are_refused(limited_client, synth_speech):
    tiny = _wav_bytes(synth_speech(seconds=0.05))
    files = [("files", (f"{i}.wav", tiny, "audio/wav")) for i in range(3)]
    response = limited_client.post("/enroll", data={"name": "alice"}, files=files)
    assert response.status_code == 413


def test_overlong_audio_is_refused_even_when_the_bytes_are_small(tmp_path, synth_speech):
    """Compressed audio expands: a byte cap alone does not bound the work."""
    config = Config(similarity_threshold=0.5, known_speaker_margin=0.001, max_audio_seconds=0.5)
    app = create_app(
        database_path=str(tmp_path / "db.npz"),
        backend="mfcc", use_vad=False, config=config,
    )
    with TestClient(app) as client:
        payload = _wav_bytes(synth_speech(seconds=2.0))
        response = client.post("/identify", files={"file": ("long.wav", payload, "audio/wav")})
        assert response.status_code == 413
        assert "longer than" in response.json()["detail"]


def test_empty_upload_is_rejected(limited_client):
    response = limited_client.post("/identify", files={"file": ("empty.wav", b"", "audio/wav")})
    assert response.status_code == 400


# ── API: authentication ─────────────────────────────────────────────────────

@pytest.fixture
def secured_client(tmp_path):
    config = Config(similarity_threshold=0.5, known_speaker_margin=0.001, api_key="s3cret-key")
    app = create_app(
        database_path=str(tmp_path / "db.npz"),
        backend="mfcc", use_vad=False, config=config,
    )
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("method,path", [
    ("get", "/speakers"),
    ("get", "/analytics"),
    ("get", "/employees"),
    ("delete", "/speakers/alice"),
    ("post", "/reload"),
])
def test_protected_routes_require_the_api_key(secured_client, method, path):
    assert getattr(secured_client, method)(path).status_code == 401


@pytest.mark.parametrize("path", ["/healthz", "/readyz", "/prompts"])
def test_probe_routes_stay_open(secured_client, path):
    """Liveness probes must work without a credential or the orchestrator
    cannot tell a locked-down service from a dead one."""
    assert secured_client.get(path).status_code == 200


def test_correct_api_key_is_accepted(secured_client):
    response = secured_client.get("/speakers", headers={"X-API-Key": "s3cret-key"})
    assert response.status_code == 200


def test_wrong_api_key_is_rejected(secured_client):
    response = secured_client.get("/speakers", headers={"X-API-Key": "s3cret-keZ"})
    assert response.status_code == 401


def test_no_key_configured_means_open_access(tmp_path):
    app = create_app(
        database_path=str(tmp_path / "db.npz"),
        backend="mfcc", use_vad=False, config=TEST_CONFIG,
    )
    with TestClient(app) as client:
        assert client.get("/speakers").status_code == 200
        assert client.get("/healthz").json()["auth_required"] is False


# ── API: degraded startup ───────────────────────────────────────────────────

def test_engine_failure_degrades_instead_of_crash_looping(tmp_path, monkeypatch):
    """If the model cannot load, the process must still answer probes.

    Raising out of lifespan takes the whole app down, which in a container
    means a restart loop whose only diagnostic is the exit code.
    """
    def explode(_backend="auto"):
        raise RuntimeError("model weights unavailable")

    monkeypatch.setattr(SpeakerEncoder, "load", staticmethod(explode))
    app = create_app(database_path=str(tmp_path / "db.npz"), backend="mfcc", config=TEST_CONFIG)

    with TestClient(app) as client:
        health = client.get("/healthz")
        assert health.status_code == 503
        assert health.json()["status"] == "degraded"
        assert "model weights unavailable" in health.json()["detail"]

        assert client.get("/readyz").status_code == 503
        # Routes needing the engine report 503, not 500.
        assert client.get("/speakers").status_code == 503
        # Static content still works, so the dashboard can show the error.
        assert client.get("/prompts").status_code == 200


# ── API: no traceback leakage ───────────────────────────────────────────────

def test_unhandled_errors_return_a_request_id_not_a_traceback(tmp_path, monkeypatch, synth_speech):
    app = create_app(
        database_path=str(tmp_path / "db.npz"),
        backend="mfcc", use_vad=False, config=TEST_CONFIG,
    )

    def boom(self, audio):
        raise RuntimeError("secret internal detail /etc/passwd")

    monkeypatch.setattr(SpeakerVerifier, "identify", boom)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/identify",
            files={"file": ("q.wav", _wav_bytes(synth_speech()), "audio/wav")},
        )
    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "internal server error"
    assert "secret internal detail" not in response.text
    assert "Traceback" not in response.text
    assert len(body["request_id"]) == 12


def test_security_headers_are_set(tmp_path):
    app = create_app(
        database_path=str(tmp_path / "db.npz"),
        backend="mfcc", use_vad=False, config=TEST_CONFIG,
    )
    with TestClient(app) as client:
        headers = client.get("/healthz").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert headers["X-Request-ID"]


# ── API: speaker name validation ────────────────────────────────────────────

@pytest.mark.parametrize("name,expected", [
    ("   ", "must not be empty"),
    ("x" * 500, "at most"),
    ("bad\x00name", "control or text-direction"),
    ("‮evil", "control or text-direction"),          # right-to-left override
])
def test_invalid_speaker_names_are_rejected(tmp_path, synth_speech, name, expected):
    app = create_app(
        database_path=str(tmp_path / "db.npz"),
        backend="mfcc", use_vad=False, config=TEST_CONFIG,
    )
    with TestClient(app) as client:
        response = client.post(
            "/enroll",
            data={"name": name},
            files={"files": ("a.wav", _wav_bytes(synth_speech()), "audio/wav")},
        )
    assert response.status_code == 400
    assert expected in response.json()["detail"]


def test_missing_name_field_is_a_client_error(tmp_path, synth_speech):
    """An empty value is dropped by the multipart encoder, so the framework
    reports a missing field (422) before our validator ever sees it."""
    app = create_app(
        database_path=str(tmp_path / "db.npz"),
        backend="mfcc", use_vad=False, config=TEST_CONFIG,
    )
    with TestClient(app) as client:
        response = client.post(
            "/enroll",
            data={"name": ""},
            files={"files": ("a.wav", _wav_bytes(synth_speech()), "audio/wav")},
        )
    assert response.status_code == 422


def test_markup_in_a_speaker_name_is_stored_verbatim(tmp_path, synth_speech):
    """The server stores it as data; the dashboard renders it via textContent.

    Rejecting angle brackets would be the wrong fix — escaping belongs at the
    point of rendering, not the point of storage.
    """
    app = create_app(
        database_path=str(tmp_path / "db.npz"),
        backend="mfcc", use_vad=False, config=TEST_CONFIG,
    )
    payload = '<img src=x onerror="alert(1)">'
    with TestClient(app) as client:
        response = client.post(
            "/enroll",
            data={"name": payload},
            files={"files": ("a.wav", _wav_bytes(synth_speech()), "audio/wav")},
        )
        assert response.status_code == 200
        assert response.json()["enrolled"] == payload
        assert payload in client.get("/speakers").json()["speakers"]


# ── Persistence survives a restart with a mid-write crash ───────────────────

def test_a_failed_save_leaves_the_previous_database_intact(tmp_path, monkeypatch):
    """Atomicity: readers see the old file or the new one, never a stub."""
    path = tmp_path / "db.npz"
    db = SpeakerDatabase()
    db.add("alice", np.ones(8, dtype=np.float32))
    db.save(path)
    good = path.read_bytes()

    db.add("bob", np.zeros(8, dtype=np.float32))
    real_replace = __import__("os").replace

    def fail_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr("os.replace", fail_replace)
    with pytest.raises(OSError):
        db.save(path)
    monkeypatch.setattr("os.replace", real_replace)

    assert path.read_bytes() == good, "the previous database must survive a failed write"
    assert SpeakerDatabase.load(path).names() == ["alice"]
    assert list(tmp_path.glob("*.tmp")) == [], "temp file must be cleaned up"
