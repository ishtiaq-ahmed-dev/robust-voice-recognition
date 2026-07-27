"""FastAPI service tests — exercised via TestClient (no real network)."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from speech_sense.api import create_app
from tests.conftest import TEST_CONFIG


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(
        database_path=str(tmp_path / "speakers.npz"),
        backend="mfcc", use_vad=False, config=TEST_CONFIG,
    )
    with TestClient(app) as c:
        yield c


def _wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backend"] == "mfcc"
    assert body["n_speakers"] == 0


def test_empty_speakers(client):
    r = client.get("/speakers")
    assert r.status_code == 200
    assert r.json() == {"speakers": [], "count": 0}


def test_enroll_and_identify_flow(client, synth_speech):
    # Enroll alice (f0=220).
    files = [
        ("files", (f"take_{i}.wav", _wav_bytes(synth_speech(f0=220.0, seed=i, speaker="alice")), "audio/wav"))
        for i in range(3)
    ]
    r = client.post("/enroll", data={"name": "alice"}, files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["enrolled"] == "alice"
    assert body["clips"] == 3

    # Enroll bob (f0=110).
    files = [
        ("files", (f"take_{i}.wav", _wav_bytes(synth_speech(f0=110.0, seed=i + 10, speaker="bob")), "audio/wav"))
        for i in range(3)
    ]
    r = client.post("/enroll", data={"name": "bob"}, files=files)
    assert r.status_code == 200

    # Identify a fresh alice clip.
    r = client.post(
        "/identify",
        files={"file": ("query.wav", _wav_bytes(synth_speech(f0=220.0, seed=99, speaker="alice")), "audio/wav")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["speaker"] == "alice"
    assert body["is_known"] is True
    assert 0.0 <= body["confidence"] <= 1.0


def test_verify_endpoint(client, synth_speech):
    files = [
        ("files", (f"a.wav", _wav_bytes(synth_speech(f0=220.0, seed=i, speaker="alice")), "audio/wav"))
        for i in range(3)
    ]
    client.post("/enroll", data={"name": "alice"}, files=files)

    r = client.post(
        "/verify",
        data={"name": "alice"},
        files={"file": ("query.wav", _wav_bytes(synth_speech(f0=220.0, seed=99, speaker="alice")), "audio/wav")},
    )
    assert r.status_code == 200
    assert r.json()["is_known"] is True

    r = client.post(
        "/verify",
        data={"name": "ghost"},
        files={"file": ("query.wav", _wav_bytes(synth_speech(f0=220.0, seed=99, speaker="alice")), "audio/wav")},
    )
    assert r.status_code == 200
    assert r.json()["is_known"] is False


def test_delete_speaker(client, synth_speech):
    files = [
        ("files", (f"a.wav", _wav_bytes(synth_speech(f0=220.0, seed=i, speaker="alice")), "audio/wav"))
        for i in range(3)
    ]
    client.post("/enroll", data={"name": "alice"}, files=files)
    r = client.delete("/speakers/alice")
    assert r.status_code == 200
    assert r.json()["deleted"] == "alice"
    r = client.delete("/speakers/alice")
    assert r.status_code == 404


def test_enroll_rejects_empty_name(client, synth_speech):
    r = client.post(
        "/enroll",
        data={"name": "   "},
        files={"files": ("a.wav", _wav_bytes(synth_speech()), "audio/wav")},
    )
    assert r.status_code == 400


def test_identify_rejects_invalid_audio(client):
    r = client.post("/identify", files={"file": ("junk.wav", b"not really a wav", "audio/wav")})
    assert r.status_code == 400
    assert "decode" in r.json()["detail"]


def test_identify_empty_db_reports_missing_speakers(client, synth_speech):
    r = client.post(
        "/identify",
        files={"file": ("query.wav", _wav_bytes(synth_speech()), "audio/wav")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["speaker"] is None
    assert not body["is_known"]


def test_identify_silence_flags_no_speech(tmp_path: Path, synth_speech):
    # Use a VAD-enabled client for this test specifically — silence rejection
    # is the whole point.
    app = create_app(database_path=str(tmp_path / "vad_db.npz"), backend="mfcc", use_vad=True)
    with TestClient(app) as c:
        # Seed the database directly, bypassing enrollment (which would fail
        # VAD on our synthetic tones). This is the exact scenario we want to
        # test: known speakers exist, incoming audio is silence.
        import numpy as np
        from speech_sense.database import SpeakerDatabase

        db = SpeakerDatabase.load(str(tmp_path / "vad_db.npz"))
        db.add("alice", np.random.RandomState(0).randn(78).astype(np.float32))
        db.save(str(tmp_path / "vad_db.npz"))
        c.post("/reload")

        r = c.post(
            "/identify",
            files={"file": ("silence.wav", _wav_bytes(np.zeros(16000, dtype=np.float32)), "audio/wav")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["is_known"] is False
        assert body["contains_speech"] is False


def test_root_serves_web_demo(client):
    r = client.get("/")
    assert r.status_code == 200
    # Any of the expected page markers is enough; page title changed with the UI redesign.
    body = r.text.lower()
    assert "robust voice recognition" in body or "speech_sense" in body


def test_batch_identify_endpoint(client, synth_speech):
    files = [
        ("files", (f"a.wav", _wav_bytes(synth_speech(f0=220.0, seed=i, speaker="alice")), "audio/wav"))
        for i in range(3)
    ]
    client.post("/enroll", data={"name": "alice"}, files=files)

    batch = [
        ("files", ("q1.wav", _wav_bytes(synth_speech(f0=220.0, seed=99, speaker="alice")), "audio/wav")),
        ("files", ("q2.wav", _wav_bytes(synth_speech(f0=220.0, seed=100, speaker="alice")), "audio/wav")),
    ]
    r = client.post("/batch/identify", files=batch)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 2
    assert body["results"][0]["filename"] == "q1.wav"
    for row in body["results"]:
        assert row["is_known"] is True
        assert row["speaker"] == "alice"


def test_batch_enroll_endpoint(client, synth_speech):
    files = [
        ("files", (f"a_{i}.wav", _wav_bytes(synth_speech(f0=220.0, seed=i, speaker="alice")), "audio/wav"))
        for i in range(4)
    ]
    r = client.post("/batch/enroll", data={"name": "alice"}, files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["enrolled"] == "alice"
    assert body["clips_used"] == 4


def test_batch_enroll_skips_bad_but_uses_good(client, synth_speech):
    files = [
        ("files", ("good.wav", _wav_bytes(synth_speech(f0=220.0, seed=1, speaker="alice")), "audio/wav")),
        ("files", ("bad.wav", b"junk", "audio/wav")),
    ]
    r = client.post("/batch/enroll", data={"name": "alice"}, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["clips_used"] == 1
    assert len(body["skipped"]) == 1


def test_batch_enroll_all_bad_returns_400(client):
    files = [
        ("files", ("bad1.wav", b"garbage", "audio/wav")),
        ("files", ("bad2.wav", b"more garbage", "audio/wav")),
    ]
    r = client.post("/batch/enroll", data={"name": "alice"}, files=files)
    assert r.status_code == 400


def test_employees_endpoint_lists_empty_when_no_folder(tmp_path: Path, monkeypatch, synth_speech):
    monkeypatch.setenv("SPEECH_SENSE_EMPLOYEES_AUDIO", str(tmp_path / "no_such_dir"))
    app = create_app(database_path=str(tmp_path / "db.npz"), backend="mfcc", use_vad=False, config=TEST_CONFIG)
    with TestClient(app) as c:
        r = c.get("/employees")
        assert r.status_code == 200
        body = r.json()
        assert body["exists"] is False
        assert body["employees"] == []


def test_employees_endpoint_lists_folder_contents(tmp_path: Path, monkeypatch, synth_speech):
    root = tmp_path / "emp_audio"
    for name, f0 in [("alice", 220.0), ("bob", 110.0)]:
        sp = root / name
        sp.mkdir(parents=True)
        for i in range(2):
            from speech_sense.audio import save_wav
            save_wav(sp / f"clip_{i}.wav", synth_speech(f0=f0, seed=i, speaker=name))
    monkeypatch.setenv("SPEECH_SENSE_EMPLOYEES_AUDIO", str(root))
    app = create_app(database_path=str(tmp_path / "db.npz"), backend="mfcc", use_vad=False, config=TEST_CONFIG)
    with TestClient(app) as c:
        r = c.get("/employees")
        assert r.status_code == 200
        body = r.json()
        assert body["exists"] is True
        names = {e["name"] for e in body["employees"]}
        assert names == {"alice", "bob"}
        assert all(e["clips"] == 2 for e in body["employees"])


def test_employees_enroll_all_endpoint(tmp_path: Path, monkeypatch, synth_speech):
    root = tmp_path / "emp_audio"
    for name, f0 in [("alice", 220.0), ("bob", 110.0), ("carol", 175.0)]:
        sp = root / name
        sp.mkdir(parents=True)
        for i in range(2):
            from speech_sense.audio import save_wav
            save_wav(sp / f"clip_{i}.wav", synth_speech(f0=f0, seed=i, speaker=name))
    monkeypatch.setenv("SPEECH_SENSE_EMPLOYEES_AUDIO", str(root))
    app = create_app(database_path=str(tmp_path / "db.npz"), backend="mfcc", use_vad=False, config=TEST_CONFIG)
    with TestClient(app) as c:
        r = c.post("/employees/enroll-all")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n_speakers"] == 3
        assert body["n_clips"] == 6
        speakers = c.get("/speakers").json()["speakers"]
        assert set(speakers) == {"alice", "bob", "carol"}


def test_employees_enroll_all_missing_dir_returns_404(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SPEECH_SENSE_EMPLOYEES_AUDIO", str(tmp_path / "nope"))
    app = create_app(database_path=str(tmp_path / "db.npz"), backend="mfcc", use_vad=False, config=TEST_CONFIG)
    with TestClient(app) as c:
        r = c.post("/employees/enroll-all")
        assert r.status_code == 404


def test_persistence_across_new_client(tmp_path: Path, synth_speech):
    """A restart must not lose enrollments (regression for pickle -> npz migration)."""
    db_path = tmp_path / "speakers.npz"
    app1 = create_app(database_path=str(db_path), backend="mfcc", use_vad=False, config=TEST_CONFIG)
    with TestClient(app1) as c:
        files = [
            ("files", (f"a.wav", _wav_bytes(synth_speech(f0=220.0, seed=i)), "audio/wav"))
            for i in range(3)
        ]
        assert c.post("/enroll", data={"name": "alice"}, files=files).status_code == 200

    app2 = create_app(database_path=str(db_path), backend="mfcc", use_vad=False, config=TEST_CONFIG)
    with TestClient(app2) as c:
        r = c.get("/speakers")
        assert r.status_code == 200
        assert "alice" in r.json()["speakers"]


def test_analytics_endpoint(client, synth_speech):
    """Analytics should start at zero, then increment after an identify."""
    r = client.get("/analytics")
    assert r.status_code == 200
    body = r.json()
    assert body["speakers_enrolled"] == 0
    assert body["total_identifications"] == 0
    assert body["total_operations"] == 0

    # Enroll alice then identify — counters should tick up.
    files = [
        ("files", (f"a.wav", _wav_bytes(synth_speech(f0=220.0, seed=i, speaker="alice")), "audio/wav"))
        for i in range(3)
    ]
    client.post("/enroll", data={"name": "alice"}, files=files)
    query = _wav_bytes(synth_speech(f0=220.0, seed=99, speaker="alice"))
    client.post("/identify", files={"file": ("q.wav", query, "audio/wav")})

    r2 = client.get("/analytics")
    body2 = r2.json()
    assert body2["speakers_enrolled"] == 1
    assert body2["total_enrollments"] == 1
    assert body2["total_identifications"] == 1
    assert body2["total_operations"] == 2


def test_prompts_endpoint(client):
    r = client.get("/prompts")
    assert r.status_code == 200
    body = r.json()
    assert "prompts" in body
    assert body["count"] >= 5
    for p in body["prompts"]:
        assert "text" in p
        assert "category" in p
        assert "id" in p

