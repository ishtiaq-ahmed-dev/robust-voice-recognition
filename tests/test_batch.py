"""Batch enrollment / identification tests.

We run with `workers=1` so tests don't spawn a process pool on every CI hit,
but the code path is otherwise identical — the same `_embed_one` runs, just
in-process. The multiprocessing path is smoke-tested separately when
SPEECH_SENSE_TEST_PARALLEL=1 is set, so slow CI can skip it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from speech_sense.audio import save_wav
from speech_sense.batch import (
    batch_enroll_directory,
    batch_enroll_streaming,
    batch_identify,
    batch_identify_directory,
    batch_identify_stream,
    iter_audio_files,
)
from speech_sense.database import SpeakerDatabase
from speech_sense.embedding import MfccBackend, SpeakerEncoder
from speech_sense.verifier import SpeakerVerifier
from tests.conftest import TEST_CONFIG


@pytest.fixture
def dataset_dir(tmp_path: Path, synth_speech) -> Path:
    root = tmp_path / "dataset"
    for name, f0 in [("alice", 220.0), ("bob", 110.0), ("carol", 175.0)]:
        sp = root / name
        sp.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            save_wav(sp / f"clip_{i}.wav", synth_speech(f0=f0, seed=i, speaker=name))
    return root


@pytest.fixture
def verifier() -> SpeakerVerifier:
    return SpeakerVerifier(
        encoder=SpeakerEncoder(MfccBackend()),
        database=SpeakerDatabase(),
        config=TEST_CONFIG,
        use_vad=False,
    )


def test_iter_audio_files_finds_all(dataset_dir):
    paths = list(iter_audio_files(dataset_dir))
    assert len(paths) == 9  # 3 speakers x 3 clips


def test_iter_audio_files_single_file(dataset_dir):
    one = next(iter_audio_files(dataset_dir))
    assert list(iter_audio_files(one)) == [one]


def test_batch_enroll_directory(dataset_dir, verifier):
    report = batch_enroll_directory(
        verifier, dataset_dir, workers=1, backend="mfcc", config=verifier.config,
    )
    assert set(report.enrolled) == {"alice", "bob", "carol"}
    assert all(count == 3 for count in report.enrolled.values())
    assert report.n_speakers == 3
    assert report.n_clips == 9
    assert set(verifier.database.names()) == {"alice", "bob", "carol"}


def test_batch_enroll_handles_bad_file(tmp_path: Path, synth_speech, verifier):
    root = tmp_path / "ds"
    (root / "alice").mkdir(parents=True)
    save_wav(root / "alice" / "good.wav", synth_speech(f0=220, speaker="alice"))
    (root / "alice" / "bad.wav").write_bytes(b"not really audio")
    report = batch_enroll_directory(
        verifier, root, workers=1, backend="mfcc", config=verifier.config,
    )
    assert report.enrolled == {"alice": 1}
    assert any("bad.wav" in item for item in report.skipped.get("alice", []))


def test_batch_enroll_empty_dataset(tmp_path: Path, verifier):
    empty = tmp_path / "empty"
    empty.mkdir()
    report = batch_enroll_directory(
        verifier, empty, workers=1, backend="mfcc", config=verifier.config,
    )
    assert report.n_speakers == 0
    assert report.enrolled == {}


def test_batch_enroll_rejects_non_directory(tmp_path: Path, verifier):
    p = tmp_path / "file.wav"
    p.write_text("hi")
    with pytest.raises(ValueError):
        batch_enroll_directory(verifier, p, workers=1, config=verifier.config)


def test_batch_identify_returns_rows_in_order(dataset_dir, verifier):
    batch_enroll_directory(
        verifier, dataset_dir, workers=1, backend="mfcc", config=verifier.config,
    )
    paths = sorted(iter_audio_files(dataset_dir / "alice"))
    rows = batch_identify(
        verifier, paths, workers=1, backend="mfcc", config=verifier.config,
    )
    assert [r.path for r in rows] == [str(p) for p in paths]
    # All alice clips must land on alice.
    assert all(
        max(r.result.scores, key=r.result.scores.get) == "alice"
        for r in rows
    )


def test_batch_identify_bad_file_reports_no_speech(tmp_path: Path, verifier, dataset_dir):
    batch_enroll_directory(
        verifier, dataset_dir, workers=1, backend="mfcc", config=verifier.config,
    )
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"junk")
    rows = batch_identify(
        verifier, [bad], workers=1, backend="mfcc", config=verifier.config,
    )
    assert len(rows) == 1
    assert rows[0].result.is_known is False
    assert "could not read audio" in rows[0].result.reason


def test_batch_identify_empty_input(verifier):
    assert batch_identify(verifier, [], workers=1) == []


def test_batch_identify_directory(dataset_dir, verifier):
    batch_enroll_directory(
        verifier, dataset_dir, workers=1, backend="mfcc", config=verifier.config,
    )
    rows = batch_identify_directory(
        verifier, dataset_dir / "bob", workers=1, backend="mfcc", config=verifier.config,
    )
    assert len(rows) == 3
    # bob's clips should map to bob.
    assert all(
        max(r.result.scores, key=r.result.scores.get) == "bob"
        for r in rows
    )


@pytest.mark.skipif(
    os.environ.get("SPEECH_SENSE_TEST_PARALLEL") != "1",
    reason="parallel-mode smoke test — enable with SPEECH_SENSE_TEST_PARALLEL=1",
)
def test_batch_identify_parallel(dataset_dir, verifier):
    batch_enroll_directory(verifier, dataset_dir, workers=1, backend="mfcc", config=verifier.config)
    paths = list(iter_audio_files(dataset_dir))
    rows = batch_identify(verifier, paths, workers=2, backend="mfcc", config=verifier.config)
    assert len(rows) == len(paths)


def test_batch_enroll_streaming_yields_per_speaker(dataset_dir, verifier):
    events = list(batch_enroll_streaming(
        verifier, dataset_dir, workers=1, backend="mfcc", config=verifier.config,
    ))
    names = [e[0] for e in events]
    assert set(names) == {"alice", "bob", "carol"}
    for name, n_clips, meta in events:
        assert n_clips == 3
        assert meta["streaming"] is True
    assert set(verifier.database.names()) == {"alice", "bob", "carol"}


def test_batch_identify_stream_yields_rows(dataset_dir, verifier):
    batch_enroll_directory(
        verifier, dataset_dir, workers=1, backend="mfcc", config=verifier.config,
    )
    paths = list(iter_audio_files(dataset_dir))
    rows = list(batch_identify_stream(
        verifier, paths, workers=1, backend="mfcc", config=verifier.config, chunk_size=4,
    ))
    assert len(rows) == len(paths)


def test_batch_identify_stream_empty_input(verifier):
    assert list(batch_identify_stream(verifier, [], workers=1)) == []


# ── Resilience regressions ──────────────────────────────────────────────────

def test_falls_back_to_serial_when_a_process_pool_cannot_be_created(
    tmp_path, verifier, synth_speech, monkeypatch, caplog
):
    """Sandboxes, exhausted descriptors and missing /dev/shm all break spawn.

    A batch job that refuses to run at all is worse than a slow one.
    """
    from speech_sense import batch as batch_mod
    from speech_sense.audio import save_wav

    for i in range(4):
        save_wav(tmp_path / f"clip_{i}.wav", synth_speech(f0=150 + 10 * i, speaker=f"s{i}"))

    class BrokenPool:
        def __init__(self, *args, **kwargs):
            raise OSError("cannot allocate worker processes")

    monkeypatch.setattr(batch_mod, "ProcessPoolExecutor", BrokenPool)
    with caplog.at_level("WARNING", logger="speech_sense.batch"):
        rows = batch_mod.batch_identify(
            verifier, sorted(tmp_path.glob("*.wav")),
            workers=4, backend="mfcc", config=verifier.config,
        )

    assert len(rows) == 4
    assert all(r.result.backend == verifier.encoder.name for r in rows)
    assert any("serially" in rec.getMessage() for rec in caplog.records)


def test_streaming_identify_does_not_reload_the_encoder_per_chunk(
    tmp_path, verifier, synth_speech, monkeypatch
):
    """The serial path re-entered _init_worker for every chunk, which reloaded
    the model (and, for resemblyzer, re-read its weights) every batch_size
    files."""
    from speech_sense import batch as batch_mod
    from speech_sense.audio import save_wav

    for i in range(6):
        save_wav(tmp_path / f"clip_{i}.wav", synth_speech(f0=150 + 10 * i, speaker=f"s{i}"))

    monkeypatch.setattr(batch_mod, "_WORKER_ENCODER", None)
    monkeypatch.setattr(batch_mod, "_WORKER_BACKEND", None)

    loads = []
    real_load = batch_mod.SpeakerEncoder.load
    monkeypatch.setattr(
        batch_mod.SpeakerEncoder, "load",
        staticmethod(lambda backend="auto": (loads.append(backend), real_load(backend))[1]),
    )

    rows = list(batch_mod.batch_identify_stream(
        verifier, sorted(tmp_path.glob("*.wav")),
        workers=1, backend="mfcc", config=verifier.config, chunk_size=2,
    ))

    assert len(rows) == 6
    assert len(loads) == 1, f"encoder reloaded {len(loads)} times across 3 chunks"


def test_one_bad_speaker_does_not_abandon_the_rest(tmp_path, verifier, synth_speech):
    """An unusable speaker folder is reported in `skipped`, not raised."""
    from speech_sense.audio import save_wav
    from speech_sense.batch import batch_enroll_directory

    for name in ("alice", "bob"):
        (tmp_path / name).mkdir()
        save_wav(tmp_path / name / "a.wav", synth_speech(speaker=name))
    # A folder whose name can never be a valid speaker name.
    bad = tmp_path / "bad‮name"
    bad.mkdir()
    save_wav(bad / "a.wav", synth_speech(speaker="x"))

    report = batch_enroll_directory(verifier, tmp_path, workers=1, backend="mfcc", config=verifier.config)
    assert set(report.enrolled) == {"alice", "bob"}
    assert report.n_speakers == 2
    assert any("text-direction" in msg for msgs in report.skipped.values() for msg in msgs)
