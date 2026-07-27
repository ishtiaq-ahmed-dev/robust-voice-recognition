"""Database persistence, durability, and input validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from speech_sense.database import SpeakerDatabase, clean_speaker_name


def test_add_and_lookup():
    db = SpeakerDatabase()
    emb = np.arange(64, dtype=np.float32) / 63.0
    db.add("alice", emb, meta={"source": "test"})
    assert "alice" in db
    assert len(db) == 1
    np.testing.assert_array_equal(db.embeddings["alice"], emb)
    assert db.metadata["alice"]["source"] == "test"


def test_remove():
    db = SpeakerDatabase()
    db.add("alice", np.zeros(4, dtype=np.float32))
    assert db.remove("alice")
    assert not db.remove("alice")
    assert len(db) == 0


def test_matrix_shape():
    import zlib

    db = SpeakerDatabase()
    for name in ["a", "b", "c"]:
        db.add(name, np.random.RandomState(zlib.crc32(name.encode()) & 0xFFFF).randn(16).astype(np.float32))
    names, mat = db.matrix()
    assert names == ["a", "b", "c"]
    assert mat.shape == (3, 16)


def test_save_load_roundtrip(tmp_path: Path):
    db = SpeakerDatabase()
    db.add("alice", np.linspace(-1, 1, 8, dtype=np.float32), meta={"x": 1})
    db.add("bob", np.linspace(0, 1, 8, dtype=np.float32))
    p = tmp_path / "db.npz"
    db.save(p)
    loaded = SpeakerDatabase.load(p)
    assert set(loaded.names()) == {"alice", "bob"}
    np.testing.assert_array_equal(loaded.embeddings["alice"], db.embeddings["alice"])
    assert loaded.metadata["alice"] == {"x": 1}


def test_load_missing_returns_empty(tmp_path: Path):
    db = SpeakerDatabase.load(tmp_path / "does_not_exist.npz")
    assert len(db) == 0


def test_save_empty_then_load(tmp_path: Path):
    db = SpeakerDatabase()
    p = tmp_path / "empty.npz"
    db.save(p)
    loaded = SpeakerDatabase.load(p)
    assert len(loaded) == 0


def test_save_writes_exactly_the_requested_path(tmp_path: Path):
    """np.savez appends '.npz' to a path argument; save() must not."""
    db = SpeakerDatabase()
    db.add("alice", np.ones(4, dtype=np.float32))
    p = tmp_path / "speakers_db"  # deliberately no .npz suffix
    db.save(p)
    assert p.exists(), "saved somewhere other than the requested path"
    assert not (tmp_path / "speakers_db.npz").exists()
    assert SpeakerDatabase.load(p).names() == ["alice"]


def test_save_leaves_no_temp_files_behind(tmp_path: Path):
    db = SpeakerDatabase()
    db.add("alice", np.ones(4, dtype=np.float32))
    p = tmp_path / "db.npz"
    db.save(p)
    db.save(p)
    assert [f.name for f in tmp_path.iterdir()] == ["db.npz"]


def test_corrupt_database_is_quarantined_not_fatal(tmp_path: Path):
    """A damaged file must not stop startup, and must not be destroyed."""
    p = tmp_path / "db.npz"
    p.write_bytes(b"PK\x03\x04 truncated garbage")
    db = SpeakerDatabase.load(p)
    assert len(db) == 0
    assert not p.exists(), "corrupt file should have been moved aside"
    quarantined = list(tmp_path.glob("db.npz.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"PK\x03\x04 truncated garbage"


def test_mixed_dimension_embeddings_are_rejected():
    """Mixing backends silently would make every cosine score meaningless."""
    db = SpeakerDatabase()
    db.add("alice", np.ones(78, dtype=np.float32))
    with pytest.raises(ValueError, match="does not match"):
        db.add("bob", np.ones(256, dtype=np.float32))


def test_reenrolling_the_only_speaker_may_change_dimension():
    db = SpeakerDatabase()
    db.add("alice", np.ones(78, dtype=np.float32))
    db.add("alice", np.ones(256, dtype=np.float32))
    assert db.dim == 256


def test_load_drops_minority_dimension_rather_than_crashing(tmp_path: Path):
    """A file written across a backend swap must still load and score."""
    import json

    p = tmp_path / "mixed.npz"
    np.savez(
        p,
        emb__0=np.ones(8, dtype=np.float32),
        emb__1=np.ones(8, dtype=np.float32),
        emb__2=np.ones(3, dtype=np.float32),
        __names__=np.array(["a", "b", "odd"]),
        __meta__=np.array(json.dumps({})),
    )
    db = SpeakerDatabase.load(p)
    assert set(db.names()) == {"a", "b"}
    names, mat = db.matrix()
    assert mat.shape == (2, 8)


def test_non_finite_embedding_is_rejected():
    db = SpeakerDatabase()
    with pytest.raises(ValueError, match="NaN or infinity"):
        db.add("alice", np.array([1.0, np.nan, 0.0], dtype=np.float32))


def test_matrix_cache_invalidated_on_mutation():
    db = SpeakerDatabase()
    db.add("alice", np.ones(4, dtype=np.float32))
    assert db.matrix()[1].shape == (1, 4)
    db.add("bob", np.zeros(4, dtype=np.float32))
    assert db.matrix()[1].shape == (2, 4)
    db.remove("alice")
    assert db.matrix()[0] == ["bob"]


@pytest.mark.parametrize("bad", ["", "   ", "a\x00b", "x" * 1000])
def test_clean_speaker_name_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        clean_speaker_name(bad)


def test_clean_speaker_name_normalises():
    assert clean_speaker_name("  Ishtiaq Ahmed  ") == "Ishtiaq Ahmed"
