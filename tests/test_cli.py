"""CLI smoke tests — exercised by invoking the parser directly."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from speech_sense.audio import save_wav
from speech_sense.cli import build_parser, main


def _write_wav(path: Path, synth_speech, f0: float, seed: int, speaker: str = "default") -> Path:
    save_wav(path, synth_speech(f0=f0, seconds=1.5, seed=seed, speaker=speaker))
    return path


def test_enroll_and_identify_via_cli(tmp_path: Path, synth_speech, capsys):
    db = str(tmp_path / "db.npz")
    files = [_write_wav(tmp_path / f"a_{i}.wav", synth_speech, 220.0, i, speaker="alice") for i in range(3)]

    rc = main([
        "--database", db, "--backend", "mfcc",
        "enroll", "--name", "alice", "--audio", *[str(f) for f in files],
    ])
    assert rc == 0

    query = _write_wav(tmp_path / "q.wav", synth_speech, 220.0, 99, speaker="alice")
    rc = main([
        "--database", db, "--backend", "mfcc",
        "identify", "--audio", str(query),
    ])
    # Identify returns exit code 0 iff a known speaker was found.
    assert rc == 0
    captured = capsys.readouterr().out
    assert "alice" in captured


def test_list_and_delete(tmp_path: Path, synth_speech, capsys):
    db = str(tmp_path / "db.npz")
    files = [_write_wav(tmp_path / f"a_{i}.wav", synth_speech, 220.0, i, speaker="alice") for i in range(3)]
    main(["--database", db, "--backend", "mfcc", "enroll", "--name", "alice", "--audio", *[str(f) for f in files]])
    main(["--database", db, "--backend", "mfcc", "list"])
    out = capsys.readouterr().out
    assert "alice" in out

    rc = main(["--database", db, "--backend", "mfcc", "delete", "--name", "alice"])
    assert rc == 0

    rc = main(["--database", db, "--backend", "mfcc", "delete", "--name", "alice"])
    # Second delete should fail cleanly.
    assert rc == 1


def test_parser_has_all_subcommands():
    import argparse

    parser = build_parser()
    subparsers_action = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    for name in ["enroll", "identify", "verify", "list", "delete", "serve", "evaluate", "benchmark", "repl", "record"]:
        assert name in subparsers_action.choices
