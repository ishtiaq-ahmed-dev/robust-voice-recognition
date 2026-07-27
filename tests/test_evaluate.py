"""Evaluation harness: split hygiene, metrics, confusion matrix, reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from speech_sense.database import SpeakerDatabase
from speech_sense.embedding import MfccBackend, SpeakerEncoder
from speech_sense.evaluate import (
    SpeakerSample,
    evaluate,
    evaluate_noisy,
    plot_confusion,
    run_synthetic_evaluation,
    speaker_split,
    synthetic_speaker_dataset,
    write_reports,
)
from speech_sense.verifier import SpeakerVerifier


def test_speaker_split_has_no_leakage():
    samples = synthetic_speaker_dataset(n_speakers=4, clips_per_speaker=5, seed=1)
    enroll, test = speaker_split(samples, enroll_per_speaker=2, rng=np.random.default_rng(1))
    # Every speaker appears in both sets (2 enroll, 3 test)
    assert {s.speaker for s in enroll} == {s.speaker for s in test}
    # No exact clip appears in both sets.
    enroll_ids = {(s.speaker, s.tag) for s in enroll}
    test_ids = {(s.speaker, s.tag) for s in test}
    assert enroll_ids.isdisjoint(test_ids)


def test_evaluate_runs_and_reports_reasonable_accuracy():
    dataset = synthetic_speaker_dataset(n_speakers=4, clips_per_speaker=4, seed=3)
    enroll, test = speaker_split(dataset, enroll_per_speaker=2, rng=np.random.default_rng(3))

    verifier = SpeakerVerifier(
        encoder=SpeakerEncoder(MfccBackend()),
        database=SpeakerDatabase(),
        use_vad=False,
    )
    grouped: dict[str, list[np.ndarray]] = {}
    for s in enroll:
        grouped.setdefault(s.speaker, []).append(s.audio)
    for name, clips in grouped.items():
        verifier.enroll_from_audio(name, clips)

    report = evaluate(verifier, test)
    # On synthetic well-separated speakers, MFCC-mean does very well.
    assert report.accuracy >= 0.8
    assert 0.0 <= report.macro_f1 <= 1.0
    assert report.confusion.shape[0] == report.confusion.shape[1]
    assert report.n_samples == len(test)


def test_evaluate_noisy_produces_condition_per_snr():
    dataset = synthetic_speaker_dataset(n_speakers=3, clips_per_speaker=4, seed=2)
    enroll, test = speaker_split(dataset, enroll_per_speaker=2, rng=np.random.default_rng(2))
    verifier = SpeakerVerifier(
        encoder=SpeakerEncoder(MfccBackend()),
        database=SpeakerDatabase(),
        use_vad=False,
    )
    grouped: dict[str, list[np.ndarray]] = {}
    for s in enroll:
        grouped.setdefault(s.speaker, []).append(s.audio)
    for name, clips in grouped.items():
        verifier.enroll_from_audio(name, clips)

    reports = evaluate_noisy(verifier, test, snrs_db=[20.0, 5.0])
    conds = [r.condition for r in reports]
    assert "clean" in conds
    assert any("noise_snr_20" in c for c in conds)
    assert any("noise_snr_5" in c for c in conds)


def test_write_reports_saves_json_and_pngs(tmp_path: Path):
    dataset = synthetic_speaker_dataset(n_speakers=3, clips_per_speaker=3, seed=4)
    enroll, test = speaker_split(dataset, enroll_per_speaker=2, rng=np.random.default_rng(4))
    verifier = SpeakerVerifier(
        encoder=SpeakerEncoder(MfccBackend()),
        database=SpeakerDatabase(),
        use_vad=False,
    )
    grouped: dict[str, list[np.ndarray]] = {}
    for s in enroll:
        grouped.setdefault(s.speaker, []).append(s.audio)
    for name, clips in grouped.items():
        verifier.enroll_from_audio(name, clips)

    report = evaluate(verifier, test)
    plot_confusion(report, tmp_path / "cm.png")
    assert (tmp_path / "cm.png").exists()
    write_reports([report], tmp_path)
    assert (tmp_path / "reports.json").exists()
    assert (tmp_path / f"confusion_{report.condition}.png").exists()


def test_synthetic_end_to_end(tmp_path: Path):
    path, reports = run_synthetic_evaluation(
        out_dir=tmp_path, backend="mfcc", n_speakers=3, clips_per_speaker=3, snrs_db=(10.0,), seed=5,
    )
    assert path.exists()
    assert any(r.condition == "clean" for r in reports)


def test_adversarial_pure_silence_never_claims_a_speaker():
    """Regression test — a silent input must never be reported as 'recognised'."""
    dataset = synthetic_speaker_dataset(n_speakers=3, clips_per_speaker=3, seed=7)
    verifier = SpeakerVerifier(
        encoder=SpeakerEncoder(MfccBackend()),
        database=SpeakerDatabase(),
        use_vad=True,  # VAD ON — this is the exact path the API uses.
    )
    enroll, _ = speaker_split(dataset, enroll_per_speaker=2, rng=np.random.default_rng(7))
    grouped: dict[str, list[np.ndarray]] = {}
    for s in enroll:
        grouped.setdefault(s.speaker, []).append(s.audio)
    for name, clips in grouped.items():
        verifier.enroll_from_audio(name, clips)

    silence = np.zeros(16000, dtype=np.float32)
    result = verifier.identify(silence)
    assert result.speaker is None
    assert not result.is_known
    assert not result.contains_speech
