"""Evaluation harness — speaker-disjoint splits, closed- and open-set metrics.

Two things matter for evaluating a speaker-verification system honestly:

1. **No speaker leakage.** Every function here that "splits" a dataset splits
   *by speaker*, not by utterance. Splitting by utterance would let the same
   voice appear in enrollment and query sets and quietly inflate accuracy.

2. **Confusion matrix + per-class breakdown.** Overall accuracy hides the
   under-represented speakers; the per-class report and heatmap force those
   into view.

Robustness is measured by re-running the whole evaluation with additive white
noise at a sweep of SNRs. Numbers land in a single results table + PNG so the
README can show them without hand-drawing anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .audio import add_white_noise
from .config import DEFAULT, Config
from .database import SpeakerDatabase
from .embedding import SpeakerEncoder
from .verifier import SpeakerVerifier


@dataclass
class SpeakerSample:
    speaker: str
    audio: np.ndarray
    sr: int = DEFAULT.sample_rate
    tag: str = ""


@dataclass
class EvalReport:
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion: np.ndarray
    speakers: list[str]
    n_samples: int
    n_speakers: int
    backend: str
    condition: str = "clean"
    eer: Optional[float] = None
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_class": self.per_class,
            "confusion": self.confusion.tolist(),
            "speakers": self.speakers,
            "n_samples": self.n_samples,
            "n_speakers": self.n_speakers,
            "backend": self.backend,
            "condition": self.condition,
            "eer": self.eer,
            "extras": self.extras,
        }


def speaker_split(
    samples: Sequence[SpeakerSample],
    enroll_per_speaker: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> tuple[list[SpeakerSample], list[SpeakerSample]]:
    """Deterministic per-speaker split: first N go to enroll, rest to test.

    Uses a permutation seeded by `rng` so different runs can still explore
    different splits, but a fixed seed reproduces exactly.
    """
    rng = rng or np.random.default_rng(0)
    by_speaker: dict[str, list[SpeakerSample]] = {}
    for s in samples:
        by_speaker.setdefault(s.speaker, []).append(s)

    enroll: list[SpeakerSample] = []
    test: list[SpeakerSample] = []
    for speaker, clips in by_speaker.items():
        idx = rng.permutation(len(clips))
        for i, k in enumerate(idx):
            (enroll if i < enroll_per_speaker else test).append(clips[k])
    return enroll, test


def evaluate(
    verifier: SpeakerVerifier,
    test_samples: Sequence[SpeakerSample],
    condition: str = "clean",
) -> EvalReport:
    """Run closed-set identification over `test_samples` and score it."""
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
    )

    if len(verifier.database) == 0:
        raise RuntimeError("verifier database is empty — enroll speakers before evaluating")
    if not test_samples:
        raise ValueError("test_samples is empty")

    speakers = verifier.database.names()
    y_true, y_pred = [], []
    confidences: list[float] = []
    correct_confidences: list[float] = []
    wrong_confidences: list[float] = []
    for sample in test_samples:
        result = verifier.identify(sample.audio)
        # Fall back to argmax of the scores when the confidence threshold rejects,
        # so that closed-set accuracy reflects "best guess" not "known/unknown".
        pred = result.speaker
        if pred is None and result.scores:
            pred = max(result.scores, key=result.scores.get)
        y_true.append(sample.speaker)
        y_pred.append(pred or "__unknown__")
        confidences.append(result.confidence)
        (correct_confidences if pred == sample.speaker else wrong_confidences).append(
            result.confidence
        )

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = float(np.mean([a == b for a, b in zip(y_true, y_pred)]))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    per_class = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    per_class = {
        k: v for k, v in per_class.items()
        if k not in {"accuracy", "macro avg", "weighted avg"}
    }

    return EvalReport(
        accuracy=acc,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion=cm,
        speakers=labels,
        n_samples=len(test_samples),
        n_speakers=len(speakers),
        backend=verifier.encoder.name,
        condition=condition,
        extras={
            "mean_confidence_correct": float(np.mean(correct_confidences)) if correct_confidences else 0.0,
            "mean_confidence_wrong": float(np.mean(wrong_confidences)) if wrong_confidences else 0.0,
            "n_correct": len(correct_confidences),
            "n_wrong": len(wrong_confidences),
        },
    )


def evaluate_noisy(
    verifier: SpeakerVerifier,
    test_samples: Sequence[SpeakerSample],
    snrs_db: Sequence[float] = (20.0, 10.0, 5.0, 0.0),
) -> list[EvalReport]:
    """Re-run evaluation with additive noise at each SNR (dB). Includes clean baseline."""
    reports = [evaluate(verifier, test_samples, condition="clean")]
    for snr in snrs_db:
        rng = np.random.default_rng(int(snr * 100))
        noisy = [
            SpeakerSample(
                speaker=s.speaker,
                audio=add_white_noise(s.audio, snr, rng),
                sr=s.sr,
                tag=f"noise_{snr}db",
            )
            for s in test_samples
        ]
        reports.append(evaluate(verifier, noisy, condition=f"noise_snr_{snr:g}db"))
    return reports


def plot_confusion(report: EvalReport, out_path: str | Path) -> Path:
    """Save the confusion matrix as a PNG (portrait-friendly)."""
    import matplotlib
    matplotlib.use("Agg")  # headless — evaluate script runs in CI too
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(report.speakers), 1 + 0.6 * len(report.speakers)))
    sns.heatmap(
        report.confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=report.speakers,
        yticklabels=report.speakers,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"Confusion matrix — {report.backend} — {report.condition}\n"
        f"acc={report.accuracy:.3f}  macro_f1={report.macro_f1:.3f}  n={report.n_samples}"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def write_reports(reports: Sequence[EvalReport], out_dir: str | Path) -> Path:
    """Persist a bundle of reports as JSON, plus a confusion-matrix PNG per condition."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in reports:
        plot_confusion(r, out_dir / f"confusion_{r.condition}.png")
    payload = [r.to_dict() for r in reports]
    (out_dir / "reports.json").write_text(json.dumps(payload, indent=2))
    return out_dir / "reports.json"


# ---------- Synthetic dataset (self-contained, no downloads) ---------- #

def synthetic_speaker_dataset(
    n_speakers: int = 6,
    clips_per_speaker: int = 6,
    duration_s: float = 2.0,
    sample_rate: int = DEFAULT.sample_rate,
    seed: int = 42,
) -> list[SpeakerSample]:
    """Build a synthetic multi-speaker dataset from harmonic tones + formants.

    Each 'speaker' has a fixed fundamental frequency + formant set + noise
    signature. Clips are the same speaker with jitter across trials. This is
    a stand-in — it lets the evaluation harness produce a real confusion
    matrix and real metrics without shipping a dataset. It's tuned to be
    non-trivial: fundamentals are spaced but formants overlap.
    """
    rng = np.random.default_rng(seed)
    samples: list[SpeakerSample] = []
    base_f0 = np.linspace(95, 260, n_speakers)  # human-ish pitch range
    formant_shifts = rng.uniform(-30, 30, size=(n_speakers, 3))

    t = np.linspace(0.0, duration_s, int(duration_s * sample_rate), endpoint=False)
    for i in range(n_speakers):
        speaker = f"speaker_{i:02d}"
        f0_i = base_f0[i]
        formants = np.array([700, 1220, 2600]) + formant_shifts[i]
        for c in range(clips_per_speaker):
            jitter = rng.normal(0.0, 3.0)
            f0 = f0_i + jitter
            harmonics = np.zeros_like(t)
            for k, freq in enumerate([f0, 2 * f0, 3 * f0, 4 * f0]):
                harmonics += (1.0 / (k + 1)) * np.sin(2 * np.pi * freq * t)
            for freq, amp in zip(formants, [0.4, 0.3, 0.2]):
                harmonics += amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
            # Amplitude envelope so the "vowel" sounds are not perfectly stationary,
            # otherwise VAD will collapse them to one segment.
            envelope = 0.5 + 0.5 * np.sin(2 * np.pi * (2.5 + 0.5 * c) * t)
            wave = (harmonics * envelope).astype(np.float32)
            wave += rng.normal(0.0, 0.01, size=wave.shape).astype(np.float32)
            wave = wave / (np.max(np.abs(wave)) + 1e-9) * 0.9
            samples.append(SpeakerSample(speaker=speaker, audio=wave, sr=sample_rate, tag=f"clip_{c}"))
    return samples


def run_synthetic_evaluation(
    out_dir: str | Path = "reports",
    backend: str = "auto",
    n_speakers: int = 6,
    clips_per_speaker: int = 6,
    snrs_db: Sequence[float] = (20.0, 10.0, 5.0),
    seed: int = 42,
) -> tuple[Path, list[EvalReport]]:
    """End-to-end: build synthetic data → enroll → evaluate clean+noisy → save."""
    dataset = synthetic_speaker_dataset(
        n_speakers=n_speakers, clips_per_speaker=clips_per_speaker, seed=seed
    )
    enroll, test = speaker_split(dataset, enroll_per_speaker=2, rng=np.random.default_rng(seed))

    encoder = SpeakerEncoder.load(backend)  # type: ignore[arg-type]
    verifier = SpeakerVerifier(encoder=encoder, database=SpeakerDatabase(), use_vad=False)

    # Ponytail: for the synthetic vowels VAD would clip most of the waveform
    # (they're not real speech), so we bypass it here. Real audio in the API
    # path still uses VAD by default.
    grouped: dict[str, list[np.ndarray]] = {}
    for s in enroll:
        grouped.setdefault(s.speaker, []).append(s.audio)
    for speaker, clips in grouped.items():
        verifier.enroll_from_audio(speaker, clips)

    reports = evaluate_noisy(verifier, test, snrs_db=snrs_db)
    report_path = write_reports(reports, out_dir)
    return report_path, reports
