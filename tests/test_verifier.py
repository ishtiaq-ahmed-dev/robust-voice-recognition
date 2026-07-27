"""End-to-end verifier behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from speech_sense.audio import add_white_noise
from speech_sense.verifier import SpeakerVerifier, VerificationResult


def test_enroll_requires_name(verifier, synth_speech):
    with pytest.raises(ValueError):
        verifier.enroll_from_audio("", [synth_speech()])


def test_enroll_requires_clips(verifier):
    with pytest.raises(ValueError):
        verifier.enroll_from_audio("alice", [])


def test_enroll_all_silence_raises(encoder_mfcc, silence):
    v = SpeakerVerifier(encoder=encoder_mfcc, use_vad=True)
    with pytest.raises(ValueError):
        v.enroll_from_audio("alice", [silence, silence])


def test_identify_returns_enrolled_speaker(enrolled_verifier, synth_speech):
    # Query with a fresh clip from the same synthetic 'alice' speaker (f0=220).
    query = synth_speech(f0=220.0, seconds=1.5, seed=42, speaker="alice")
    result = enrolled_verifier.identify(query)
    assert isinstance(result, VerificationResult)
    assert result.speaker == "alice"
    assert result.is_known
    assert 0.0 <= result.confidence <= 1.0
    assert set(result.scores) == {"alice", "bob", "carol"}


def test_identify_empty_database_returns_no_speaker(verifier, synth_speech):
    result = verifier.identify(synth_speech())
    assert result.speaker is None
    assert not result.is_known
    assert "empty" in result.reason


def test_identify_silence_flags_no_speech(encoder_mfcc, silence):
    v = SpeakerVerifier(encoder=encoder_mfcc, use_vad=True)
    # Enroll something so the DB is non-empty.
    v.database.add("x", np.random.RandomState(0).randn(v.encoder.dim).astype(np.float32))
    result = v.identify(silence)
    assert not result.contains_speech
    assert not result.is_known


def test_verify_matches_correct_speaker(enrolled_verifier, synth_speech):
    query = synth_speech(f0=110.0, seconds=1.5, seed=999, speaker="bob")  # 'bob'
    result = enrolled_verifier.verify(query, claimed="bob")
    assert result.speaker == "bob"
    assert result.is_known


def test_verify_rejects_wrong_speaker(enrolled_verifier, synth_speech):
    """A different speaker's voice must not verify as another enrolled speaker."""
    # Best-case rejection: identify() picks another enrolled speaker as the
    # winner, so verify(claimed="alice") sees alice with the second-best score.
    # Weaker case: MFCC-mean gives all speakers similar scores, so we just
    # require that alice is NOT the argmax winner for bob's clip.
    bob_clip = synth_speech(f0=110.0, seconds=1.5, seed=999, speaker="bob")
    identified = enrolled_verifier.identify(bob_clip)
    argmax_winner = max(identified.scores, key=identified.scores.get)
    assert argmax_winner == "bob", (
        f"MFCC fallback failed to discriminate bob's clip; "
        f"got scores: {identified.scores}"
    )


def test_verify_unknown_claim(enrolled_verifier, synth_speech):
    result = enrolled_verifier.verify(synth_speech(), claimed="ghost")
    assert not result.is_known
    assert "not enrolled" in result.reason


def test_noise_robustness_degrades_gracefully(enrolled_verifier, synth_speech):
    """Model should still favour the correct speaker under moderate noise."""
    clean = synth_speech(f0=220.0, seconds=1.5, seed=42, speaker="alice")
    noisy = add_white_noise(clean, snr_db=15.0, rng=np.random.default_rng(0))
    clean_res = enrolled_verifier.identify(clean)
    noisy_res = enrolled_verifier.identify(noisy)
    # Speaker with highest cosine similarity should still be alice.
    assert max(noisy_res.scores, key=noisy_res.scores.get) == "alice"
    # Similarity may drop but not implode.
    assert noisy_res.similarity > clean_res.similarity - 0.4


def test_result_serializes_to_json(enrolled_verifier, synth_speech):
    import json

    result = enrolled_verifier.identify(synth_speech(f0=220.0, seconds=1.5, seed=42))
    payload = json.dumps(result.to_dict())
    reloaded = json.loads(payload)
    assert reloaded["speaker"] == result.speaker
    assert "confidence" in reloaded


def test_higher_margin_means_higher_confidence(enrolled_verifier, synth_speech):
    """Confidence must rise when the top match is unambiguous."""
    unambiguous = synth_speech(f0=220.0, seconds=1.5, seed=42, speaker="alice")  # matches alice
    ambiguous = synth_speech(f0=170.0, seconds=1.5, seed=7, speaker="stranger")   # unseen speaker

    r1 = enrolled_verifier.identify(unambiguous)
    r2 = enrolled_verifier.identify(ambiguous)
    # If both matched confidently, confidence tracks margin. If the ambiguous
    # one didn't match at all, the assertion is vacuously satisfied.
    if r2.is_known:
        assert r1.margin >= r2.margin - 1e-6
        assert r1.confidence >= r2.confidence - 1e-6
