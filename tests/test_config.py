"""Config is a trust boundary: env vars come from operators and CI, not code."""

from __future__ import annotations

import pytest

from speech_sense.config import Config


def test_defaults_are_self_consistent():
    cfg = Config()
    assert cfg.sample_rate == 16_000
    assert 0.0 <= cfg.similarity_threshold <= 1.0
    assert cfg.workers >= 1
    assert not cfg.auth_enabled


def test_direct_construction_rejects_out_of_range():
    """A bad literal in calling code is a bug and should be loud."""
    with pytest.raises(ValueError, match="similarity_threshold"):
        Config(similarity_threshold=5.0)
    with pytest.raises(ValueError, match="sample_rate"):
        Config(sample_rate=0)
    with pytest.raises(ValueError, match="workers"):
        Config(workers=0)


def test_env_override_applies_when_valid(monkeypatch):
    monkeypatch.setenv("SPEECH_SENSE_SIMILARITY_THRESHOLD", "0.9")
    monkeypatch.setenv("SPEECH_SENSE_TOP_K_SCORES", "3")
    cfg = Config.from_env()
    assert cfg.similarity_threshold == 0.9
    assert cfg.top_k_scores == 3


@pytest.mark.parametrize(
    "name,value",
    [
        ("SPEECH_SENSE_SAMPLE_RATE", "0"),            # would divide by zero downstream
        ("SPEECH_SENSE_SAMPLE_RATE", "not-a-number"),
        ("SPEECH_SENSE_SIMILARITY_THRESHOLD", "5"),   # nothing would ever match
        ("SPEECH_SENSE_WORKERS", "-4"),
        ("SPEECH_SENSE_MAX_UPLOAD_BYTES", "0"),
        ("SPEECH_SENSE_LOG_LEVEL", "CHATTY"),
    ],
)
def test_bad_env_falls_back_to_default_with_a_warning(monkeypatch, caplog, name, value):
    """A deployment typo must degrade, not stop a 24/7 service from booting."""
    monkeypatch.setenv(name, value)
    with caplog.at_level("WARNING", logger="speech_sense.config"):
        cfg = Config.from_env()
    field = name.removeprefix("SPEECH_SENSE_").lower()
    assert getattr(cfg, field) == getattr(Config(), field)
    assert any(name in record.getMessage() for record in caplog.records), \
        "falling back to a default must never be silent"


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_env_var_is_treated_as_unset(monkeypatch, blank):
    monkeypatch.setenv("SPEECH_SENSE_SIMILARITY_THRESHOLD", blank)
    monkeypatch.setenv("SPEECH_SENSE_DATABASE_PATH", blank)
    cfg = Config.from_env()
    assert cfg.similarity_threshold == Config().similarity_threshold
    assert cfg.database_path == Config().database_path


def test_bool_env_parsing(monkeypatch):
    # No bool fields today, but the caster is shared — check it directly.
    from speech_sense.config import _cast

    assert _cast("yes", bool) is True
    assert _cast("OFF", bool) is False
    with pytest.raises(ValueError):
        _cast("maybe", bool)


def test_auth_and_cors_helpers():
    assert not Config().auth_enabled
    assert Config(api_key="  ").auth_enabled is False
    assert Config(api_key="s3cret").auth_enabled is True
    assert Config(cors_origins="https://a.test, https://b.test ,").allowed_origins == [
        "https://a.test", "https://b.test",
    ]
    assert Config().allowed_origins == []
