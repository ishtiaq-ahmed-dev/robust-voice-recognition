"""Central configuration for speech_sense.

Every knob the pipeline needs to be re-parameterised at runtime lives here so
downstream modules don't hard-code numbers, paths, or credentials.

Config values, in precedence order (highest wins):
    1. explicit constructor kwargs   -> Config(similarity_threshold=0.8)
    2. environment variables         -> SPEECH_SENSE_SIMILARITY_THRESHOLD=0.8
    3. defaults defined below.

Environment variables are a trust boundary: they come from an operator, a
container spec, or a CI job, and any of those can hold a typo. Two different
failure modes are therefore deliberate:

    * `Config(...)` with an out-of-range value **raises** — that's a bug in
      calling code and should be loud.
    * `Config.from_env()` with an out-of-range env var **warns and falls back
      to the default** — a typo in a deployment variable must not stop a
      24/7 service from booting.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields
from typing import Any, Callable

log = logging.getLogger(__name__)

ENV_PREFIX = "SPEECH_SENSE_"

_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n"}

# field name -> (inclusive_min, inclusive_max). Numeric fields only; anything
# absent from this table is validated by `_EXTRA_RULES` or not at all.
_BOUNDS: dict[str, tuple[float, float]] = {
    "sample_rate": (8_000, 192_000),
    "record_duration": (0.1, 300.0),
    "channels": (1, 2),
    "vad_frame_ms": (5, 200),
    "vad_min_speech_ms": (0, 60_000),
    "vad_max_silence_ms": (0, 60_000),
    "vad_speech_floor_ratio": (0.0, 1.0),
    "vad_abs_energy_floor": (0.0, 1.0),
    "vad_zcr_max": (0.0, 1.0),
    "similarity_threshold": (0.0, 1.0),
    "known_speaker_margin": (0.0, 1.0),
    "enroll_num_recordings": (1, 50),
    "stream_chunk_ms": (10, 60_000),
    "stream_window_s": (0.1, 300.0),
    "workers": (1, 256),
    "batch_size": (1, 1_000_000),
    "max_upload_bytes": (1_024, 2_147_483_648),
    "max_audio_seconds": (0.1, 86_400.0),
    "max_files_per_request": (1, 10_000),
    "max_speaker_name_len": (1, 512),
    "top_k_scores": (1, 100_000),
    "request_timeout_s": (1.0, 3_600.0),
}

_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}

# field name -> predicate that must hold for non-numeric fields.
_EXTRA_RULES: dict[str, Callable[[Any], bool]] = {
    "database_path": lambda v: isinstance(v, str) and v.strip() != "",
    "reports_dir": lambda v: isinstance(v, str) and v.strip() != "",
    "employees_audio": lambda v: isinstance(v, str) and v.strip() != "",
    "log_level": lambda v: isinstance(v, str) and v.upper() in _LOG_LEVELS,
}


def _cast(raw: str, cast: type) -> Any:
    """Cast a raw env string. Raises ValueError on anything unparseable."""
    if cast is bool:
        lowered = raw.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise ValueError(f"expected a boolean, got {raw!r}")
    return cast(raw)


def _check(name: str, value: Any) -> None:
    """Raise ValueError if `value` is outside this field's allowed range."""
    if name in _BOUNDS:
        lo, hi = _BOUNDS[name]
        # bool is a subclass of int — exclude it so a flag never gets bounds-checked.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric, got {value!r}")
        if not (lo <= value <= hi):
            raise ValueError(f"{name} must be between {lo} and {hi}, got {value!r}")
    rule = _EXTRA_RULES.get(name)
    if rule is not None and not rule(value):
        raise ValueError(f"{name} has an invalid value: {value!r}")


def _env(name: str, default: Any, cast: type) -> Any:
    """Read one env var with a typed default.

    Unset, empty, unparseable, or out-of-range all fall back to `default` so a
    deployment typo degrades to the documented behaviour instead of a crash
    loop. Every fallback is logged at WARNING so it is never silent.
    """
    raw = os.environ.get(f"{ENV_PREFIX}{name.upper()}")
    if raw is None or raw.strip() == "":
        return default
    try:
        value = _cast(raw, cast)
        _check(name, value)
        return value
    except (TypeError, ValueError) as exc:
        log.warning(
            "ignoring %s%s=%r (%s); using default %r",
            ENV_PREFIX, name.upper(), raw, exc, default,
        )
        return default


@dataclass(frozen=True)
class Config:
    # ── Audio ──
    sample_rate: int = 16_000
    record_duration: float = 5.0
    channels: int = 1

    # ── VAD ──
    vad_frame_ms: int = 30
    vad_min_speech_ms: int = 300
    vad_max_silence_ms: int = 500
    # A frame counts as speech when its RMS clears this fraction of the loudest
    # frame in the clip (~26 dB down at 0.05) *and* an absolute floor. See vad.py.
    vad_speech_floor_ratio: float = 0.05
    vad_abs_energy_floor: float = 1e-4
    vad_zcr_max: float = 0.35

    # ── Verification ──
    similarity_threshold: float = 0.75
    known_speaker_margin: float = 0.05
    enroll_num_recordings: int = 3
    # Cap how many per-speaker scores leave the process. Returning the full
    # table lets any caller enumerate every enrolled person from one request,
    # and makes responses grow linearly with the roster.
    top_k_scores: int = 10

    # ── Streaming ──
    stream_chunk_ms: int = 500
    stream_window_s: float = 2.0

    # ── Batch / parallelism ──
    workers: int = max(1, (os.cpu_count() or 2) - 1)
    batch_size: int = 32

    # ── Service limits (DoS ceilings for the REST API) ──
    max_upload_bytes: int = 25 * 1024 * 1024      # 25 MiB per file
    # Bytes alone don't bound the work: a few MiB of FLAC or Opus can decode to
    # hours of PCM, so cap the decoded duration too.
    max_audio_seconds: float = 600.0
    max_files_per_request: int = 25
    max_speaker_name_len: int = 128
    request_timeout_s: float = 300.0

    # ── Service security ──
    # Empty string = open service (fine for a local demo). Set
    # SPEECH_SENSE_API_KEY to require an X-API-Key header on mutating routes.
    api_key: str = ""
    # Comma-separated allow-list. Empty = no CORS headers, i.e. same-origin only.
    cors_origins: str = ""

    # ── Files ──
    database_path: str = "speakers.npz"
    reports_dir: str = "reports"
    # Env var is SPEECH_SENSE_EMPLOYEES_AUDIO — matches the folder it points at.
    employees_audio: str = "employees_audio"

    # ── Observability ──
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        for f in fields(self):
            _check(f.name, getattr(self, f.name))

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key.strip())

    @property
    def allowed_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @classmethod
    def from_env(cls, **overrides: Any) -> "Config":
        """Build a Config from environment variables, then apply overrides.

        `from __future__ import annotations` turns every field's `.type` into a
        string, so the cast is taken from the default value's runtime type.
        """
        env_values = {f.name: _env(f.name, f.default, type(f.default)) for f in fields(cls)}
        env_values.update(overrides)
        return cls(**env_values)


DEFAULT = Config.from_env()
