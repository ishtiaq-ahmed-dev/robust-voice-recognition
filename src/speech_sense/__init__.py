"""speech_sense — speaker verification, VAD, evaluation, and REST API."""

from .config import Config
from .verifier import SpeakerVerifier, VerificationResult
from .database import SpeakerDatabase

__version__ = "0.2.0"
__all__ = [
    "Config",
    "SpeakerVerifier",
    "VerificationResult",
    "SpeakerDatabase",
    "__version__",
]
