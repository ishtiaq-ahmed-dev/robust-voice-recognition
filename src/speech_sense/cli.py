"""Command-line interface for speech_sense.

Sub-commands:

    speech-sense enroll   --name X  --audio a.wav [b.wav ...]
    speech-sense identify --audio q.wav
    speech-sense verify   --name X  --audio q.wav
    speech-sense list
    speech-sense delete   --name X
    speech-sense serve    [--host 0.0.0.0 --port 8000]
    speech-sense evaluate [--out reports/]
    speech-sense benchmark
    speech-sense repl     # interactive record/enroll/identify loop
    speech-sense record   --duration 5 --out clip.wav   # mic capture (requires sounddevice)

Design: all subcommands share a single `SpeakerVerifier` construction path, so
users don't have to reason about whether the same DB is used from CLI, API,
and REPL.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import DEFAULT
from .database import SpeakerDatabase
from .embedding import SpeakerEncoder
from .verifier import SpeakerVerifier

log = logging.getLogger(__name__)


def _build_verifier(args: argparse.Namespace) -> SpeakerVerifier:
    encoder = SpeakerEncoder.load(args.backend)
    database = SpeakerDatabase.load(args.database)
    return SpeakerVerifier(encoder=encoder, database=database, config=DEFAULT)


def _persist(v: SpeakerVerifier, path: str) -> None:
    v.database.save(path)


def cmd_enroll(args: argparse.Namespace) -> int:
    v = _build_verifier(args)
    v.enroll_from_files(args.name, args.audio)
    _persist(v, args.database)
    print(f"Enrolled {args.name!r} from {len(args.audio)} clip(s) using {v.encoder.name} backend.")
    return 0


def cmd_identify(args: argparse.Namespace) -> int:
    v = _build_verifier(args)
    result = v.identify_file(args.audio)
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.is_known else 1


def cmd_verify(args: argparse.Namespace) -> int:
    v = _build_verifier(args)
    from .audio import load_wav

    audio = load_wav(args.audio, DEFAULT.sample_rate)
    result = v.verify(audio, args.name)
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.is_known else 1


def cmd_list(args: argparse.Namespace) -> int:
    db = SpeakerDatabase.load(args.database)
    if not db:
        print("(no enrolled speakers)")
        return 0
    for name in db.names():
        meta = db.metadata.get(name, {})
        print(f"  {name}  ({meta})" if meta else f"  {name}")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    db = SpeakerDatabase.load(args.database)
    if not db.remove(args.name):
        print(f"speaker {args.name!r} not found", file=sys.stderr)
        return 1
    db.save(args.database)
    print(f"Deleted {args.name!r}.")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn  # noqa: WPS433 (deferred import)
    except ImportError:
        print("uvicorn not installed. Install with: pip install speech_sense[api]", file=sys.stderr)
        return 2

    from . import api as api_mod

    if args.host not in {"127.0.0.1", "localhost", "::1"} and not DEFAULT.auth_enabled:
        # Binding to a public interface with no API key exposes enrolment and
        # deletion of biometric records to anyone who can reach the port.
        print(
            f"warning: serving on {args.host} with no API key. Set "
            f"SPEECH_SENSE_API_KEY to require an X-API-Key header.",
            file=sys.stderr,
        )

    api_mod.app = api_mod.create_app(database_path=args.database, backend=args.backend)
    uvicorn.run(api_mod.app, host=args.host, port=args.port, log_level=DEFAULT.log_level.lower())
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import run_synthetic_evaluation

    path, reports = run_synthetic_evaluation(
        out_dir=args.out,
        backend=args.backend,
        n_speakers=args.speakers,
        clips_per_speaker=args.clips,
        snrs_db=args.snr,
    )
    print(f"Wrote {path}")
    for r in reports:
        print(f"  {r.condition:20s}  acc={r.accuracy:.3f}  macro_f1={r.macro_f1:.3f}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmark import run_benchmark

    result = run_benchmark(backend=args.backend, seconds=args.seconds, iterations=args.iterations)
    print(json.dumps(result, indent=2))
    return 0


def cmd_batch_enroll(args: argparse.Namespace) -> int:
    from .batch import batch_enroll_directory

    v = _build_verifier(args)
    def progress(path: str, done: int, total: int) -> None:
        pct = 100 * done / total
        print(f"\r[{done}/{total} {pct:5.1f}%] {path[-60:]}", end="", file=sys.stderr, flush=True)
    print(f"Enrolling from {args.dataset!r} with {args.workers or 'auto'} workers...", file=sys.stderr)
    report = batch_enroll_directory(
        v, args.dataset, workers=args.workers,
        backend=args.backend, config=v.config, on_progress=progress,
    )
    print(file=sys.stderr)  # newline after progress bar
    _persist(v, args.database)
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def cmd_batch_identify(args: argparse.Namespace) -> int:
    from .batch import batch_identify, batch_identify_directory

    if not args.directory and not args.audio:
        # Silently returning an empty list here reads as "nothing matched"
        # rather than "you forgot an argument".
        print("error: pass --audio FILE [FILE ...] or --directory DIR", file=sys.stderr)
        return 2

    v = _build_verifier(args)
    inputs = args.audio or []
    if args.directory:
        rows = batch_identify_directory(
            v, args.directory, workers=args.workers,
            backend=args.backend, config=v.config,
        )
    else:
        rows = batch_identify(
            v, inputs, workers=args.workers, backend=args.backend, config=v.config,
        )
    payload = [r.to_dict() for r in rows]
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.out} ({len(rows)} rows)", file=sys.stderr)
    else:
        print(json.dumps(payload, indent=2))
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed. Install with: pip install speech_sense[mic]", file=sys.stderr)
        return 2
    from .audio import save_wav

    if args.duration <= 0:
        print("error: --duration must be positive", file=sys.stderr)
        return 2

    print(f"Recording {args.duration:.1f}s at {DEFAULT.sample_rate} Hz...")
    try:
        frames = sd.rec(
            int(args.duration * DEFAULT.sample_rate),
            samplerate=DEFAULT.sample_rate, channels=1,
        )
        sd.wait()
    except Exception as exc:  # noqa: BLE001 — sounddevice raises a family of PortAudio errors
        print(f"error: could not record from the microphone ({exc})", file=sys.stderr)
        return 1
    save_wav(args.out, frames.reshape(-1))
    print(f"Saved -> {args.out}")
    return 0


def cmd_repl(args: argparse.Namespace) -> int:
    """Interactive record/enroll/identify loop for use without the web dashboard."""
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed. Install with: pip install speech_sense[mic]", file=sys.stderr)
        return 2

    from .audio import add_white_noise  # noqa: F401  (available in repl scope)

    v = _build_verifier(args)

    def record(duration: float):
        print(f"Recording {duration:.1f}s...")
        frames = sd.rec(int(duration * DEFAULT.sample_rate), samplerate=DEFAULT.sample_rate, channels=1)
        sd.wait()
        return frames.reshape(-1)

    while True:
        print("\n=== speech_sense REPL ===")
        print("1) Enroll  2) Identify  3) List  4) Delete  5) Exit")
        choice = input("> ").strip()
        try:
            if choice == "1":
                name = input("Name: ").strip()
                if not name:
                    print("Name required.")
                    continue
                clips = []
                for i in range(DEFAULT.enroll_num_recordings):
                    input(f"Press Enter for recording {i+1}/{DEFAULT.enroll_num_recordings}...")
                    clips.append(record(DEFAULT.record_duration))
                v.enroll_from_audio(name, clips)
                _persist(v, args.database)
                print(f"Enrolled {name}.")
            elif choice == "2":
                input("Press Enter to record identification clip...")
                audio = record(DEFAULT.record_duration)
                result = v.identify(audio)
                print(json.dumps(result.to_dict(), indent=2))
            elif choice == "3":
                for n in v.database.names():
                    print(f"  {n}")
            elif choice == "4":
                name = input("Delete which: ").strip()
                if v.database.remove(name):
                    _persist(v, args.database)
                    print(f"Deleted {name}.")
                else:
                    print("Not found.")
            elif choice == "5":
                print("Bye.")
                return 0
            else:
                print("Unknown choice.")
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            return 0
        except Exception as exc:  # noqa: BLE001  — REPL should never crash on user error
            print(f"Error: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="speech-sense", description="Speaker verification CLI.")
    parser.add_argument("--database", default=DEFAULT.database_path, help="path to speaker database (.npz or legacy .pkl)")
    parser.add_argument(
        "--backend", default="auto", choices=["auto", "resemblyzer", "mfcc"],
        help="speaker embedding backend (default: auto)",
    )
    parser.add_argument(
        "--log-level", default=DEFAULT.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help=f"logging verbosity (default: {DEFAULT.log_level})",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enroll", help="enroll a speaker from one or more audio files")
    e.add_argument("--name", required=True)
    e.add_argument("--audio", nargs="+", required=True, help="paths to WAV files")
    e.set_defaults(func=cmd_enroll)

    i = sub.add_parser("identify", help="identify the speaker in a WAV file")
    i.add_argument("--audio", required=True)
    i.set_defaults(func=cmd_identify)

    ve = sub.add_parser("verify", help="verify a WAV file against a claimed speaker")
    ve.add_argument("--name", required=True)
    ve.add_argument("--audio", required=True)
    ve.set_defaults(func=cmd_verify)

    ls = sub.add_parser("list", help="list enrolled speakers")
    ls.set_defaults(func=cmd_list)

    d = sub.add_parser("delete", help="remove a speaker")
    d.add_argument("--name", required=True)
    d.set_defaults(func=cmd_delete)

    s = sub.add_parser("serve", help="run the FastAPI service")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    ev = sub.add_parser("evaluate", help="run synthetic-data evaluation and write reports")
    ev.add_argument("--out", default="reports", type=Path)
    ev.add_argument("--speakers", type=int, default=6)
    ev.add_argument("--clips", type=int, default=6)
    ev.add_argument("--snr", nargs="+", type=float, default=[20.0, 10.0, 5.0])
    ev.set_defaults(func=cmd_evaluate)

    b = sub.add_parser("benchmark", help="measure embedding latency and model size")
    b.add_argument("--seconds", type=float, default=3.0)
    b.add_argument("--iterations", type=int, default=20)
    b.set_defaults(func=cmd_benchmark)

    r = sub.add_parser("record", help="record a clip from the microphone")
    r.add_argument("--duration", type=float, default=5.0)
    r.add_argument("--out", required=True, type=Path)
    r.set_defaults(func=cmd_record)

    rep = sub.add_parser("repl", help="interactive record/enroll/identify loop")
    rep.set_defaults(func=cmd_repl)

    be = sub.add_parser("batch-enroll", help="enroll every speaker under a dataset/<speaker>/*.wav directory in parallel")
    be.add_argument("--dataset", required=True, help="root directory with one subdir per speaker")
    be.add_argument("--workers", type=int, default=None, help="worker processes (default: SPEECH_SENSE_WORKERS or CPU-1)")
    be.set_defaults(func=cmd_batch_enroll)

    bi = sub.add_parser("batch-identify", help="identify many audio files in parallel")
    bi.add_argument("--audio", nargs="*", help="individual audio files")
    bi.add_argument("--directory", help="or a directory to recursively identify all audio files under")
    bi.add_argument("--workers", type=int, default=None)
    bi.add_argument("--out", help="write results as JSON to this file (default: stdout)")
    bi.set_defaults(func=cmd_batch_identify)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a subcommand, turning every expected failure into a clean message.

    A CLI that answers a missing file or an unenrolled speaker with a Python
    traceback is telling the user about our stack frames instead of their
    problem. Unexpected exceptions still print a traceback at DEBUG so real
    bugs stay diagnosable.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except BrokenPipeError:
        # `speech-sense list | head` closes the pipe early; that is not an error.
        return 0
    except (ValueError, FileNotFoundError, IsADirectoryError, PermissionError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        log.debug("command failed", exc_info=True)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"error: unexpected failure ({type(exc).__name__}: {exc})", file=sys.stderr)
        print("Re-run with --log-level DEBUG for the full traceback.", file=sys.stderr)
        log.debug("unexpected failure", exc_info=True)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
