"""Export the Resemblyzer encoder to ONNX for portable inference.

Only meaningful for the Resemblyzer backend — the MFCC fallback has no learned
weights. This script is a nice-to-have; the pipeline runs perfectly fine
without ONNX. The exported model takes a 40-mel-bin log-mel spectrogram of
shape (T, 40) and returns a 256-D unit vector, matching Resemblyzer's own
input contract.

Usage:
    python scripts/export_onnx.py --out models/voice_encoder.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src/ so we can import the encoder without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def export(out_path: Path, opset: int = 17) -> Path:
    import torch
    from resemblyzer import VoiceEncoder

    encoder = VoiceEncoder(verbose=False)
    encoder.eval()
    # Resemblyzer's model expects mel-spectrogram frames as float32 tensors
    # of shape (batch, T, 40). Give it a plausible dummy input.
    dummy = torch.zeros(1, 160, 40, dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        encoder,
        (dummy,),
        str(out_path),
        input_names=["mel_spectrogram"],
        output_names=["speaker_embedding"],
        dynamic_axes={
            "mel_spectrogram": {0: "batch", 1: "time"},
            "speaker_embedding": {0: "batch"},
        },
        opset_version=opset,
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Exported ONNX model -> {out_path} ({size_mb:.2f} MB)")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Resemblyzer VoiceEncoder to ONNX.")
    parser.add_argument("--out", type=Path, default=Path("models/voice_encoder.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    try:
        export(args.out, args.opset)
    except ImportError as exc:
        print(f"error: missing dependency ({exc}). Install with: pip install torch resemblyzer", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
