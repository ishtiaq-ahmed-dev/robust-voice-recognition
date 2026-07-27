"""Script to generate employee audio dataset with folder structure:
employees_audio/
├── Ishtiaq_Ahmed/
│   ├── take_01.wav
│   ├── take_02.wav
│   └── take_03.wav
├── Sarah_Jenkins/
│   ├── take_01.wav
│   ├── take_02.wav
│   └── take_03.wav
...
"""

import sys
import zlib
from pathlib import Path

import numpy as np

# Add src/ to sys.path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speech_sense.audio import save_wav
from speech_sense.config import DEFAULT


def generate_employee_audio(
    f0: float = 180.0,
    seconds: float = 2.5,
    sr: int = DEFAULT.sample_rate,
    seed: int = 0,
    speaker: str = "default",
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # crc32, not hash(): Python salts str hashing per process, so hash() would
    # give each employee a completely different synthetic voice on every run
    # and this generator would not be reproducible.
    speaker_rng = np.random.default_rng(zlib.crc32(speaker.encode()))


    formant_shift = speaker_rng.uniform(-350, 350, size=3)
    formant_amps = speaker_rng.uniform(0.15, 0.5, size=3)
    extra_resonance = speaker_rng.uniform(3200, 4800)
    extra_amp = speaker_rng.uniform(0.05, 0.2)

    t = np.linspace(0.0, seconds, int(seconds * sr), endpoint=False)
    wave = np.zeros_like(t)
    for k, freq in enumerate([f0, 2 * f0, 3 * f0]):
        wave += (1.0 / (k + 1)) * np.sin(2 * np.pi * freq * t)
    for freq, amp in zip(np.array([700, 1220, 2600]) + formant_shift, formant_amps):
        wave += amp * np.sin(2 * np.pi * freq * t + speaker_rng.uniform(0, 2 * np.pi))
    wave += extra_amp * np.sin(2 * np.pi * extra_resonance * t)
    # Speech cadence / pause envelope
    cadence = 0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t)
    wave *= cadence
    wave += rng.normal(0, 0.008, size=wave.shape)
    wave = wave / (np.max(np.abs(wave)) + 1e-9) * 0.85
    return wave.astype(np.float32)


def main():
    target_dir = ROOT / "employees_audio"
    target_dir.mkdir(exist_ok=True)

    employees = [
        ("Ishtiaq_Ahmed", 140.0),
        ("Sarah_Jenkins", 210.0),
        ("David_Chen", 160.0),
        ("Elena_Rostova", 225.0),
        ("Marcus_Vance", 125.0),
        ("Aisha_Khan", 235.0),
    ]

    print(f"Creating employee audio folder structure at: {target_dir}")
    for name, base_f0 in employees:
        emp_folder = target_dir / name
        emp_folder.mkdir(exist_ok=True)
        for take_idx in range(1, 4):
            wav_path = emp_folder / f"take_0{take_idx}.wav"
            f0 = base_f0 + (take_idx - 2) * 4.0
            audio = generate_employee_audio(
                f0=f0,
                seconds=2.5,
                seed=take_idx * 101,
                speaker=name,
            )
            save_wav(wav_path, audio, sr=DEFAULT.sample_rate)
            print(f"  [+] Created {wav_path.relative_to(ROOT)}")

    print("Employee audio dataset created successfully!")


if __name__ == "__main__":
    main()
