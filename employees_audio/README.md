# employees_audio/

This is the convention folder for organizing your workforce's voice samples for
one-click batch enrollment. Layout:

```
employees_audio/
├── alice_johnson/
│   ├── take_001.wav
│   ├── take_002.mp3
│   └── take_003.m4a
├── bob_singh/
│   ├── voice_1.wav
│   └── voice_2.wav
└── carol_diaz/
    └── ...
```

Rules:
1. **One folder per employee.** The folder name is the enrolled speaker name.
2. **Any audio format works** — WAV, MP3, M4A, OGG, FLAC. Decoded automatically.
3. **Any number of clips per employee.** More clips = more stable voice-print (5–10 is a sweet spot).
4. **Names are the enrolled ID.** Use consistent, unambiguous naming (`ishtiaq_ahmed`, not `IA` or `i.a.`).

## One-click enrollment

From the browser UI: click **"Enroll from employees_audio/"** on the home page.

From the CLI:

```bash
speech-sense batch-enroll --dataset ./employees_audio --workers 8
```

From the API:

```bash
curl -X POST http://localhost:8000/employees/enroll-all
```

All three routes hit the same parallel batch-enroll pipeline, skip unreadable
files, and return per-employee stats.
