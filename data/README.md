# data/

Place your speaker corpora here. The batch commands expect:

```
data/
├── alice/
│   ├── take_001.wav
│   └── take_002.wav
├── bob/
│   └── ...
```

Everything under `data/` (except this README and `.gitkeep`) is gitignored so you can't accidentally commit voice recordings.

Enroll everything with:

```bash
speech-sense batch-enroll --dataset ./data --workers 8
```
