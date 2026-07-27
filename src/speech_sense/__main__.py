"""Enable `python -m speech_sense …` as an alias for the `speech-sense` script.

Useful before an editable install exists, and in environments where the console
script isn't on PATH.
"""

from .cli import main

raise SystemExit(main())
