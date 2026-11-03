"""Build ``data/emotion_vad.csv`` from the official NRC VAD lexicon.

Downloads the lexicon from saifmohammad.com (needs internet), or reads a
local copy with ``--lexicon path/to/NRC-VAD-Lexicon.txt``. Prints the table.
"""

from __future__ import annotations

import argparse

from circumplex.vad import build_emotion_vad_table


def main() -> None:
    """Build and print the 27-emotion VAD table."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lexicon", default=None, help="local NRC-VAD-Lexicon.txt (skips download)")
    args = ap.parse_args()
    table = build_emotion_vad_table(lexicon_path=args.lexicon)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
