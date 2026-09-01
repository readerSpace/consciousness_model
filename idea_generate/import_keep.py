"""Create a RawNote JSONL input file from a Google Keep export directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from idea_generate.src.keep_importer import export_keep_notes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory containing Google Keep JSON files.")
    parser.add_argument("--output", type=Path, default=Path("idea_generate/data/keepmemo_research_notes.jsonl"))
    parser.add_argument("--limit", type=int, default=100)
    arguments = parser.parse_args()
    notes = export_keep_notes(arguments.source, arguments.output, arguments.limit)
    print(f"Imported {len(notes)} research notes into {arguments.output}")