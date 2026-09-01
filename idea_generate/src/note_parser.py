"""Input and output helpers for JSON Lines research notes."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .models import RawNote


def load_notes(path: str | Path) -> list[RawNote]:
    notes: list[RawNote] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("id") or not record.get("text"):
                raise ValueError(f"Note at line {line_number} needs id and text.")
            notes.append(RawNote(**record))
    return notes


def load_markdown_note(path: str | Path, note_id: str = "update") -> RawNote:
    """Load a Markdown research update as one additional RawNote."""
    source = Path(path)
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Markdown update must not be empty.")
    return RawNote(id=note_id, text=text, created_at=str(source.stat().st_mtime_ns // 1_000))


def write_json(path: str | Path, records: list[object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, records: list[RawNote]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")