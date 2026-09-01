"""Import a bounded research-note subset from a Google Keep JSON export."""

from __future__ import annotations

import json
from pathlib import Path

from .models import RawNote
from .note_parser import write_jsonl


RESEARCH_KEYWORDS = (
    "意識", "consciousness", "認知", "強化学習", "機械学習", "学習",
    "仮説", "実験", "研究", "workspace", "モデル", "agent", "ai",
)


def import_keep_notes(
    source_directory: str | Path,
    limit: int = 100,
    keywords: tuple[str, ...] = RESEARCH_KEYWORDS,
) -> list[RawNote]:
    """Return the newest relevant, non-trashed Keep notes as RawNote records."""
    if limit < 1:
        raise ValueError("limit must be at least one")
    candidates: list[tuple[int, Path, dict[str, object]]] = []
    for path in Path(source_directory).glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        text = str(record.get("textContent") or "").strip()
        title = str(record.get("title") or "").strip()
        searchable = f"{title}\n{text}".lower()
        if record.get("isTrashed") or not text or not any(keyword.lower() in searchable for keyword in keywords):
            continue
        edited_at = int(record.get("userEditedTimestampUsec") or record.get("createdTimestampUsec") or 0)
        candidates.append((edited_at, path, record))

    notes: list[RawNote] = []
    for index, (edited_at, path, record) in enumerate(sorted(candidates, reverse=True)[:limit], start=1):
        title = str(record.get("title") or "").strip()
        text = str(record["textContent"]).strip()
        full_text = f"{title}\n{text}" if title and title not in text else text
        lower_text = full_text.lower()
        tags = [keyword for keyword in keywords if keyword.lower() in lower_text]
        notes.append(RawNote(
            id=f"keep-{index:03d}", text=full_text, created_at=str(edited_at), tags=tags or [path.stem],
        ))
    return notes


def export_keep_notes(source_directory: str | Path, destination: str | Path, limit: int = 100) -> list[RawNote]:
    notes = import_keep_notes(source_directory, limit=limit)
    write_jsonl(destination, notes)
    return notes