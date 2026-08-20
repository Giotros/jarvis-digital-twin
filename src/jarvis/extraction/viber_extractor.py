"""Viber backup → normalized message records (Phase 1, bronze input).

Output format is one JSON object per line (JSONL), matching the schema of
bronze.viber_messages_raw on Databricks:

    {"timestamp": "...ISO-8601...", "chat_id": "...", "sender": "...",
     "is_me": true/false, "text": "..."}

Supported inputs:
  * CSV export            → ViberExtractor.from_csv(path, mapping)
  * JSON export           → ViberExtractor.from_json(path, mapping)
  * Viber Desktop SQLite  → ViberExtractor.from_sqlite(path, query)

Because Viber's export schema differs between versions, column names are
passed as a mapping instead of being hardcoded — adjust in one place
(config/settings.yaml or the call site) rather than editing parser code.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_CSV_MAPPING = {
    "timestamp": "Date",
    "sender": "Sender",
    "text": "Message",
    "chat_id": "Chat",
}

#: Works for common viber.db layouts; override via `query=` if yours differs.
DEFAULT_SQLITE_QUERY = """
    SELECT m.timestamp AS timestamp,
           c.name      AS sender,
           m.body      AS text,
           m.chat_id   AS chat_id,
           m.is_outgoing AS is_me
    FROM messages m
    LEFT JOIN contacts c ON c.id = m.sender_id
    ORDER BY m.timestamp
"""


@dataclass
class Message:
    timestamp: str          # ISO-8601, UTC
    chat_id: str
    sender: str
    is_me: bool
    text: str

    @staticmethod
    def normalize_timestamp(value: str | int | float) -> str:
        """Accept epoch (s or ms) or common string formats → ISO-8601 UTC."""
        if isinstance(value, (int, float)):
            if value > 1e12:                       # milliseconds
                value = value / 1000.0
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
        value = str(value).strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y, %H:%M",
                    "%d/%m/%Y %H:%M", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc).isoformat()
            except ValueError:
                continue
        return value                                # keep as-is; silver layer re-parses


class ViberExtractor:
    """Parses Viber backups into normalized :class:`Message` records."""

    def __init__(self, me: str) -> None:
        """*me* — the sender name/identifier that marks George's own messages."""
        self.me = me

    # -- parsers ------------------------------------------------------------

    def from_csv(
        self, path: str | Path, mapping: dict[str, str] | None = None,
        encoding: str = "utf-8-sig",
    ) -> Iterator[Message]:
        cols = {**DEFAULT_CSV_MAPPING, **(mapping or {})}
        with open(path, newline="", encoding=encoding) as f:
            for row in csv.DictReader(f):
                text = (row.get(cols["text"]) or "").strip()
                if not text:
                    continue
                sender = (row.get(cols["sender"]) or "").strip()
                yield Message(
                    timestamp=Message.normalize_timestamp(row.get(cols["timestamp"], "")),
                    chat_id=(row.get(cols["chat_id"]) or "unknown").strip(),
                    sender=sender,
                    is_me=sender == self.me,
                    text=text,
                )

    def from_json(
        self, path: str | Path, mapping: dict[str, str] | None = None,
    ) -> Iterator[Message]:
        cols = {**DEFAULT_CSV_MAPPING, **(mapping or {})}
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(records, dict):               # some exports wrap in {"messages": []}
            records = records.get("messages", [])
        for row in records:
            text = str(row.get(cols["text"]) or "").strip()
            if not text:
                continue
            sender = str(row.get(cols["sender"]) or "").strip()
            yield Message(
                timestamp=Message.normalize_timestamp(row.get(cols["timestamp"], "")),
                chat_id=str(row.get(cols["chat_id"]) or "unknown"),
                sender=sender,
                is_me=sender == self.me,
                text=text,
            )

    def from_sqlite(self, path: str | Path, query: str = DEFAULT_SQLITE_QUERY) -> Iterator[Message]:
        conn = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            for row in conn.execute(query):
                text = (row["text"] or "").strip()
                if not text:
                    continue
                sender = (row["sender"] or "").strip()
                is_me = bool(row["is_me"]) if "is_me" in row.keys() else sender == self.me
                yield Message(
                    timestamp=Message.normalize_timestamp(row["timestamp"]),
                    chat_id=str(row["chat_id"]),
                    sender=sender or ("me" if is_me else "unknown"),
                    is_me=is_me,
                    text=text,
                )
        finally:
            conn.close()

    # -- output -------------------------------------------------------------

    @staticmethod
    def write_jsonl(messages: Iterable[Message], path: str | Path) -> int:
        """Write messages as JSONL (bronze-layer input). Returns count."""
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for msg in messages:
                f.write(json.dumps(asdict(msg), ensure_ascii=False) + "\n")
                n += 1
        return n
