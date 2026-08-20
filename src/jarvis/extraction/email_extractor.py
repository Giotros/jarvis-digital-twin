"""Email (mbox / Google Takeout) → instruction-response pairs (Phase 1).

Strategy: George's *sent* messages are the "assistant" side of a pair;
the message each one replies to (via In-Reply-To) is the "user" side.
Quoted reply blocks are stripped so the pair contains only fresh text.
"""

from __future__ import annotations

import mailbox
import re
from collections.abc import Iterator
from email.message import Message as EmailMessage
from pathlib import Path

_QUOTE_MARKERS = (
    re.compile(r"^\s*>"),                                   # "> quoted"
    re.compile(r"^On .{5,80} wrote:\s*$"),                  # EN reply header
    re.compile(r"^Στις .{5,80} έγραψε:\s*$"),               # GR reply header
    re.compile(r"^-{2,}\s*(Original|Forwarded) Message"),   # outlook style
)


def _plain_body(msg: EmailMessage) -> str:
    """Best-effort plain-text body of a (possibly multipart) email."""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace")
        return ""
    payload = msg.get_payload(decode=True)
    if not payload:
        return ""
    return payload.decode(msg.get_content_charset() or "utf-8", errors="replace")


def strip_quoted(text: str) -> str:
    """Drop quoted-reply blocks and signatures; keep the fresh text."""
    fresh: list[str] = []
    for line in text.splitlines():
        if any(marker.match(line) for marker in _QUOTE_MARKERS):
            break                                            # everything below is quote
        if line.strip() == "--":                             # signature delimiter
            break
        fresh.append(line)
    return "\n".join(fresh).strip()


class EmailExtractor:
    """Builds instruction/response pairs from an mbox archive."""

    def __init__(self, my_addresses: set[str]) -> None:
        self.my_addresses = {a.lower() for a in my_addresses}

    def _is_mine(self, msg: EmailMessage) -> bool:
        sender = (msg.get("From") or "").lower()
        return any(addr in sender for addr in self.my_addresses)

    def pairs_from_mbox(self, path: str | Path) -> Iterator[dict]:
        """Yield {"instruction", "response", "timestamp"} pairs."""
        box = mailbox.mbox(str(path))
        by_message_id: dict[str, EmailMessage] = {}
        for msg in box:
            mid = (msg.get("Message-ID") or "").strip()
            if mid:
                by_message_id[mid] = msg

        for msg in box:
            if not self._is_mine(msg):
                continue
            parent_id = (msg.get("In-Reply-To") or "").strip()
            parent = by_message_id.get(parent_id)
            if parent is None or self._is_mine(parent):
                continue
            instruction = strip_quoted(_plain_body(parent))
            response = strip_quoted(_plain_body(msg))
            if len(instruction) < 3 or len(response) < 3:
                continue
            yield {
                "instruction": instruction,
                "response": response,
                "timestamp": msg.get("Date", ""),
                "source": "email",
            }
