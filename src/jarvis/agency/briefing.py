"""Turning signals into something George would actually write.

The briefing is the twin's one unprompted output, so it has to sound like
him and not like a status page. Two rules follow from that:

Say what is missing. A brief assembled while Gmail was unreachable looks
identical to a brief assembled on a quiet morning, and the difference
matters enormously. Every degraded source is named.

Never invent a quiet day. If nothing was reachable, the honest output is
"I could not check", not "nothing to report" — those are opposite claims and
only one of them is supported by the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from jarvis.agency.signals import Signal, Source, Urgency, dedupe, rank, should_speak

#: Greek weekday and month names, indexed from the datetime values.
_WEEKDAYS = (
    "Δευτέρα", "Τρίτη", "Τετάρτη", "Πέμπτη",
    "Παρασκευή", "Σάββατο", "Κυριακή",
)
_MONTHS = (
    "Ιανουαρίου", "Φεβρουαρίου", "Μαρτίου", "Απριλίου", "Μαΐου", "Ιουνίου",
    "Ιουλίου", "Αυγούστου", "Σεπτεμβρίου", "Οκτωβρίου", "Νοεμβρίου",
    "Δεκεμβρίου",
)

_SOURCE_LABEL: dict[Source, str] = {
    Source.CALENDAR: "το ημερολόγιο",
    Source.EMAIL: "τα email",
    Source.GITHUB: "το GitHub",
    Source.WEATHER: "ο καιρός",
    Source.CONVERSATION: "οι συνομιλίες",
    Source.SYSTEM: "το σύστημα",
}


def greek_date(when: datetime) -> str:
    return f"{_WEEKDAYS[when.weekday()]} {when.day} {_MONTHS[when.month - 1]}"


@dataclass(frozen=True)
class Briefing:
    """The message and the reason for it, decided together.

    These were two functions computing the same decision separately, and
    they disagreed: one returned text while the other reported
    ``silent_below_threshold``. Anything derived from a decision has to come
    out of the same evaluation of it, or the pair drifts the moment either
    side is edited.
    """

    text: str
    #: "spoke" | "spoke_degraded" | "silent_below_threshold"
    #: | "silent_nothing_observed"
    status: str

    @property
    def sent(self) -> bool:
        return bool(self.text)


def build_briefing(
    signals: list[Signal],
    unavailable: list[Source] | None = None,
    already_said: set[str] | None = None,
    now: datetime | None = None,
) -> Briefing:
    """Decide whether to speak, and write the message if so.

    Empty text is a valid and common result. The caller sends nothing at
    all rather than "nothing to report", which costs the same attention as
    real news and delivers none.
    """
    reference = now or datetime.now(timezone.utc)
    missing = list(unavailable or [])
    fresh = dedupe(signals, already_said)

    if not fresh and not missing:
        return Briefing("", "silent_nothing_observed")

    worth_saying = should_speak(fresh, already_said=already_said, now=reference)

    # A degraded run always speaks, even below threshold. "I could not check
    # your calendar" is information; withholding it leaves the user believing
    # the day is clear because nothing arrived.
    if not worth_saying and not missing:
        return Briefing("", "silent_below_threshold")

    status = "spoke" if worth_saying and not missing else "spoke_degraded"

    lines = [f"Καλημέρα. {greek_date(reference)}."]

    if fresh:
        lines.append("")
        for signal in rank(fresh, reference):
            prefix = "→"
            if signal.urgency is Urgency.BLOCKING:
                prefix = "!"
            elif signal.urgency is Urgency.TIME_BOUND:
                prefix = "⏱"
            line = f"{prefix} {signal.summary}"
            if signal.detail:
                line += f" — {signal.detail}"
            lines.append(line)

    # Named explicitly, and last, so it reads as a caveat on everything
    # above rather than as another item.
    if missing:
        labels = ", ".join(_SOURCE_LABEL.get(s, s.value) for s in missing)
        lines.append("")
        if fresh:
            lines.append(f"Δεν μπόρεσα να δω {labels} — μπορεί να μου ξεφεύγει κάτι.")
        else:
            lines.append(
                f"Δεν μπόρεσα να δω {labels}, οπότε δεν ξέρω αν έχεις κάτι σήμερα."
            )

    return Briefing("\n".join(lines), status)


def compose_briefing(*args, **kwargs) -> str:
    """The text alone, for callers that do not need the reason."""
    return build_briefing(*args, **kwargs).text


def briefing_status(*args, **kwargs) -> str:
    """The reason alone.

    Kept so a silent morning is distinguishable from a broken scheduler:
    "no message arrived" has at least three causes, and without a reported
    status they look identical from outside.
    """
    return build_briefing(*args, **kwargs).status
