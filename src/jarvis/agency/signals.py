"""What the twin noticed, and whether any of it is worth saying.

A signal is one observation from one source — a meeting starting soon, an
unanswered message, a failing build. Signals are cheap to produce and
expensive to deliver: each one the twin volunteers spends a little of the
user's attention, and that budget is not renewable.

The decision to speak is therefore separated from the decision of what to
say. :func:`should_speak` is the gate; composing the message happens only
after it opens.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum


class Source(str, Enum):
    """Where an observation came from.

    Recorded per signal so a briefing can say which tools were reachable —
    and, more usefully, which were not. A brief assembled while Gmail was
    down looks identical to one assembled on a quiet morning unless the
    difference is carried through.
    """

    CALENDAR = "calendar"
    EMAIL = "email"
    GITHUB = "github"
    WEATHER = "weather"
    CONVERSATION = "conversation"
    SYSTEM = "system"


class Urgency(int, Enum):
    """How much of the attention budget an observation justifies."""

    BACKGROUND = 1   # true, uninteresting: "no meetings today"
    NOTABLE = 2      # worth knowing: "three unread from the supervisor"
    TIME_BOUND = 3   # decays if ignored: "meeting in 40 minutes"
    BLOCKING = 4     # someone is waiting on George right now


@dataclass(frozen=True)
class Signal:
    """One observation, ready to be weighed against the others."""

    source: Source
    summary: str
    urgency: Urgency = Urgency.NOTABLE
    #: When the thing happens, not when it was observed. Used to decay
    #: relevance: a meeting that ended an hour ago is no longer news.
    occurs_at: datetime | None = None
    #: Free-form origin marker (message id, event id). Used for deduplication
    #: when present, which is more reliable than comparing summary text.
    key: str = ""
    detail: str = ""
    tags: frozenset[str] = field(default_factory=frozenset)

    def identity(self) -> str:
        """A stable handle for this observation.

        Falls back to a hash of the normalised summary when the source gives
        no id, so a reworded-but-identical observation still collapses.
        """
        if self.key:
            return f"{self.source.value}:{self.key}"
        folded = unicodedata.normalize("NFD", self.summary.casefold())
        folded = "".join(c for c in folded if not unicodedata.combining(c))
        digest = hashlib.sha256(" ".join(folded.split()).encode()).hexdigest()
        return f"{self.source.value}:{digest[:16]}"


#: Below this, the twin says nothing at all.
#:
#: Chosen so that a day with only BACKGROUND signals stays silent no matter
#: how many of them there are: five "nothing happening" observations score
#: 5 × 1 × 0.5 = 2.5, under the threshold. One TIME_BOUND item alone scores
#: 3.0 and gets through. The number is a parameter rather than a constant
#: because the right value depends on how often the user wants to hear from
#: it, which is not something the code can know.
DEFAULT_THRESHOLD: float = 3.0

#: Weight applied per urgency level. Deliberately not linear in count:
#: fifteen background items are not more important than one blocking one,
#: and a sum over raw urgencies would say they are.
_URGENCY_WEIGHT: dict[Urgency, float] = {
    Urgency.BACKGROUND: 0.5,
    Urgency.NOTABLE: 1.0,
    Urgency.TIME_BOUND: 3.0,
    Urgency.BLOCKING: 4.0,
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def time_decay(signal: Signal, now: datetime | None = None) -> float:
    """Relevance multiplier based on how soon the thing happens.

    Something happening in twenty minutes is worth interrupting for.
    The same thing next week is not, and the same thing yesterday is noise.
    Signals with no time attached are unaffected.
    """
    if signal.occurs_at is None:
        return 1.0

    reference = now or _now()
    occurs = signal.occurs_at
    if occurs.tzinfo is None:
        occurs = occurs.replace(tzinfo=timezone.utc)

    delta = occurs - reference

    if delta < timedelta(0):
        # Already happened. Kept rather than dropped so a briefing can still
        # mention it in passing, but it cannot on its own justify speaking.
        return 0.2
    if delta <= timedelta(hours=1):
        return 1.5
    if delta <= timedelta(hours=8):
        return 1.0
    if delta <= timedelta(days=1):
        return 0.7
    return 0.3


def salience(signals: list[Signal], now: datetime | None = None) -> float:
    """Total weight of a set of observations."""
    return sum(
        _URGENCY_WEIGHT[s.urgency] * time_decay(s, now) for s in signals
    )


def should_speak(
    signals: list[Signal],
    threshold: float = DEFAULT_THRESHOLD,
    already_said: set[str] | None = None,
    now: datetime | None = None,
) -> bool:
    """Whether the twin has enough reason to start a conversation.

    Repeats are removed before weighing, not after. Otherwise the same
    unread email would clear the threshold every hour, and the twin would
    become a system that reports the same thing until it is switched off —
    which is the specific failure this gate exists to prevent.
    """
    fresh = dedupe(signals, already_said)
    if not fresh:
        return False
    return salience(fresh, now) >= threshold


def dedupe(
    signals: list[Signal], already_said: set[str] | None = None
) -> list[Signal]:
    """Drop repeats, keeping the first occurrence and preserving order.

    Order is preserved because the caller may have sorted by importance, and
    silently reordering someone's list is the kind of helpfulness that
    causes bugs two layers away.
    """
    seen = set(already_said or ())
    out: list[Signal] = []
    for signal in signals:
        marker = signal.identity()
        if marker in seen:
            continue
        seen.add(marker)
        out.append(signal)
    return out


def rank(signals: list[Signal], now: datetime | None = None) -> list[Signal]:
    """Most worth saying first.

    Sorted by individual weight rather than by source, so a blocking email
    outranks a routine calendar entry instead of being filed under "email".
    """
    return sorted(
        signals,
        key=lambda s: _URGENCY_WEIGHT[s.urgency] * time_decay(s, now),
        reverse=True,
    )
