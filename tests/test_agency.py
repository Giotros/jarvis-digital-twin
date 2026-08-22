"""Tests for proactive behaviour — mostly tests that it stays quiet."""

from datetime import datetime, timedelta, timezone

import pytest

from jarvis.agency.briefing import briefing_status, compose_briefing, greek_date
from jarvis.agency.signals import (
    DEFAULT_THRESHOLD,
    Signal,
    Source,
    Urgency,
    dedupe,
    rank,
    salience,
    should_speak,
    time_decay,
)

NOW = datetime(2026, 8, 24, 8, 0, tzinfo=timezone.utc)


def sig(summary, urgency=Urgency.NOTABLE, source=Source.EMAIL, offset=None, key=""):
    return Signal(
        source=source,
        summary=summary,
        urgency=urgency,
        occurs_at=NOW + offset if offset else None,
        key=key,
    )


# ── Restraint ───────────────────────────────────────────────────

def test_a_quiet_day_stays_quiet():
    """Five true but uninteresting observations are still not news.

    This is the failure mode the gate exists for: a system that reports
    everything it notices keeps working perfectly while nobody reads it.
    """
    quiet = [sig(f"τίποτα {i}", Urgency.BACKGROUND) for i in range(5)]
    assert salience(quiet, NOW) < DEFAULT_THRESHOLD
    assert not should_speak(quiet, now=NOW)


def test_one_time_bound_item_is_enough():
    """A meeting in forty minutes justifies interrupting on its own."""
    assert should_speak(
        [sig("σύσκεψη με την ομάδα", Urgency.TIME_BOUND,
             Source.CALENDAR, timedelta(minutes=40))],
        now=NOW,
    )


def test_blocking_signal_always_speaks():
    assert should_speak([sig("περιμένουν απάντησή σου", Urgency.BLOCKING)], now=NOW)


def test_no_signals_means_silence():
    assert not should_speak([], now=NOW)


def test_threshold_is_a_parameter_not_a_constant():
    """How chatty the twin should be is the user's call, not the code's."""
    one = [sig("κάτι", Urgency.NOTABLE)]
    assert not should_speak(one, now=NOW)
    assert should_speak(one, threshold=0.5, now=NOW)


# ── Repetition ──────────────────────────────────────────────────

def test_the_same_item_is_not_reported_twice():
    """Otherwise the same unread email clears the threshold every hour.

    A system that reports the same thing until it is switched off is worse
    than one that never speaks, because it also trains the user to ignore
    the times it is right.
    """
    s = sig("αναπάντητο από τον επιβλέποντα", Urgency.BLOCKING, key="msg-1")
    assert should_speak([s], now=NOW)
    assert not should_speak([s], already_said={s.identity()}, now=NOW)


def test_rewording_does_not_defeat_deduplication():
    """Sources without stable ids get a hash of the normalised summary."""
    a = Signal(source=Source.EMAIL, summary="Τρία αδιάβαστα  μηνύματα")
    b = Signal(source=Source.EMAIL, summary="τρια αδιαβαστα μηνυματα")
    assert a.identity() == b.identity()


def test_an_explicit_key_beats_the_text():
    """Same event, differently phrased, still one event."""
    a = Signal(source=Source.CALENDAR, summary="Σύσκεψη 10:00", key="evt-9")
    b = Signal(source=Source.CALENDAR, summary="Ραντεβού το πρωί", key="evt-9")
    assert a.identity() == b.identity()


def test_dedupe_preserves_order():
    """The caller may have sorted by importance already."""
    a, b, c = sig("α", key="1"), sig("β", key="2"), sig("γ", key="3")
    assert [s.summary for s in dedupe([a, b, c, a])] == ["α", "β", "γ"]


# ── Time ────────────────────────────────────────────────────────

def test_imminent_outranks_distant():
    soon = sig("σύσκεψη", Urgency.TIME_BOUND, Source.CALENDAR, timedelta(minutes=30))
    later = sig("σύσκεψη", Urgency.TIME_BOUND, Source.CALENDAR, timedelta(days=6))
    assert time_decay(soon, NOW) > time_decay(later, NOW)


def test_past_events_cannot_justify_speaking_alone():
    """Yesterday's meeting is not a reason to start a conversation today."""
    past = sig("σύσκεψη", Urgency.TIME_BOUND, Source.CALENDAR, timedelta(hours=-20))
    assert not should_speak([past], now=NOW)


def test_signals_without_a_time_are_unaffected():
    assert time_decay(sig("κάτι"), NOW) == 1.0


def test_rank_puts_the_urgent_first_regardless_of_source():
    """A blocking email outranks a routine calendar entry."""
    routine = sig("τακτικό standup", Urgency.BACKGROUND, Source.CALENDAR)
    blocking = sig("περιμένουν εσένα", Urgency.BLOCKING, Source.EMAIL)
    assert rank([routine, blocking], NOW)[0] is blocking


# ── Composition ─────────────────────────────────────────────────

def test_silence_returns_an_empty_string():
    """The caller sends nothing — not "nothing to report"."""
    assert compose_briefing([sig("τίποτα", Urgency.BACKGROUND)], now=NOW) == ""


def test_unreachable_sources_are_named():
    """A brief written blind must not read like a brief written on a quiet day.

    Those are opposite claims, and only one of them is supported.
    """
    text = compose_briefing([], unavailable=[Source.EMAIL], now=NOW)
    assert "Δεν μπόρεσα να δω" in text
    assert "τα email" in text


def test_a_blind_brief_does_not_claim_the_day_is_clear():
    text = compose_briefing([], unavailable=[Source.CALENDAR], now=NOW)
    assert "δεν ξέρω αν έχεις κάτι" in text


def test_degraded_brief_still_reports_what_was_seen():
    text = compose_briefing(
        [sig("σύσκεψη στις 10", Urgency.TIME_BOUND, Source.CALENDAR,
             timedelta(hours=2))],
        unavailable=[Source.EMAIL],
        now=NOW,
    )
    assert "σύσκεψη στις 10" in text
    assert "μπορεί να μου ξεφεύγει κάτι" in text


@pytest.mark.parametrize("status,signals,unavailable", [
    ("silent_nothing_observed", [], None),
    ("silent_below_threshold", [sig("τίποτα", Urgency.BACKGROUND)], None),
    ("spoke_degraded", [], [Source.EMAIL]),
    ("spoke", [sig("επείγον", Urgency.BLOCKING)], None),
])
def test_status_distinguishes_the_reasons_for_silence(status, signals, unavailable):
    """"No message arrived" has at least three causes.

    Without a reported status they are indistinguishable, and a broken
    scheduler looks exactly like a quiet morning.
    """
    assert briefing_status(signals, unavailable, now=NOW) == status


def test_greek_date_is_declined_correctly():
    assert greek_date(datetime(2026, 8, 24)) == "Δευτέρα 24 Αυγούστου"
    assert greek_date(datetime(2026, 5, 1)) == "Παρασκευή 1 Μαΐου"


# ── Text and status must agree ──────────────────────────────────

from jarvis.agency.briefing import build_briefing  # noqa: E402


@pytest.mark.parametrize("signals,unavailable", [
    ([], None),
    ([sig("τίποτα", Urgency.BACKGROUND)], None),
    ([], [Source.EMAIL]),
    ([sig("τίποτα", Urgency.BACKGROUND)], [Source.EMAIL]),
    ([sig("επείγον", Urgency.BLOCKING)], None),
    ([sig("επείγον", Urgency.BLOCKING)], [Source.CALENDAR]),
])
def test_status_never_contradicts_the_text(signals, unavailable):
    """These were two functions deciding the same thing separately.

    They disagreed in production: one produced a message while the other
    reported "silent_below_threshold". A status that claims silence beside a
    sent message is worse than no status, because it is trusted.
    """
    b = build_briefing(signals, unavailable, now=NOW)
    assert b.sent == b.status.startswith("spoke")


def test_degraded_run_speaks_even_below_threshold():
    """"I could not check your calendar" is information.

    Withholding it because the remaining signals were dull leaves the user
    believing the day is clear, which is the one conclusion the evidence
    does not support.
    """
    b = build_briefing([sig("τίποτα", Urgency.BACKGROUND)],
                       [Source.CALENDAR], now=NOW)
    assert b.sent
    assert b.status == "spoke_degraded"
