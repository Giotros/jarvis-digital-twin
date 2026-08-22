"""Proactive behaviour: deciding when the twin should speak unprompted.

Everything else in this system is reactive — a message arrives, an answer
goes back. The thesis brief asks for autonomy, and autonomy means the twin
sometimes starts the conversation.

The hard part is not fetching data. It is restraint. An assistant that
reports everything it notices is a notification firehose, and people mute
those within a day; the failure is silent, because the system keeps working
perfectly while nobody reads it. So the salience model here is deliberately
conservative and its threshold is a stated parameter rather than a constant
buried in a function.
"""

from jarvis.agency.signals import (  # noqa: F401
    Signal,
    Source,
    Urgency,
    dedupe,
    salience,
    should_speak,
)

from jarvis.agency.briefing import (  # noqa: F401,E402
    Briefing,
    build_briefing,
    compose_briefing,
)
