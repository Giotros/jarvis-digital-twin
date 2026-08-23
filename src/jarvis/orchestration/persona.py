"""Relationship-conditioned register.

The twin should not answer a supervisor the way it answers a childhood
friend. This module turns two free-text fields collected at the start of a
session — a name and an *ιδιότητα* ("what this person is to me") — into a
prompt fragment and a set of generation parameters.

**Why register and not per-person personalisation.** The obvious design is
one persona per contact, learned from that contact's thread. Measuring the
corpus first showed that would be building on nothing: across George's eight
most frequent correspondents the pairwise style distance is 0.041 on average
and 0.083 at most, well inside the 0.15 band that
:func:`jarvis.evaluation.metrics.style_distance` treats as "the same voice".
He writes to his friends in one voice.

The exception is the single business contact, who receives 9.1 words per
message against a 5.4 average and 0.23 questions per message against 0.08.
The variation that exists is therefore between *kinds* of relationship, not
between individuals, and that is the granularity implemented here: four
registers, chosen by keyword.

The word targets are scaled up from those per-message figures, because a
chat message is not a reply. George sends three short messages in a row
where the twin sends one; 5.4 words is what a fragment of a turn looks like,
not a turn. The measured *ratio* is preserved — professional is roughly
three times close — and ``Register.measured`` marks which targets rest on
corpus evidence at all. The academic register does not: George has never had
a viva over Viber, so its numbers are stated as chosen, not measured.

**Nothing is stored.** The name and ιδιότητα live in the request that
carries them and in the caller's session memory. They are never written to
disk, never added to the corpus, and never used for training. A third
party's name that is merely echoed back within one conversation is not a
record; one that is persisted is, and the distinction is the whole reason
this module has no I/O.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


def _fold(text: str) -> str:
    """Lower-case and strip diacritics.

    ``casefold`` alone is not enough: it maps final sigma to sigma, so a
    keyword list written naturally ("συνάδελφος") would fail to match the
    accusative the user actually types ("συνάδελφο"). Folding accents makes
    the keyword table readable *and* matchable.
    """
    decomposed = unicodedata.normalize("NFD", text.casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


@dataclass(frozen=True)
class Register:
    """One conversational register.

    Length is stated in exactly one place: ``target_words``. ``instructions``
    must not also claim a sentence count. The first version had ΥΦΟΣ saying
    "2-4 προτάσεις" while ΜΗΚΟΣ said "12 λέξεις" — quantities that cannot
    both hold — and a model resolving that contradiction is not following
    either.

    ``measured`` records whether the number came from the corpus. Three of
    the four did; the academic register has no corpus evidence behind it,
    because George has never had this conversation over Viber, and the thesis
    has to be able to say which is which.
    """

    name: str
    label: str
    #: Words per reply.
    target_words: int
    #: Questions per reply in the same subset.
    target_question_rate: float
    #: True when the targets come from the corpus, False when chosen.
    measured: bool
    instructions: str
    temperature: float = 0.6
    max_new_tokens: int = 150
    keywords: frozenset[str] = field(default_factory=frozenset)
    #: ``(question, answer)`` pairs injected as prior turns.
    #:
    #: Instructions alone did not work. The adapter was trained on 13k casual
    #: Viber messages, and that training dominates a sentence of prompt: asked
    #: to address a professor, the model replied "Ειμαι καλά αγορι μ να ξερς".
    #: Demonstrations are a much stronger signal than descriptions for a model
    #: fine-tuned this heavily — they speak in the same channel the training
    #: did, which a system prompt does not.
    examples: tuple[tuple[str, str], ...] = ()


CLOSE = Register(
    name="close",
    label="Κοντινό πρόσωπο",
    target_words=6,
    target_question_rate=0.08,
    measured=True,
    temperature=0.7,
    max_new_tokens=120,
    instructions=(
        "Μιλάς σε κοντινό σου άνθρωπο. Πολύ χαλαρά, σε δεύτερο ενικό. "
        "Μπορείς να είσαι αστείος ή απότομος, όπως με φίλο. Χωρίς εισαγωγές."
    ),
    keywords=frozenset(_fold(w) for w in {
        "φιλος", "φιλη", "κολλητος", "κολλητη", "αδερφος", "αδερφη",
        "αδελφος", "αδελφη", "ξαδερφος", "ξαδερφη", "ξαδελφος",
        "μπαμπας", "πατερας", "μαμα", "μητερα", "γιαγια", "παππους",
        "θειος", "θεια", "οικογενεια", "συγγενης", "κοπελα", "συντροφος",
        "γκομενα", "παρεα", "συμμαθητης", "συγκατοικος",
    }),
    examples=(
        ("τι κανεις;", "Καλά ρε, εσύ τι κάνεις;"),
        ("θα ερθεις αυριο;", "Ναι ρε, τι ώρα λέμε;"),
    ),
)

PROFESSIONAL = Register(
    name="professional",
    label="Επαγγελματική σχέση",
    target_words=18,
    target_question_rate=0.23,
    measured=True,
    temperature=0.5,
    max_new_tokens=220,
    instructions=(
        "Μιλάς σε επαγγελματικό πλαίσιο. Ευγενικά αλλά όχι τυπικά — "
        "δεύτερο ενικό, χωρίς πληθυντικό ευγενείας εκτός αν σου μιλήσουν έτσι. "
        "Απάντησε συγκεκριμένα και, αν χρειάζεται διευκρίνιση, ρώτησε. "
        "Χωρίς αργκό και χωρίς βωμολοχίες."
    ),
    keywords=frozenset(_fold(w) for w in {
        "συναδελφος", "συναδελφη", "συνεργατης", "πελατης", "πελατισσα",
        "προισταμενος", "διευθυντης", "εργοδοτης", "αφεντικο", "manager",
        "recruiter", "hr", "client", "colleague", "boss", "δουλεια",
        "εταιρεια", "επαγγελματικα", "επιχειρηση", "προμηθευτης",
    }),
    examples=(
        ("τι κανεις;",
         "Καλά είμαι, ευχαριστώ. Εσύ πώς πας με το project;"),
        ("μπορεις να το δεις μεχρι αυριο;",
         "Ναι, θα το κοιτάξω σήμερα το απόγευμα και σου απαντάω. "
         "Θέλεις να το δω όλο ή μόνο το κομμάτι που άλλαξε;"),
    ),
)

ACADEMIC = Register(
    name="academic",
    label="Ακαδημαϊκό πλαίσιο",
    target_words=40,
    target_question_rate=0.15,
    measured=False,
    temperature=0.4,
    max_new_tokens=320,
    instructions=(
        "Μιλάς σε ακαδημαϊκό πλαίσιο — καθηγητής, επιβλέπων ή εξεταστής. "
        "Σαφής και τεκμηριωμένος, στο πρώτο ενικό. Ολοκληρωμένες προτάσεις. "
        "Χωρίς αργκό, χωρίς βωμολοχίες, χωρίς υπερβολική οικειότητα. "
        "Αν δεν γνωρίζεις κάτι, πες το ευθέως αντί να το επινοήσεις."
    ),
    keywords=frozenset(_fold(w) for w in {
        "καθηγητης", "καθηγητρια", "επιβλεπων", "επιβλεπουσα", "εξεταστης",
        "επιτροπη", "μεταπτυχιακο", "διδακτορικο", "διπλωματικη",
        "πανεπιστημιο", "σχολη", "φοιτητης", "φοιτητρια", "ερευνητης",
        "professor", "supervisor", "phd", "msc", "ακαδημαικο",
    }),
    examples=(
        ("τι τεχνολογιες χρησιμοποιησες;",
         "Χρησιμοποίησα το Krikri-8B ως βασικό μοντέλο, με QLoRA fine-tuning "
         "σε 4-bit, και Ray για την κατανεμημένη εκπαίδευση. Η ανάκτηση "
         "γίνεται με υβριδική αναζήτηση BM25 και dense embeddings, και η "
         "ενορχήστρωση με n8n."),
        ("γιατι δεν χρησιμοποιησες GPT;",
         "Επειδή τα δεδομένα είναι προσωπικές συνομιλίες τρίτων. Ένα κλειστό "
         "μοντέλο θα σήμαινε αποστολή τους σε εξωτερικό πάροχο, που δεν "
         "μπορούσα να το δικαιολογήσω ως προς τον GDPR. Το Krikri τρέχει "
         "τοπικά και είναι εκπαιδευμένο στα ελληνικά."),
    ),
)

NEUTRAL = Register(
    name="neutral",
    label="Άγνωστος",
    target_words=15,
    target_question_rate=0.15,
    measured=False,
    temperature=0.55,
    max_new_tokens=180,
    instructions=(
        "Δεν ξέρεις ποιος σου μιλάει. Φιλικά αλλά συγκρατημένα, "
        "σε δεύτερο ενικό. Μη μοιράζεσαι προσωπικές λεπτομέρειες "
        "που δεν θα έλεγες σε κάποιον που μόλις γνώρισες."
    ),
    keywords=frozenset(),
    examples=(
        ("τι κανεις;", "Καλά είμαι, ευχαριστώ. Εσύ;"),
    ),
)

REGISTERS: tuple[Register, ...] = (CLOSE, PROFESSIONAL, ACADEMIC, NEUTRAL)

#: Checked in order. ACADEMIC precedes PROFESSIONAL because a supervisor is
#: both, and the academic framing is the one that matters at a viva.
_MATCH_ORDER: tuple[Register, ...] = (ACADEMIC, PROFESSIONAL, CLOSE)

#: Registers that receive the project facts *regardless of the question*.
#:
#: CLOSE was originally excluded on the argument that a friend asking "τι
#: έκανες φέτος" wants the person rather than a recitation of the method,
#: and that the worst case was an invented Arduino project — a story about
#: a lamp, not a false statement about the work.
#:
#: The measurement refuted that. Asked the same question as a friend, the
#: twin reported building "ένα chatbot σε Node.js χρησιμοποιώντας OpenAI
#: GPT-4" and using "Python και Django για το site της εταιρείας". Neither
#: is a lamp: one names a commercial API in a system whose entire premise
#: is that it runs locally, and the other names the wrong web framework for
#: this repository. Casual packaging does not make a claim less false, and
#: an examiner who tries the friend register — which is the register the
#: demo defaults to — reads it as the system's answer.
#:
#: Grounding is therefore driven by the *question*, not the listener; see
#: ``is_technical_question``. This set now only forces facts in for the two
#: registers where a technical question is likely even when the phrasing
#: does not look like one.
_GROUNDED_REGISTERS: frozenset[str] = frozenset({"academic", "professional"})

#: Question shapes that must be answered from the facts file.
#:
#: Deliberately broader than "asks about the thesis". The failing probe was
#: "με τι τεχνολογίες δούλεψες φέτος" — which never mentions the thesis, and
#: which the model answered by assembling a plausible-sounding CV. Any
#: question that invites a list of tools is a question that can be answered
#: with invented tools.
_TECHNICAL_QUESTION_PATTERNS: tuple[str, ...] = (
    r"τεχνολογ",           # τι τεχνολογίες, ποιες τεχνολογίες
    r"εργαλει",            # τι εργαλεία
    r"framework|βιβλιοθηκ",
    r"γλωσσ\w*\s+προγραμ|προγραμματισμ",
    r"στοιβα|stack\b",
    r"διπλωματικ|πτυχιακ|εργασια\s+σου|thesis",
    r"μοντελ\w*\s|\bmodel\b|krikri|llama|qlora|lora\b",
    r"εκπαιδευ|training|fine.?tun",
    r"αρχιτεκτονικ|υλοποιησ|implementation",
    r"δουλεψ\w*\s+(?:με|φετος|πανω)|ασχοληθηκ|χρησιμοποι",
    r"τι\s+εκανες\s+(?:φετος|στη|στο|για)",
    r"πως\s+(?:το\s+)?(?:εφτιαξ|υλοποιησ|εκανες|δουλευει)",
    r"γιατι\s+(?:διαλεξ|επελεξ)",
)

#: The patterns are folded before compiling, not written pre-folded.
#:
#: ``_fold`` maps final sigma to sigma, so "εκανες" written naturally becomes
#: "εκανεσ" in the haystack and a pattern spelling it with ς matches nothing —
#: silently, since a regex that never fires looks exactly like a question that
#: never gets asked. This project has now shipped that bug five times, in five
#: different modules, and every occurrence was found by a human reading output
#: rather than by a failing test.
#:
#: Folding the patterns through the same function as the text makes the two
#: impossible to disagree. Regex metacharacters survive it untouched: ``\s``,
#: ``\w``, ``(?:`` and ``|`` carry no case and no diacritics.
_TECHNICAL_QUESTION_RE = re.compile(
    "|".join(_fold(p) for p in _TECHNICAL_QUESTION_PATTERNS)
)


def is_technical_question(message: str) -> bool:
    """True when the answer should come from the facts file, not the weights.

    Folded before matching, so the patterns are written without accents and
    still match the accented, capitalised, or final-sigma forms a real user
    types.
    """
    if not message:
        return False
    return bool(_TECHNICAL_QUESTION_RE.search(_fold(message)))

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def classify_relationship(role: str) -> Register:
    """Pick a register from a free-text ιδιότητα.

    Matching is on whole folded words, plus a prefix check so that inflected
    forms resolve: the user types "συνάδελφο" or "καθηγητή", never the
    dictionary form. An empty or unrecognised value yields :data:`NEUTRAL`,
    which is deliberately the *narrower* persona — an unknown interlocutor
    is the case where over-sharing costs the most.
    """
    if not role or not role.strip():
        return NEUTRAL

    tokens = [_fold(t) for t in _WORD_RE.findall(role)]
    if not tokens:
        return NEUTRAL

    for register in _MATCH_ORDER:
        for token in tokens:
            if token in register.keywords:
                return register
            # Inflected form: "καθηγητη" against the stored "καθηγητης".
            if any(
                len(token) >= 5
                and (kw.startswith(token) or token.startswith(kw[:-1]))
                for kw in register.keywords
                if len(kw) >= 5
            ):
                return register
    return NEUTRAL


def _clean_name(name: str) -> str:
    """Keep a plausible first name and drop anything else.

    The field is free text on a public demo, so it is treated as untrusted:
    letters, spaces and hyphens only, one token, length-capped. This stops a
    pasted paragraph — or a prompt injection — from entering the system
    prompt through a form the examiners will be invited to type into.
    """
    if not name:
        return ""
    tokens = _WORD_RE.findall(name.strip())
    if not tokens:
        return ""
    first = tokens[0][:24]
    return first.capitalize()


#: Everything about George that does not change with the audience.
IDENTITY_PROMPT = (
    "Είσαι ο Γιώργος Τροχίδης, φοιτητής στο Πανεπιστήμιο Πελοποννήσου στην Τρίπολη.\n"
    "Μιλάς ΠΑΝΤΑ σε πρώτο ενικό πρόσωπο — εσύ ΕΙΣΑΙ ο Γιώργος, ΟΧΙ AI ή βοηθός."
)

#: Rules that hold no matter who is asking.
RULES_PROMPT = (
    "ΚΑΝΟΝΕΣ: Μην επινοείς πληροφορίες. Αν δεν ξέρεις κάτι, πες 'δεν θυμάμαι' ή 'δεν ξέρω'."
)


def build_system_prompt(
    name: str = "",
    role: str = "",
    identity: str = "",
    rules: str = "",
    message: str = "",
) -> tuple[str, Register]:
    """Assemble the complete system prompt for one interlocutor.

    The style section is *generated* from the register rather than appended
    after a fixed one. The first version appended it, and the three registers
    produced near-identical replies: the base prompt already asserted "ΥΦΟΣ:
    Φιλικό, casual, σαν να μιλάς σε φίλο" and "ΜΗΚΟΣ: 1-3 προτάσεις", so the
    academic register was asking an 8B model to resolve a direct
    contradiction between two capitalised instructions. It resolved it by
    obeying the first one, every time.

    Identity and rules stay invariant — who George is does not depend on who
    is asking, only how he says it does.
    """
    register = classify_relationship(role)
    clean = _clean_name(name)

    parts = [identity or IDENTITY_PROMPT]

    if clean:
        parts.append(
            f"ΣΥΝΟΜΙΛΗΤΗΣ: Σου μιλάει ο/η {clean} — {register.label.lower()}. "
            "Μπορείς να τον προσφωνήσεις με το όνομά του, χωρίς υπερβολή."
        )

    parts.append(f"ΥΦΟΣ: {register.instructions}")
    parts.append(
        f"ΜΗΚΟΣ: Γύρω στις {register.target_words} λέξεις. "
        "Μην ξεπερνάς αισθητά αυτό το μήκος."
    )
    parts.append(rules or RULES_PROMPT)

    # Ground the technical registers in the real project. An examiner asks
    # technical questions, and the model answers them by inventing a stack
    # that sounds right for a student. Facts are cheap to supply and the
    # alternative — a confident wrong answer about the method under
    # examination — is the most expensive failure this system has.
    #
    # Professional is included because grounding only academic was not
    # enough: asked the same question as a colleague, the twin described a
    # different thesis entirely — "πρόβλεψη τιμών ενέργειας μέσω deep
    # learning", with Django and PostgreSQL. Whoever asks a technical
    # question deserves the same technical truth; only the tone should differ.
    #
    # The question is now also consulted, because the listener turned out to
    # be the weaker signal of the two. "Με τι τεχνολογίες δούλεψες φέτος"
    # asked in the close register produced GPT-4 and Django — a false
    # statement about the work, delivered casually. Tone is a property of the
    # register; truth is not.
    if register.name in _GROUNDED_REGISTERS or is_technical_question(message):
        from jarvis.inference.thesis_facts import (
            load_thesis_facts,
            load_thesis_facts_brief,
        )

        # The size of the evidence has to match the size of the answer.
        #
        # The full block is ~715 words. Given to the close register — whose
        # measured target is six — it produced a correct 37-word
        # specification sheet: true in every particular and in nobody's
        # voice. The model reproduces the shape of what it is given, and no
        # ΜΗΚΟΣ line survives contact with 700 words pulling the other way.
        #
        # So the short registers get a short grounding. Less is stated, and
        # what is stated is still true — which is the whole requirement.
        # Only the formal registers, where reciting is the correct
        # behaviour, receive the full file.
        facts = (
            load_thesis_facts()
            if register.target_words >= 18
            else load_thesis_facts_brief()
        )
        if facts:
            parts.append("")
            parts.append(facts)
            # Repeated after the block as well as inside it. The facts run to
            # ~580 words of YAML, and the single closing instruction sat at
            # the end of that — the position an 8B model is least likely to
            # weight. Stating the constraint on both sides of the evidence
            # costs two lines.
            parts.append(
                "ΥΠΕΝΘΥΜΙΣΗ: Τα παραπάνω είναι ΟΛΑ όσα ξέρεις για την "
                "υλοποίηση. Μην προσθέσεις εργαλείο, μοντέλο, πλατφόρμα ή "
                "αριθμό που δεν γράφεται εκεί, ούτε καν αν σου φαίνεται "
                "προφανές. Το «δεν το θυμάμαι ακριβώς» είναι σωστή απάντηση· "
                "ένα εργαλείο που δεν χρησιμοποίησες δεν είναι."
            )

    return "\n".join(parts), register


def describe_registers() -> str:
    """Markdown table of the registers and their measured targets."""
    rows = [
        "| Register | Ιδιότητα | Λέξεις | Ερωτήσεις | temp |",
        "|---|---|---:|---:|---:|",
    ]
    for r in REGISTERS:
        rows.append(
            f"| `{r.name}` | {r.label} | {r.target_words} | "
            f"{r.target_question_rate:.2f} | {r.temperature} |"
        )
    return "\n".join(rows)
