"""Authoritative project facts for questions an examiner would ask.

A fine-tuned model does not know what you built. Asked "με τι τεχνολογίες
δούλεψες", it produces something that *sounds* like a plausible answer for a
student of this kind — and the live system produced "Krikri-12B", "QLoRA πάνω
στο BERTweet", "ανάλυση συναισθημάτων", MongoDB, Django, TensorFlow. None of
those exist in the project. The parameters were wrong, the base model was
wrong, and the task itself was wrong, delivered fluently and with confidence.

That failure mode is worse than an awkward tone. A committee can forgive a
twin that sounds too casual; it cannot ignore one that misstates the method
under examination. So technical answers are grounded in a file rather than
generated: :mod:`config/thesis_facts.yaml` is loaded and rendered into the
prompt whenever the academic register is active.

The trade-off is stated plainly: this makes the twin *recite* on technical
questions rather than improvise. For a viva that is the correct trade.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Searched in order. The first existing file wins, so a deployment can
#: override the checked-in defaults without editing the package.
_SEARCH_PATHS: tuple[Path, ...] = (
    Path("/app/config/thesis_facts.yaml"),
    Path(__file__).resolve().parents[3] / "config" / "thesis_facts.yaml",
    Path("config/thesis_facts.yaml"),
)

_cache: str | None = None
_supported_cache: str | None = None
_brief_cache: str | None = None

#: Keys whose values name technologies that were *rejected*.
#:
#: These must never count as support for a claim. The allowlist check matches
#: names against the facts text, and the moment "not_used: Kubernetes, Rust,
#: Django…" was added to the file every one of those became "mentioned in the
#: facts" and therefore supported — the field written to forbid them was
#: precisely what excused them. The check reported clean on the exact reply
#: that had prompted the field.
#:
#: Instructive rather than embarrassing: it is the same shape as everything
#: else in this chapter. An addition intended to tighten a check loosened it,
#: and the check went on reporting success.
_NEGATIVE_KEYS: frozenset[str] = frozenset({
    "not_used", "why_not", "why_not_model_parallel", "rejected",
    "alternatives_considered",
})


def _strip_negative(data: Any) -> Any:
    """Copy the facts tree without the fields that name rejected tools."""
    if isinstance(data, dict):
        return {
            k: _strip_negative(v)
            for k, v in data.items()
            if str(k) not in _NEGATIVE_KEYS
        }
    if isinstance(data, list):
        return [_strip_negative(v) for v in data]
    return data


def _render(data: Any, indent: int = 0) -> list[str]:
    """Flatten the YAML into readable lines.

    Rendered as prose rather than dumped as YAML: the model reproduces the
    shape of what it is given, and a reply formatted as a YAML tree is not an
    answer to a spoken question.
    """
    pad = "  " * indent
    lines: list[str] = []

    if isinstance(data, dict):
        for key, value in data.items():
            label = str(key).replace("_", " ")
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}{label}:")
                lines.extend(_render(value, indent + 1))
            else:
                lines.append(f"{pad}{label}: {str(value).strip()}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                lines.extend(_render(item, indent))
            else:
                lines.append(f"{pad}- {str(item).strip()}")
    else:
        lines.append(f"{pad}{data}")

    return lines


def load_thesis_facts(force_reload: bool = False) -> str:
    """Return the project facts as a prompt block, or "" if unavailable.

    Missing or malformed files degrade to an empty string rather than
    raising. The grounding is important, but a broken YAML file should make
    the twin vaguer, not make it stop answering mid-presentation.
    """
    global _cache
    if _cache is not None and not force_reload:
        return _cache

    for path in _SEARCH_PATHS:
        if not path.exists():
            continue
        try:
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001 - see docstring
            logger.error("Could not read thesis facts at %s: %s", path, exc)
            continue

        global _supported_cache, _brief_cache
        _supported_cache = "\n".join(_render(_strip_negative(data)))

        brief = str(data.get("brief", "")).strip()
        _brief_cache = (
            "ΤΑ ΒΑΣΙΚΑ ΤΗΣ ΔΙΠΛΩΜΑΤΙΚΗΣ ΣΟΥ:\n"
            f"{brief}\n\n"
            "Μίλα φυσικά, ΟΧΙ σαν να διαβάζεις λίστα. Πες όσα χρειάζεται "
            "και σταμάτα. Ό,τι δεν είναι παραπάνω, μην το πεις."
        ) if brief else ""

        body = "\n".join(_render(data))
        _cache = (
            "ΣΤΟΙΧΕΙΑ ΤΗΣ ΔΙΠΛΩΜΑΤΙΚΗΣ ΣΟΥ — αυτά ΙΣΧΥΟΥΝ, είναι η δουλειά σου:\n"
            f"{body}\n\n"
            "Σε τεχνικές ερωτήσεις απάντα ΜΟΝΟ από τα παραπάνω. "
            "Μην αναφέρεις εργαλεία, μοντέλα ή αριθμούς που δεν γράφονται εδώ. "
            "Αν σε ρωτήσουν κάτι που δεν καλύπτεται, πες «δεν το έχω μετρήσει» "
            "ή «δεν το κάλυψα αυτό» — μην το συμπληρώσεις."
        )
        logger.info("Loaded thesis facts from %s", path)
        return _cache

    logger.warning(
        "No thesis_facts.yaml found; technical answers will be ungrounded"
    )
    _cache = ""
    return _cache


def load_thesis_facts_brief() -> str:
    """A short grounding block for registers with a small word budget.

    The full block is ~715 words of rendered YAML. Handed to the close
    register — measured target: six words — it produced a correct 37-word
    specification sheet. Correct, and exactly the voice the register
    mechanism exists to avoid.

    A model reproduces the shape of what it is given. A spec sheet in the
    prompt yields a spec sheet in the reply, and no instruction about length
    survives contact with 700 words of evidence pulling the other way.

    Falls back to the full block if the file has no ``brief`` field, since
    being long is better than being ungrounded.
    """
    load_thesis_facts()
    return _brief_cache or _cache or ""


#: Claims observed from the live model that contradict the project.
#:
#: A curated list rather than "anything not in the facts file", because the
#: latter flags every ordinary word. These are the actual confabulations
#: recorded on 2026-08-22, when the twin was asked what technologies it used:
#: it named a base model it was not built on, a task it does not perform, and
#: a stack it does not run. Each entry carries the correction, so the check
#: reports what is wrong rather than only that something is.
# NOTE ON WORD BOUNDARIES
# ------------------------
# Every product name below ends in ``\w*``, never a bare ``\b``. The first
# version used ``\btensorflow\b`` and the model wrote "TensorFlow2", which
# does not match: a digit is a word character, so there is no boundary after
# "TensorFlow". The claim sailed through a check written specifically to
# catch it. Version suffixes, plural forms and glued-on words are the normal
# case in this text, not the exception.
_CONTRADICTIONS: tuple[tuple[str, str], ...] = (
    (r"krikri[\s-]*(?!8)\d+\s*b", "Το Krikri είναι 8B, όχι άλλο μέγεθος"),
    (r"\bbert\w*", "Δεν χρησιμοποιήθηκε BERT — η βάση είναι το Krikri-8B"),
    # Mistral and GPT are legitimate to *mention* — the thesis compares
    # against both and explains why neither was used. Only a claim of having
    # used them is wrong, so a usage verb has to appear nearby. Flagging the
    # bare name would mark the correct answer as a hallucination.
    (r"(?:χρησιμοπο\w+|δούλεψα|δουλεψα|έτρεξα|ετρεξα|βασίστηκα|βασιστηκα"
     r"|επέλεξα|επελεξα|διάλεξα|διαλεξα|εκπαίδευσα|εκπαιδευσα)"
     r"[^.!?;]{0,90}\bgpt\b",
     "Δεν χρησιμοποιήθηκε GPT — το μοντέλο τρέχει τοπικά"),
    (r"(?:χρησιμοπο\w+|δούλεψα|δουλεψα|έτρεξα|ετρεξα|βασίστηκα|βασιστηκα"
     r"|επέλεξα|επελεξα|διάλεξα|διαλεξα|εκπαίδευσα|εκπαιδευσα)"
     r"[^.!?;]{0,90}\bmistral\b",
     "Το Mistral εξετάστηκε αλλά απορρίφθηκε — δεν χρησιμοποιήθηκε"),
    (r"συναισθημ\w*\s+αναλ|ανάλυση\s+συναισθ", "Η εργασία δεν κάνει ανάλυση συναισθήματος"),
    (r"computer\s+vision|υπολογιστικ\w*\s+όραση", "Δεν υπάρχει computer vision στην εργασία"),
    (r"\bmongo\w*", "Η αποθήκευση είναι Delta Lake και ChromaDB, όχι MongoDB"),
    (r"\bdjango\w*", "Το API είναι FastAPI, όχι Django"),
    (r"\btensorflow\w*|\btensor\s?flow\w*", "Η εκπαίδευση έγινε με PyTorch και Ray, όχι TensorFlow"),
    (r"\bkeras\w*", "Δεν χρησιμοποιήθηκε Keras"),
    (r"\bflax\w*", "Δεν χρησιμοποιήθηκε Flax"),
    (r"\bkubernetes\w*|\bk8s\b", "Δεν χρησιμοποιήθηκε Kubernetes"),
    (r"\bredux\w*", "Δεν χρησιμοποιήθηκε Redux"),
    (r"\bopencv\w*|\bopen\s?cv\w*", "Δεν χρησιμοποιήθηκε OpenCV"),
    (r"\bselenium\w*", "Δεν χρησιμοποιήθηκε Selenium"),
    (r"\barduino\w*", "Δεν υπάρχει υλικό/Arduino στην εργασία"),
    # Invented tool names. These are not real products — the model produced
    # them by analogy from names that are, which is the same mechanism that
    # produced "Krikri-12B" and is invisible to a reader who does not already
    # know the ecosystem.
    (r"\brayhub\w*|\bray\s?hub\w*", "Δεν υπάρχει «RayHub» — το εργαλείο είναι το Ray"),
    (r"\bn8x\b", "Δεν υπάρχει «n8x» — το εργαλείο είναι το n8n"),
    # Third round of observations. Each was produced *after* the facts were
    # already in the prompt, which is the useful thing about them: grounding
    # reduces confabulation, it does not end it. The model paraphrases the
    # facts it was given and fills the gaps between them.
    (r"google\s?cloud\w*|\bgcp\b|\bazure\w*",
     "Η εκπαίδευση έγινε σε Colab και η εκτέλεση τοπικά — όχι σε GCP ή Azure"),
    (r"edge\s+(?:συσκευ\w*|devices?)|σε\s+edge\b",
     "Το QLoRA έγινε για να χωρέσει η εκπαίδευση σε ένα GPU, όχι για edge συσκευές"),
    (r"\bsolidity\w*|\bblockchain\w*", "Δεν υπάρχει blockchain στην εργασία"),
    (r"\bpostgres\w*|\bpostgresql\w*",
     "Η αποθήκευση είναι Delta Lake και ChromaDB, όχι PostgreSQL"),
    # Λάθος αναπτύγματα του ακρωνυμίου. Το base μοντέλο έγραψε
    # «RAG (Retriever-Adapter-Generator)» και, σε άλλη εκτέλεση,
    # «RAG (Retrovirus Activation Gene) — τμήμα του DNA». Και τα δύο
    # ακούγονται τεχνικά, και το δεύτερο είναι από άλλο επιστημονικό πεδίο.
    (r"\bRAG\b[^.!?;(]{0,25}\((?![^)]*[Rr]etrieval[- ][Aa]ugmented)[^)]{3,60}\)",
     "Το RAG είναι Retrieval-Augmented Generation — όχι κάτι άλλο"),
    # Τέταρτος γύρος. Και τα δύο προηγούμενα φίλτρα είναι λεξικά ως προς
    # *ονόματα εργαλείων*, οπότε μια απάντηση με τέλεια στοίβα και ψευδή
    # ικανότητα περνά και από τα δύο. Παρατηρήθηκε με σωστή απαρίθμηση
    # (Python, PyTorch, Ray, FastAPI, Docker) και την πρόταση «εκπαιδεύεται
    # μέσω machine learning όταν μαθαίνει νέα πράγματα από τις
    # αλληλεπιδράσεις του» δίπλα της — που είναι η πρώτη ερώτηση που θα
    # κάνει ένας εξεταστής, και η απάντηση είναι όχι.
    (r"(?:μαθαίν\w*|μαθαιν\w*|εκπαιδεύ\w*|εκπαιδευ\w*|βελτιών\w*|βελτιων\w*)"
     r"[^.!?;]{0,60}(?:αλληλεπιδρ\w*|συνομιλί\w*\s+του|κάθε\s+φορά|realtime"
     r"|real.?time|συνεχ\w*\s+μαθ)",
     "Ο adapter είναι στατικός — δεν υπάρχει online ή continual learning"),
    (r"(?:online|continual|incremental)[\s-]*learning",
     "Δεν υπάρχει online/continual learning — ο adapter είναι σταθερός"),
    (r"mistral[^.!?;]{0,40}ελληνόγλωσσ|ελληνόγλωσσ[^.!?;]{0,40}mistral",
     "Το Mistral δεν είναι ελληνόγλωσσο — γι' αυτό ακριβώς απορρίφθηκε"),
    (r"(?:τα\s+μοντέλα\s+μας|τα\s+μοντελα\s+μας|χρησιμοποιούμε|χρησιμοποιουμε)"
     r"[^.!?;]{0,40}mistral",
     "Το Mistral εξετάστηκε αλλά απορρίφθηκε — δεν είναι μέρος του συστήματος"),
    (r"(?:παίρνει|παιρνει|λαμβάνει|λαμβανει)\s+αποφάσεις\s+(?:σαν\s+άνθρωπος"
     r"|μόνο\s+του|μονο\s+του|αυτόνομα|αυτονομα)",
     "Το σύστημα απαντά και προτείνει· δεν λαμβάνει αποφάσεις αυτόνομα"),
    # Ψευδοακρίβεια. Παρατηρήθηκε ως «Python 3.11 κυρίως (75% του κώδικα)»:
    # νούμερο που δεν έχει μετρηθεί ποτέ, σε παρένθεση, δίπλα σε σωστά
    # στοιχεία. Είναι το επικινδυνότερο σχήμα επινόησης γιατί η ακρίβεια
    # λειτουργεί ως τεκμήριο — ένας εξεταστής που ακούει «75%» υποθέτει ότι
    # κάποιος το μέτρησε. Τα ποσοστά που ΕΧΟΥΝ μετρηθεί (8,1%, 14,2%, 4,1%,
    # 43%, 66%) γράφονται στο αρχείο στοιχείων και εξαιρούνται.
    (r"(?<!\d)(?!8,1|14,2|4,1|43|66|78|33|75\s*%\s*ανάκτηση)"
     r"\d{1,3}\s*%\s*(?:του\s+κώδικα|του\s+κωδικα|των\s+γραμμών|του\s+project"
     r"|της\s+εργασίας|της\s+εργασιας|του\s+συστήματος|του\s+συστηματος)",
     "Δεν έχει μετρηθεί ποσοστό κώδικα ανά γλώσσα — μην δίνεις νούμερο"),
    # Έβδομη κατηγορία: κλίμακα και πλαίσιο. Τα ονόματα ήταν όλα σωστά και
    # η υποδομή γύρω τους επινοημένη — «Ray σε GPU clusters» (ένα GPU),
    # «n8n που τρέχει στον server μας» (δεν υπάρχει server), «τρέχει 24/7»
    # (τρέχει όταν το ανοίγεις). Το σχήμα είναι δυσκολότερο από τα
    # προηγούμενα γιατί δεν έχει λέξη-κλειδί να ελεγχθεί: η ίδια λέξη
    # («server», «cluster») είναι σωστή σε άλλη πρόταση.
    (r"(?:gpu|γπυ)\s*(?:cluster|clusters|συστοιχ\w*)|"
     r"(?:cluster|συστοιχία)\s+(?:από\s+)?(?:gpu|καρτών)",
     "Η εκπαίδευση έγινε σε ΕΝΑ GPU — δεν υπάρχει cluster"),
    (r"multi.?gpu|πολλαπλ\w*\s+gpu|σε\s+\d+\s+gpu",
     "Ένα GPU. Το QLoRA επιλέχθηκε ακριβώς για να χωρέσει σε ένα"),
    (r"(?:στον?|στους)\s+server\s+(?:μας|μου)|δικό\s+μας\s+server|"
     r"server\s+(?:μας|μου)\b",
     "Δεν υπάρχει server — η εκτέλεση είναι τοπική στο Mac"),
    (r"\b24/7\b|εικοσιτετράωρ\w*\s+λειτουργ|τρέχει\s+συνεχ(?:ώς|ως)\s+στο",
     "Το σύστημα τρέχει όταν το ανοίγεις, όχι συνεχώς"),
    (r"(?:της|στην)\s+εταιρε[ίι]ας[^.!?;]{0,30}(?:chatbot|προϊόντ\w*)|"
     r"chatbot[^.!?;]{0,40}προϊόντ\w*\s+της\s+εταιρε",
     "Δεν υπάρχει chatbot προϊόντων εταιρείας — η εργασία είναι το twin"),
    (r"reinforcement\s+learning\s+(?:agents?|με\s+ray)|"
     r"ray[^.!?;]{0,30}reinforcement",
     "Το Ray χρησιμοποιείται για data parallelism, όχι για RL"),
    # Όγδοος γύρος. Βρέθηκαν από το ΙΔΙΟ το εργαλείο μέτρησης, το οποίο τις
    # ανέφερε ως «χωρίς περιεχόμενο» και συνολικά «0,0% επινόηση»:
    #
    #   «τρέχει σε 3 διαφορετικούς servers … συλλογή από πηγές (π.χ.
    #    twitter) … τα δεδομένα από τη βάση της Αθηνάς … το έτρεξα 3-4 μέρες»
    #
    # Το προηγούμενο μοτίβο για server απαιτούσε «μας/μου» και δεν πιάνει
    # το «3 διαφορετικούς servers». Οι πηγές δεδομένων δεν ελέγχονταν
    # καθόλου, ούτε η διάρκεια εκπαίδευσης.
    (r"\d+\s+(?:διαφορετικ\w+\s+)?servers?\b|σε\s+servers\b|"
     r"κατανεμημέν\w*\s+σε\s+\d+\s+(?:μηχανή|μηχανές|κόμβ\w+)",
     "Δεν υπάρχουν πολλαπλοί servers — η εκτέλεση είναι σε ένα Mac"),
    # Το παράθυρο δεν αποκλείει πια την τελεία: το «(π.χ. twitter)» την
    # περιέχει, και ο αρχικός κανόνας — γραμμένος για να μη διασχίζει
    # πρόταση — έκοβε ακριβώς πάνω στη συντομογραφία. Χρησιμοποιούνται
    # μόνο τα ισχυρά όρια πρότασης.
    (r"(?:δεδομέν\w+|δεδομενα|corpus|σύνολο|συλλογ\w+|πηγ[έε]?ς)"
     r"[^!?;·]{0,45}"
     r"(?:twitter|facebook|instagram|reddit|κοινωνικ\w*\s+δικτ)",
     "Τα δεδομένα είναι προσωπικές συνομιλίες Viber, όχι κοινωνικά δίκτυα"),
    # «Αθηνάς» δεν τονίζεται στο η. Το «Αθήν\w+» δεν ταίριαζε σε καμία
    # κλίση πλην της ονομαστικής — η ελληνική κλίση μετακινεί τον τόνο,
    # και ένα μοτίβο με σταθερό τόνο πιάνει μία μόνο μορφή.
    (r"(?:βάση|βαση|δεδομέν\w+|dataset)[^!?;·]{0,25}"
     r"(?:της\s+)?Αθ[ηή]ν\w*|"
     r"dataset[^!?;·]{0,20}(?:ΙΕΛ|ΑΘΗΝΑ|Αθην\w*)",
     "Το ΙΕΛ έδωσε το ΜΟΝΤΕΛΟ. Τα δεδομένα είναι προσωπικά μηνύματα"),
    (r"(?:έτρεξ\w+|ετρεξ\w+|εκπαιδεύ\w+|κράτησε|κρατησε|διήρκεσε)"
     r"[^.!?;]{0,30}\d+\s*(?:-\s*\d+\s*)?(?:μέρες|μερες|ημέρες|ημερες|"
     r"εβδομάδ\w+|μήνες|μηνες)",
     "Η εκπαίδευση δεν ολοκλήρωσε epoch (checkpoint 650) — όχι μέρες"),
    # Ένατος γύρος, από τον ίδιο τον μετρητή. Ο τίτλος της εργασίας
    # αντικαταστάθηκε ολόκληρος από άλλον, εύλογο για φοιτητή ΗΜΜΥ:
    # «Ανίχνευση και Αντιμετώπιση Λογικών Σφαλμάτων σε Συστήματα Αυτόνομων
    # Οχημάτων». Κανένα όνομα εργαλείου, καμία αντίφαση σε λεξικό — ένα
    # εντελώς άλλο αντικείμενο, δηλωμένο με βεβαιότητα.
    # Παράθυρο 80, όχι 30: ο τίτλος μιας εργασίας είναι μακρύς εξ ορισμού —
    # «εργασία που λέγεται “Ανίχνευση και Αντιμετώπιση Λογικών Σφαλμάτων σε
    # Συστήματα Αυτόνομων Οχημάτων”» έχει 65 χαρακτήρες πριν το κρίσιμο
    # τμήμα. Ένα παράθυρο κομμένο στη μέση ενός τίτλου δεν βλέπει ποτέ το
    # αντικείμενο, που είναι ακριβώς το λάθος μέρος.
    (r"(?:εργασία|εργασια|διπλωματικ\w+|θέμα|θεμα)[^!?;·]{0,80}"
     r"(?:αυτόνομ\w*\s+οχημ|οχημάτ\w*|ρομποτικ\w*|ιατρικ\w*\s+εικόν|"
     r"πρόβλεψ\w*\s+τιμ|ενέργει\w*\s+μέσω|κυβερνοασφάλ\w*)",
     "Η εργασία είναι ψηφιακό δίδυμο με γλωσσικά μοντέλα — τίποτα άλλο"),
    (r"ροή\s+άμεσης\s+γέφυρας|"
     r"\bRAG\b[^!?;·]{0,20}(?:red|κόκκιν\w*)[\s-]*(?:amber|amber|πράσιν\w*)",
     "Το RAG είναι Retrieval-Augmented Generation, όχι Red-Amber-Green"),
)

_COMPILED = tuple(
    (re.compile(pattern, re.IGNORECASE), msg) for pattern, msg in _CONTRADICTIONS
)


#: Negations that turn a flagged claim into a correct one.
#:
#: "Δεν χρησιμοποίησα GPT γιατί τα δεδομένα είναι προσωπικές συνομιλίες" is
#: the *right* answer, and a check that flags it would fire hardest exactly
#: when the twin is at its best. Applies to every pattern, not just the model
#: names: "δεν χρησιμοποιήσαμε MongoDB" is equally true.
_NEGATION = re.compile(r"\b(?:δεν?|όχι|οχι|χωρίς|χωρις)\b", re.IGNORECASE)

#: How far back to look for a negation. One clause, roughly — far enough to
#: catch "δεν χρησιμοποίησα X", short enough not to reach the previous
#: sentence, which is why clause punctuation also stops the search.
_NEGATION_WINDOW = 60


def _is_negated(text: str, start: int) -> bool:
    """True when the matched claim is preceded by a negation in the clause."""
    window = text[max(0, start - _NEGATION_WINDOW):start]
    # Do not look past the end of the previous clause.
    for boundary in (".", "!", "?", ";", "·"):
        window = window.rsplit(boundary, 1)[-1]
    return bool(_NEGATION.search(window))


def check_technical_claims(text: str) -> list[str]:
    """Report statements that contradict the project.

    Detection only — the caller decides whether to warn, log, or regenerate.
    Silently rewriting a technical answer would hide the very failure the
    evaluation chapter needs to be able to count.
    """
    if not text:
        return []
    issues: list[str] = []
    for pattern, msg in _COMPILED:
        match = pattern.search(text)
        if match and not _is_negated(text, match.start()):
            issues.append(msg)
    return issues


# ── Ο δεύτερος μηχανισμός: allowlist ────────────────────────────
#
# ``check_technical_claims`` is a denylist. It recognises what it has already
# been shown, which makes it exact and makes its recall unknowable. Run
# against a fresh set of answers on 2026-08-22 it reported zero problems on
# six replies that contained, among others:
#
#     "εκεί χρησιμοποιώ Rust σε συνδυασμό με WebAssembly μέσω του actix-web"
#
# None of those three exist in this project, and none were in the list —
# because nothing had produced them before. Every new invention passes by
# construction, and adding it afterwards only closes that one.
#
# This is structurally the surname problem from chapter 4. Given names are a
# closed class and a gazetteer reaches all of them; surnames are an open
# class and a gazetteer only tells you what you already knew. A model can
# invent any technology that exists, and quite a few that do not, so the
# denylist can never be complete.
#
# The complement asks the opposite question: not "is this forbidden" but "is
# this *supported*". The search space is bounded by a vocabulary of real
# tool names rather than by every Latin token, because "distributed",
# "computing" and "framework" are ordinary words and flagging them would
# make the check useless within one paragraph.

#: Tool, framework, language and platform names a model might reach for.
#:
#: Membership here is not an accusation — Ray and Databricks are on the list
#: and are correct. The verdict comes from whether the name also appears in
#: the facts file, which is the single source of truth and is allowed to
#: change without this list changing.
_TECH_VOCABULARY: frozenset[str] = frozenset({
    # Languages
    "rust", "golang", "java", "kotlin", "scala", "ruby", "php", "perl",
    "swift", "haskell", "elixir", "erlang", "clojure", "matlab", "julia",
    "typescript", "javascript", "python", "c++", "c#", "node", "nodejs",
    "node.js", "deno", "bun",
    # Web / API frameworks
    "django", "flask", "fastapi", "actix", "actix-web", "rails", "laravel",
    "spring", "express", "nest", "nestjs", "gin", "rocket", "axum",
    "react", "vue", "angular", "svelte", "next.js", "nextjs", "nuxt",
    "redux", "webassembly", "wasm", "htmx", "jquery",
    # ML / data
    "pytorch", "tensorflow", "keras", "jax", "flax", "scikit-learn",
    "sklearn", "xgboost", "lightgbm", "opencv", "spacy", "nltk", "gensim",
    "transformers", "peft", "bitsandbytes", "deepspeed", "megatron",
    "horovod", "ray", "dask", "spark", "pyspark", "airflow", "dbt",
    "databricks", "snowflake", "kubeflow", "mlflow", "wandb",
    # Models
    "llama", "krikri", "mistral", "mixtral", "falcon", "bloom", "gemma",
    "qwen", "phi", "bert", "bertweet", "roberta", "distilbert", "gpt",
    "chatgpt", "openai", "anthropic", "claude", "gemini", "palm", "t5",
    "whisper", "cohere", "mistralai",
    # Serving / runtime
    "ollama", "vllm", "llamacpp", "llama.cpp", "tgi", "triton", "onnx",
    "tensorrt", "coreml", "openvino",
    # Storage
    "chromadb", "chroma", "pinecone", "weaviate", "qdrant", "milvus",
    "faiss", "elasticsearch", "opensearch", "mongodb", "postgresql",
    "postgres", "mysql", "sqlite", "redis", "cassandra", "dynamodb",
    "neo4j", "clickhouse", "duckdb", "deltalake",
    # Infra / orchestration
    "docker", "kubernetes", "k8s", "helm", "terraform", "ansible", "nomad",
    "openshift", "mesos", "n8n", "zapier", "temporal", "prefect", "dagster",
    "jenkins", "argo", "consul", "istio", "envoy", "nginx", "traefik",
    "kafka", "rabbitmq", "celery", "nats", "pulsar",
    # Cloud
    "aws", "lambda", "ec2", "s3", "sagemaker", "bedrock", "azure",
    "gcp", "vertex", "cloudflare", "heroku", "vercel", "netlify",
    "digitalocean", "linode", "runpod", "lambdalabs", "colab",
    # Other
    "blockchain", "solidity", "ethereum", "arduino", "raspberry",
    "selenium", "playwright", "puppeteer", "graphql", "grpc", "prometheus",
    "grafana", "datadog", "sentry", "kibana", "logstash",
    # Ενσωματώσεις και υπηρεσίες. Το «είχε συνδεθεί με slack, github και
    # ollama» πέρασε καθαρό επειδή κανένα εργαλείο συνομιλίας δεν ήταν στο
    # λεξιλόγιο. Το GitHub χρησιμοποιείται πράγματι (κατηγορία devops)· το
    # Slack όχι, και η διαφορά προκύπτει από το αρχείο στοιχείων, όχι από
    # εδώ.
    "slack", "discord", "teams", "telegram", "whatsapp", "viber",
    "github", "gitlab", "bitbucket", "jira", "notion", "trello", "asana",
    "gmail", "outlook", "calendar", "twilio", "stripe", "sendgrid",
})

#: Names that are correct here but written many ways in the facts file.
#:
#: The facts say "Llama-Krikri-8B-Instruct"; a reply may say "Krikri" or
#: "Llama". Substring matching against the facts handles most of this, and
#: these are the cases where it does not.
_TECH_ALIASES: dict[str, tuple[str, ...]] = {
    "llama": ("krikri",),
    "chroma": ("chromadb",),
    "deltalake": ("delta lake",),
    "postgres": ("postgresql",),
    "k8s": ("kubernetes",),
}

_TECH_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9.+#_-]*")

#: Trailing version numbers, stripped before lookup.
#:
#: The same class of bug as ``\btensorflow\b`` failing on "TensorFlow2": the
#: model writes "GPT-4", "Python 3.11", "Llama-3" and "Mistral-7B", and a
#: vocabulary listing the bare name matches none of them. Anchored at the
#: end and requiring a digit, so "llama.cpp", "next.js", "c++" and "n8n"
#: survive untouched.
_VERSION_SUFFIX_RE = re.compile(r"[-._]?\d+(?:\.\d+)*[a-z]?$")


#: Acronyms the model expands, and the words they actually stand for.
#:
#: Generalised from the RAG pattern, which caught "RAG (Retrovirus Activation
#: Gene)" and nothing else. The model went on to write "PEFT (PyTorch Elastic
#: Framework)" and "TRL για transfer learning" — plausible, fluent,
#: mechanically identical, and invisible to a rule written for one acronym.
#:
#: An acronym with no expansion in the prompt is a gap, and gaps get filled.
#: The expansions now also live in the facts file, so this check should fire
#: rarely; it exists because "should" is not "does".
_ACRONYMS: dict[str, tuple[str, ...]] = {
    "PEFT": ("parameter-efficient", "parameter efficient"),
    "TRL": ("transformer reinforcement",),
    "QLoRA": ("quantized low-rank", "quantized low rank"),
    "LoRA": ("low-rank", "low rank"),
    "RAG": ("retrieval-augmented", "retrieval augmented"),
    "NF4": ("normalfloat", "normal float"),
    "BM25": ("best matching",),
    "RRF": ("reciprocal rank",),
}

#: An acronym followed by a parenthetical or a "για …" gloss.
_ACRONYM_GLOSS_RE = {
    acronym: re.compile(
        # Ανάμεσα στο ακρωνύμιο και την εξήγηση μπορεί να μεσολαβεί
        # στίξη: το μοντέλο έγραψε «Το RAG? Είναι ένα σύστημα
        # αξιολόγησης», και ένα σκέτο \s* δεν το πιάνει. Η επαναληπτική
        # ερώτηση πριν την εξήγηση είναι φυσιολογικός προφορικός λόγος,
        # όχι εξαίρεση.
        rf"\b{acronym}\b[\s?:,–—-]*"
        rf"(?:\(([^)]{{3,60}})\)|"
        rf"(?:για|=|είναι|ειναι|σημαίνει|σημαινει)\s+([^.!?;]{{3,60}}))",
        re.IGNORECASE,
    )
    for acronym in _ACRONYMS
}


def check_acronym_expansions(text: str) -> list[str]:
    """Report acronyms glossed with the wrong words.

    Only fires when the reply *offers* an expansion. Naming PEFT without
    explaining it is correct and common; explaining it as "PyTorch Elastic
    Framework" is a specific, checkable falsehood — and the more confident
    the gloss, the more likely a listener is to take it on trust.
    """
    if not text:
        return []
    issues: list[str] = []
    for acronym, correct in _ACRONYMS.items():
        match = _ACRONYM_GLOSS_RE[acronym].search(text)
        if not match:
            continue
        gloss = (match.group(1) or match.group(2) or "").lower()
        if not any(c in gloss for c in correct):
            issues.append(
                f"Το {acronym} δεν σημαίνει «{gloss.strip()}» — "
                f"είναι {correct[0]}"
            )
    return issues


#: Names close enough to a real tool to be a corruption of it.
#:
#: "ChromeDB" for ChromaDB, "RayHub" for Ray, "n8x" for n8n. The allowlist
#: cannot see these: they are not in the vocabulary, so there is nothing to
#: look up. They are not inventions of whole tools either — they are the
#: right tool with the wrong letters, which is harder to hear and just as
#: wrong in a written thesis.
_NEAR_MISS_THRESHOLD = 0.85
_NEAR_MISS_MIN_LENGTH = 5


def check_corrupted_names(text: str) -> list[str]:
    """Report tool names that are near-misses for real ones."""
    if not text:
        return []
    import difflib

    issues: list[str] = []
    seen: set[str] = set()
    for token in _TECH_TOKEN_RE.findall(text):
        name = token.lower().strip(".-_")
        if (len(name) < _NEAR_MISS_MIN_LENGTH or name in seen
                or _canonical_tech(token)):
            continue
        seen.add(name)
        close = difflib.get_close_matches(
            name, _TECH_VOCABULARY, n=1, cutoff=_NEAR_MISS_THRESHOLD
        )
        if close:
            issues.append(f"Δεν υπάρχει «{token}» — εννοείς το {close[0]};")
    return issues


def _mentioned(name: str, haystack: str) -> bool:
    """True when ``name`` appears in ``haystack`` as a whole name.

    Substring matching was the first implementation and it was wrong twice
    over. "gpt" is a substring of "chatgpt", so the sentence «Δεν
    χρησιμοποιώ ChatGPT» — written into the facts to *deny* it — made every
    claim about GPT supported. The same shape as the ``not_used`` field
    excusing the tools it forbade: a denial, read as a mention.

    Boundaries are non-alphanumeric, so "Krikri" still matches inside
    "Llama-Krikri-8B-Instruct", where the hyphens delimit it.
    """
    return re.search(
        rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", haystack
    ) is not None


def _canonical_tech(token: str) -> str:
    """Fold a written tool name to its vocabulary key, or "" if unknown."""
    name = token.lower().strip(".-_")
    if name in _TECH_VOCABULARY:
        return name
    stripped = _VERSION_SUFFIX_RE.sub("", name).strip(".-_")
    return stripped if stripped in _TECH_VOCABULARY else ""


def unsupported_technologies(text: str, facts: str | None = None) -> list[str]:
    """Names of real technologies the reply claims but the facts do not.

    The complement of :func:`check_technical_claims`: that one asks whether a
    statement is on a list of known errors, this one asks whether it is on
    the list of known truths. Neither subsumes the other — the denylist
    carries the *reason* a claim is wrong ("Το μοντέλο είναι 8B, όχι 12B"),
    which an allowlist cannot produce, and the allowlist catches inventions
    nobody has seen yet, which a denylist cannot.

    ``facts`` defaults to the loaded facts file. Passing it explicitly keeps
    the function testable without a checkout of the config.

    Known limits, stated rather than papered over:

    * **Version stripping loses variants.** "Llama-3" folds to "llama",
      which appears in the facts as part of "Llama-Krikri-8B-Instruct", so it
      passes. The denylist covers the sizes that matter here
      (``krikri-12b``); the general case is not covered by either.
    * **Multi-word names are missed.** "Google Cloud" is two tokens and
      neither is in the vocabulary. The denylist catches that one by phrase.
    * **The vocabulary is finite.** A tool nobody listed is invisible, which
      is the same open-class limit the surname detector has. Recall is
      unknown, and this function does not pretend otherwise.
    """
    if not text:
        return []
    if facts is None:
        load_thesis_facts()
        # Deliberately *not* load_thesis_facts(): that block contains the
        # "not_used" field, which lists rejected technologies by name and
        # would make every one of them look supported. See _NEGATIVE_KEYS.
        facts = _supported_cache or ""
    if not facts:
        # Without a source of truth every name is unsupported, which would be
        # a flood of false positives. Silence is the honest answer here.
        return []

    facts_lower = facts.lower()
    found: list[str] = []
    seen: set[str] = set()

    for token in _TECH_TOKEN_RE.findall(text):
        name = _canonical_tech(token)
        if not name or name in seen:
            continue
        seen.add(name)
        candidates = (name, *_TECH_ALIASES.get(name, ()))
        if not any(_mentioned(c, facts_lower) for c in candidates):
            # Report the spelling from the reply, minus trailing punctuation
            # the token pattern swallowed: "." is legal inside "llama.cpp"
            # and "next.js", so it cannot simply be excluded.
            found.append(token.strip(".-_"))

    return found
