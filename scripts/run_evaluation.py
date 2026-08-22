#!/usr/bin/env python3
"""Run the thesis evaluation against the live local system.

Sends the frozen probe set through the *complete* pipeline (webhook → intent
→ routing → Krikri → guardrails), scores the answers with
:mod:`jarvis.evaluation.metrics`, and emits a thesis-ready markdown report
plus the raw JSON.

Why the webhook and not the model directly: the thesis claims to evaluate the
*system*, not the model in isolation. Guardrails truncate, the router picks
different context per intent, and latency includes orchestration. Measuring
the model alone would report numbers the user never experiences.

Usage:
    python3 scripts/run_evaluation.py
    python3 scripts/run_evaluation.py --repeats 3     # average over runs
    python3 scripts/run_evaluation.py --out docs/eval

Output:
    docs/evaluation_results.md    paste-ready tables
    docs/evaluation_results.json  raw data
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from jarvis.evaluation.metrics import (  # noqa: E402
    aggregate_report,
    style_profile,
    report_markdown,
)
from jarvis.evaluation.three_stage import PROBE_PROMPTS  # noqa: E402

WEBHOOK = "http://localhost:5678/webhook/twin-chat"
RAG_ENDPOINT = "http://localhost:8000/orchestration/rag"

#: Intents where retrieval supplies context and grounding can be scored.
#: For casual small-talk there is nothing to ground against, so scoring it
#: would report a fixed 1.0 "ungrounded" and say nothing about the system.
GROUNDABLE_INTENTS = {"knowledge", "personal", "memory", "schedule"}

#: Measured on 2026-08-20 with the base model and no corpus wired in.
#: Kept as the "before" column so the report shows movement, not just a
#: snapshot. See docs/thesis_skeleton.md §8.3.
BASELINE = {
    "label": "Base model, RAG χωρίς corpus",
    "ungrounded_rate": 1.00,
    "refusal_rate": 0.00,
    "first_person_rate": 0.40,
    "assistant_drift_rate": 0.00,
    "mean_latency_s": 6.28,
}


def ask(message: str, timeout: int = 90) -> tuple[str, str, float]:
    """Send one probe through the full pipeline.

    Returns ``(reply, intent, seconds)``. A failed call yields an empty reply
    rather than raising, so one bad probe does not abort the whole run — an
    empty answer is itself a measurable outcome (``empty_response_rate``).
    """
    payload = json.dumps({"message": message}).encode("utf-8")
    req = urllib.request.Request(
        WEBHOOK, data=payload, headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        elapsed = time.perf_counter() - started
        return data.get("reply", ""), data.get("intent", "?"), elapsed
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"    ⚠ {type(exc).__name__}: {exc}")
        return "", "error", time.perf_counter() - started


def fetch_context(query: str, timeout: int = 15) -> str:
    """Retrieve the context the RAG layer would supply for this query.

    Scored separately from generation so grounding is measured against what
    the system *actually had available*, not against nothing.
    """
    payload = json.dumps({"query": query, "top_k": 3}).encode("utf-8")
    req = urllib.request.Request(
        RAG_ENDPOINT, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")).get("context", "")
    except Exception:
        return ""


def load_reference_style():
    """Profile George's own writing from the golden examples.

    This is the target the twin is measured against. Without it the style
    numbers are absolute and uninterpretable; with it they are a distance
    from a real human baseline.
    """
    golden = ROOT / "config" / "golden_examples.yaml"
    if not golden.exists():
        return None
    try:
        import yaml
    except ImportError:
        return None
    data = yaml.safe_load(golden.read_text(encoding="utf-8")) or {}
    responses = [
        (e.get("response") or "").strip()
        for e in data.get("examples", [])
        if (e.get("response") or "").strip()
    ]
    return style_profile(responses) if responses else None


def comparison_table(report: dict) -> str:
    """Before/after table for §8.3 — the numbers that carry the chapter."""
    acc = report.get("accuracy", {})
    rel = report["reliability"]
    nat = report["naturalness"]["style_profile"]
    perf = report.get("performance", {})

    rows = [
        ("Ανυποστήρικτοι ισχυρισμοί", None,
         acc.get("unsupported_rate"), "↓ καλύτερο"),
        ("Αυτούσια αντιγραφή", None,
         acc.get("verbatim_rate"), "↓ καλύτερο"),
        ("Ungrounded rate (παλιό)", BASELINE["ungrounded_rate"],
         acc.get("legacy_ungrounded_rate"), "μη συγκρίσιμο"),
        ("Refusal rate", BASELINE["refusal_rate"],
         rel["refusal_rate"], "↑ υγιές"),
        ("First-person rate", BASELINE["first_person_rate"],
         nat["first_person_rate"], "↑ καλύτερο"),
        ("Assistant drift", BASELINE["assistant_drift_rate"],
         rel["assistant_drift_rate"], "↓ καλύτερο"),
        ("Mean latency (s)", BASELINE["mean_latency_s"],
         perf.get("mean_latency_s"), "↓ καλύτερο"),
    ]

    out = [
        "| Μετρική | Πριν | Μετά | Κατεύθυνση |",
        "|---|---:|---:|:---|",
    ]
    for name, before, after, direction in rows:
        # Two rows have no "before": the metrics did not exist for the
        # baseline run. Printing 0.00 there would invent a comparison.
        before_s = "—" if before is None else f"{before:.2f}"
        after_s = "—" if after is None else f"{after:.2f}"
        out.append(f"| {name} | {before_s} | **{after_s}** | {direction} |")

    n_g = acc.get("n_groundable", 0)
    n_t = acc.get("n_total", 0)
    out.append("")
    if n_g:
        out.append(f"Μετρήθηκε στις {n_g} από {n_t} απαντήσεις όπου το RAG "
                   "παρείχε context· η casual κουβέντα εξαιρείται γιατί δεν "
                   "έχει τι να τεκμηριώσει.")
    else:
        out.append("Δεν μετρήθηκε: καμία απάντηση δεν συνοδεύτηκε από "
                   "ανακτημένο context.")

    out += [
        "",
        "Το παλιό *ungrounded rate* μετρούσε λεξική επικάλυψη και δίνεται μόνο",
        "για ιστορική συνέχεια. Βαθμολογεί την αυτούσια αντιγραφή με 1,00 και",
        "μια σωστή παράφραση με 0,29 — ανταμείβει δηλαδή ακριβώς την αστοχία",
        "που διορθώθηκε. Οι δύο νέες γραμμές διαβάζονται μαζί: χαμηλοί",
        "ανυποστήρικτοι ισχυρισμοί *με* χαμηλή αντιγραφή είναι ο στόχος·",
        "χαμηλοί με υψηλή αντιγραφή σημαίνει ότι το σύστημα παπαγαλίζει.",
    ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1,
                    help="πόσες φορές να τρέξει το probe set (μέσος όρος)")
    ap.add_argument("--out", default="docs", help="φάκελος εξόδου")
    args = ap.parse_args()

    print(f"\nProbe set: {len(PROBE_PROMPTS)} ερωτήσεις × {args.repeats} "
          f"επανάληψη(εις)\nΕνδέχεται να πάρει λίγα λεπτά.\n")

    all_replies: list[str] = []
    all_latencies: list[float] = []
    transcript: list[dict] = []

    for run in range(1, args.repeats + 1):
        if args.repeats > 1:
            print(f"── Εκτέλεση {run}/{args.repeats}")
        for probe in PROBE_PROMPTS:
            reply, intent, secs = ask(probe)
            all_replies.append(reply)
            all_latencies.append(secs)
            ctx = fetch_context(probe) if intent in GROUNDABLE_INTENTS else ""
            transcript.append({
                "run": run, "probe": probe, "reply": reply,
                "intent": intent, "seconds": round(secs, 2),
                "context_chars": len(ctx), "groundable": bool(ctx),
                "_context": ctx,
            })
            print(f"  [{secs:4.1f}s] [{intent:9}] {probe}")
            print(f"            → {reply or '(κενή απάντηση)'}")

    non_empty = [r for r in all_replies if r.strip()]
    if not non_empty:
        print("\n❌ Καμία απάντηση. Τρέξε πρώτα ./scripts/smoke_test.sh")
        sys.exit(1)

    reference = load_reference_style()

    # Στυλ και αξιοπιστία μετρώνται σε ΟΛΕΣ τις απαντήσεις.
    report = aggregate_report(
        all_replies, reference_style=reference, latencies_s=all_latencies
    )

    # Η τεκμηρίωση μετριέται ΜΟΝΟ όπου το RAG έδωσε context. Σε casual
    # κουβέντα δεν υπάρχει τίποτα να τεκμηριωθεί, και η συμπερίληψή της
    # θα παρήγαγε σταθερό 1.0 που δεν λέει τίποτα για το σύστημα.
    grounded = [t for t in transcript if t["groundable"] and t["reply"].strip()]
    if grounded:
        sub = aggregate_report(
            [t["reply"] for t in grounded],
            contexts=[t["_context"] for t in grounded],
        )
        report["accuracy"] = sub["accuracy"]
        report["accuracy"]["n_groundable"] = len(grounded)
        report["accuracy"]["n_total"] = len(all_replies)
    else:
        report["accuracy"] = {
            "note": "καμία απάντηση με ανακτημένο context — δεν μετρήθηκε",
            "n_groundable": 0,
            "n_total": len(all_replies),
        }

    for t in transcript:
        t.pop("_context", None)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "probes": list(PROBE_PROMPTS),
        "repeats": args.repeats,
        "baseline": BASELINE,
        "report": report,
        "transcript": transcript,
    }
    (out_dir / "evaluation_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    md = [
        "# Αποτελέσματα Αξιολόγησης",
        "",
        f"Παραγωγή: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · "
        f"{len(PROBE_PROMPTS)} probes × {args.repeats} · Krikri-8B Q4_K_M τοπικά (Metal)",
        "",
        "Οι μετρήσεις αφορούν **ολόκληρο το σύστημα** — webhook, ταξινόμηση",
        "πρόθεσης, δρομολόγηση, παραγωγή, guardrails — όχι το μοντέλο μεμονωμένα.",
        "",
        "## Πριν και μετά",
        "",
        f"Ως «πριν» χρησιμοποιείται: {BASELINE['label']}.",
        "",
        comparison_table(report),
        "",
        "## Πλήρεις μετρικές",
        "",
        report_markdown(report),
        "",
        "## Δείγμα απαντήσεων",
        "",
        "| Ερώτηση | Απάντηση | Intent | s |",
        "|---|---|---|---:|",
    ]
    for t in transcript[:len(PROBE_PROMPTS)]:
        reply = (t["reply"] or "—").replace("|", "\\|").replace("\n", " ")
        md.append(f"| {t['probe']} | {reply} | {t['intent']} | {t['seconds']} |")

    md += [
        "",
        "## Περιορισμοί",
        "",
        "Οι μετρικές είναι **λεξικοί και δομικοί προσεγγιστές**, όχι σημασιολογική",
        "κρίση. Το `style_distance` απαντά «γράφει σαν τον ίδιο άνθρωπο;», όχι",
        "«λέει το σωστό». Το `grounding_score` εντοπίζει ισχυρισμούς που δεν",
        "στηρίζονται στο δοθέν context, αλλά δεν αναγνωρίζει μια ρέουσα πρόταση",
        "που τυχαίνει να είναι λάθος ενώ υπάρχει στο context.",
        "",
        "Η αξιολόγηση με NLI για faithfulness και μια μελέτη ανθρώπινης προτίμησης",
        "είναι η επικύρωση που αυτοί οι προσεγγιστές αντικαθιστούν, και δηλώνονται",
        "ως μελλοντική εργασία (§9.3).",
    ]

    (out_dir / "evaluation_results.md").write_text("\n".join(md), encoding="utf-8")

    print("\n" + "=" * 58)
    print(comparison_table(report))
    print("=" * 58)
    print(f"\n→ {out_dir/'evaluation_results.md'}")
    print(f"→ {out_dir/'evaluation_results.json'}")
    if reference is None:
        print("\n⚠ Δεν βρέθηκαν golden examples — λείπει η στήλη style_distance.")


if __name__ == "__main__":
    main()
