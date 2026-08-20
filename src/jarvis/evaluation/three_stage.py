"""Three-stage evaluation: baseline → Persona-Chat → Viber (thesis §2.4, item 5).

The sequential training design makes a natural ablation: the SAME probe
prompts are answered by the model at each stage, so the contribution of
each stage to style fidelity is directly visible. Model access is injected
as callables, so this module has no heavy dependencies and is unit-testable;
the notebook wires in the real generate functions.

Planned additions (from the literature review):
  * NLI-based faithfulness (Synthetic-Persona-Chat, Jandaghi et al. 2024)
  * persona authentication classifier (PersonaGPT, Tang et al. 2021)
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

#: Fixed probes — bilingual, everyday topics a digital twin must handle.
#: Frozen list: changing probes between runs invalidates stage comparisons.
PROBE_PROMPTS: tuple[str, ...] = (
    "Τι κάνεις; Όλα καλά;",
    "Θα έρθεις τελικά το Σάββατο;",
    "Μπορείς να μου στείλεις την αναφορά μέχρι αύριο;",
    "Πώς σου φάνηκε η συνάντηση σήμερα;",
    "Έχεις κανένα νέο για το project;",
    "Τι λες να φάμε το βράδυ;",
    "Can you join the call at 3pm tomorrow?",
    "Ευχαριστώ πολύ για τη βοήθεια χθες!",
    "Πότε μπορούμε να τα πούμε από κοντά;",
    "Στείλε μου όταν φτάσεις σπίτι.",
)

GenerateFn = Callable[[str], str]


class ThreeStageEvaluator:
    """Collects per-stage generations for the fixed probe set."""

    def __init__(self, probes: Sequence[str] = PROBE_PROMPTS) -> None:
        self.probes = list(probes)
        self.stages: dict[str, list[str]] = {}

    def run_stage(self, stage_name: str, generate_fn: GenerateFn) -> list[str]:
        """Generate an answer for every probe; stores and returns them."""
        outputs = [generate_fn(probe) for probe in self.probes]
        self.stages[stage_name] = outputs
        return outputs

    def add_stage(self, stage_name: str, outputs: Sequence[str]) -> None:
        """Register pre-computed outputs (e.g. reloaded from a JSON file)."""
        if len(outputs) != len(self.probes):
            raise ValueError(f"{stage_name}: expected {len(self.probes)} outputs")
        self.stages[stage_name] = list(outputs)

    # -- reporting ----------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "probes": self.probes,
            "stages": self.stages,
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def to_markdown(self) -> str:
        """Side-by-side markdown table (paste-ready for the thesis)."""
        names = list(self.stages)
        header = "| # | Probe | " + " | ".join(names) + " |"
        sep = "|---" * (len(names) + 2) + "|"
        rows = [header, sep]
        for i, probe in enumerate(self.probes):
            cells = [self.stages[s][i].replace("\n", " ").replace("|", "\\|") for s in names]
            rows.append(f"| {i + 1} | {probe} | " + " | ".join(cells) + " |")
        return "\n".join(rows)

    @classmethod
    def load_json(cls, path: str | Path) -> ThreeStageEvaluator:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        ev = cls(probes=data["probes"])
        for name, outputs in data["stages"].items():
            ev.add_stage(name, outputs)
        return ev
