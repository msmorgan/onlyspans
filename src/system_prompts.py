"""System prompts for every model call, built from the one spec (prompts/endophora_spec.md).

- annotate:   the spec alone (Tier 1).
- escalate:   a short preamble for re-annotating one card that Tier 1 couldn't
              settle, then the spec (Tier 2, run_pipeline.py --escalate-uncertain).
- adjudicate: a short preamble for settling disagreements between independent
              annotations, then the spec (adjudicate.py).
"""

from __future__ import annotations

import hashlib
import pathlib

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT_DIR / "prompts" / "endophora_spec.md"

LOOKUPS = (
    "When a resolution depends on the Comprehensive Rules (e.g. CR 607 linked abilities, CR 608.2c/h resolving "
    "spells and abilities, CR 603 triggered abilities, CR 614 replacement effects), use the mtg-rules lookup "
    "scripts if the message names their directory. Run them by absolute path (`<dir>/scripts/rule 607.2a`, "
    "`<dir>/scripts/define <term>`), with no `cd` or pipes: only those scripts are permitted, and any other "
    "shell command is refused."
)

ESCALATE = f"""\
# Second Opinion

You receive a card that a first annotator could not settle: either it returned `verdict: "uncertain"`, or its output failed the automatic grounding checks. Its output and the reason for escalation (its `uncertainty_reason` and/or the list of grounding errors) follow the card text.

- Re-annotate the whole card from scratch under the specification below. Do not copy the first annotator's mentions without checking them.
- Fix every listed grounding error: spans must be verbatim substrings of the named ability, and occurrence indices must be correct.
- {LOOKUPS}
- If you can resolve the uncertainty, set `verdict: "ok"` and `uncertainty_reason: null`. If the text is genuinely ambiguous even under the rules, keep `verdict: "uncertain"` and explain the exact rule or linguistic defect.

The full specification follows.

---

"""

ADJUDICATE = f"""\
# Adjudication

You receive a card that several annotators labelled independently under the specification below, together with their annotations, the card's official rulings, and the mentions on which they disagreed or which they flagged.

- Write the final annotation for the whole card, deciding every mention by meaning. Agreement between annotators is evidence, not proof.
- {LOOKUPS}
- Set `confident: false` only if, after careful analysis, two readings remain genuinely plausible; those cards go to a human.

The full specification follows.

---

"""


def spec_text() -> str:
    return SPEC_PATH.read_text(encoding="utf-8")


def spec_sha() -> str:
    """Short hash of the spec, recorded on every output row."""
    return hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()[:12]


def system_prompt(kind: str) -> str:
    preamble = {"annotate": "", "escalate": ESCALATE, "adjudicate": ADJUDICATE}[kind]
    return preamble + spec_text()
