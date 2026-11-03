#!/usr/bin/env python3
"""Pretty-print the current endophora annotation for each card in an output JSONL log.

The log is append-only; the last row per oracle_id is the current result.
Pass --all to print every row instead.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any


def print_annotation(ann: dict[str, Any], indent: str = "  ") -> None:
    if ann.get("uncertainty_reason"):
        print(f"{indent}! uncertainty_reason: {ann['uncertainty_reason']}")
    mentions = ann.get("mentions", [])
    if not mentions:
        print(f"{indent}(no endophoric mentions)")
    for m in mentions:
        ant = m["antecedent"]
        sec = f" (via/whole: {ant['secondary_text']!r})" if ant.get("secondary_text") else ""
        conf = "" if m.get("confident", True) else " [UNCERTAIN]"
        print(
            f"{indent}- [{m['ability']}] {m['mention_text']!r} (#{m['occurrence']}, {m['category']}) "
            f"in {m['context_snippet']!r}\n"
            f"{indent}    -> {ant['kind']} ({ant['referent_type']}) @ {ant['ability']}: {ant['text']!r}{sec}{conf}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", default="output/spike.jsonl")
    parser.add_argument("--all", action="store_true", help="Print every row, not just the current one per card")
    args = parser.parse_args()

    with open(args.path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not args.all:
        latest: dict[str, dict[str, Any]] = {}
        for r in rows:
            latest.pop(r["oracle_id"], None)  # keep last-seen order
            latest[r["oracle_id"]] = r
        rows = list(latest.values())

    for r in rows:
        status = r.get("status")
        ann = r.get("annotation") or {}
        tier = r.get("tier") or (r.get("tier2") or {}).get("tier")
        print(f"=== {r['card_name']} (status={status}, verdict={ann.get('verdict')}, tier={tier}) ===")
        for aid, text in {**r.get("costs", {}), **r.get("abilities", {})}.items():
            print(f"  [{aid}] {text}")
        for key in ("grounding_errors", "grounding_warnings", "schema_errors"):
            for msg in r.get(key) or []:
                print(f"  ! {key[:-1].replace('_', ' ')}: {msg}")
        if status == "escalation_error":
            t2 = r.get("tier2", {})
            print(f"  ! Tier-2 failed ({t2.get('status')}): "
                  f"{t2.get('error') or t2.get('grounding_errors') or t2.get('stderr')}")
        if ann:
            print_annotation(ann)
        t1 = r.get("tier1")
        if t1 and t1.get("annotation"):
            print("  Tier-1 had:")
            print_annotation(t1["annotation"], indent="    ")
        print()

    statuses = Counter(r.get("status") for r in rows)
    verdicts = Counter((r.get("annotation") or {}).get("verdict") for r in rows if r.get("status") == "ok")
    escalated = sum(1 for r in rows if r.get("tier1"))
    print(f"{len(rows)} cards | status {dict(statuses)} | verdict {dict(verdicts)} | escalated {escalated}")


if __name__ == "__main__":
    main()
