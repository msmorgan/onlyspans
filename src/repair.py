#!/usr/bin/env python3
"""Repair adjudicated cards whose only fault is an antecedent's ability ID.

Opus sometimes cites the right antecedent span but the wrong ability, typically
for linked abilities ("cards exiled with this artifact" read back from another
ability). For each card with grounding errors, every antecedent span (or split
part) not found in its cited ability is looked up across the card. If each part
occurs in exactly one ability, the ID is corrected (a split whose parts are in
different abilities gets one ID per part, "A0 + A1"). The card is written as
repaired only if it then passes validation completely; anything else is left for
re-adjudication.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib

from preprocess import DEFAULT_SCRYFALL, format_preprocessed_card, iter_scryfall_cards
from validate import validate_grounding, validate_schema

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adjudicated", required=True, help="Adjudication output JSONL")
    ap.add_argument("--output", required=True, help="Repaired rows are appended here")
    ap.add_argument("--scryfall", default=DEFAULT_SCRYFALL)
    args = ap.parse_args()

    rows: dict[str, dict] = {}
    for line in (ROOT_DIR / args.adjudicated).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["oracle_id"]] = r
    bad = {oid: r for oid, r in rows.items() if r.get("status") == "grounding_error"}
    infos = {c["oracle_id"]: format_preprocessed_card(c)
             for c in iter_scryfall_cards((ROOT_DIR / args.scryfall).resolve()) if c["oracle_id"] in bad}

    repaired = []
    for oid, r in bad.items():
        info = infos[oid]
        ann = copy.deepcopy(r["annotation"])
        sources = {**info["abilities"], **info["costs"]}
        fixes = []
        for m in ann["mentions"]:
            a = m["antecedent"]
            if a.get("text") is None:
                continue
            parts = a["text"].split(" + ") if a.get("kind") == "split" else [a["text"]]
            if all(p in sources.get(a.get("ability"), "") for p in parts):
                continue
            homes = [[k for k, v in sources.items() if p in v] for p in parts]
            if all(len(h) == 1 for h in homes):
                ids = [h[0] for h in homes]
                new = ids[0] if len(set(ids)) == 1 else " + ".join(ids)
                fixes.append(f"{m['mention_text']!r}: antecedent ability {a.get('ability')} -> {new}")
                a["ability"] = new
        errors, warnings = validate_grounding(ann, info)
        if fixes and not errors and not validate_schema(ann):
            repaired.append({**r, "annotation": ann, "status": "ok", "grounding_errors": [],
                             "grounding_warnings": warnings, "repaired": fixes,
                             "original_grounding_errors": r["grounding_errors"]})
            print(f"repaired {r['card_name']}: {fixes}")
        else:
            print(f"left {r['card_name']}: {errors[:2]}")
    with open(ROOT_DIR / args.output, "a", encoding="utf-8") as f:
        for row in repaired:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(repaired)} of {len(bad)} repaired")


if __name__ == "__main__":
    main()
