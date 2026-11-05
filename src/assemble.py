#!/usr/bin/env python3
"""Assemble the OnlySpans reference: one line per Vintage-legal card, every mention and its antecedent.

For each card, in order of precedence:
- declined (release/declined.jsonl): left out, and listed in the manifest;
- queued by the consensus check: the repaired adjudication if one exists,
  otherwise the adjudicated annotation;
- accepted by consensus: the first run's annotation (both runs agree on it);
- no candidate mentions: an empty annotation.

Every final annotation is re-validated against freshly preprocessed card text.
The script refuses to write the reference if any card fails or is missing.
"""

from __future__ import annotations

import argparse
import gzip
import lzma
import json
import pathlib
import sys
from collections import Counter

from system_prompts import spec_sha
from consensus import load, queue_lines
from preprocess import DEFAULT_SCRYFALL, format_preprocessed_card, iter_scryfall_cards
from run_pipeline import load_scryfall_snapshot
from validate import validate_grounding, validate_schema

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent

MENTION_FIELDS = ("ability", "mention_text", "occurrence", "context_snippet", "category", "antecedent", "confident")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True, help="Annotator runs; the first supplies accepted cards")
    ap.add_argument("--adjudicated", nargs="+", required=True,
                    help="Adjudication outputs, lowest precedence first (repairs last)")
    ap.add_argument("--declined", default="release/declined.jsonl")
    ap.add_argument("--scryfall", default=DEFAULT_SCRYFALL)
    ap.add_argument("--output", default="release/onlyspans.jsonl.xz")
    args = ap.parse_args()

    runs = [load(ROOT_DIR / p) for p in args.runs]
    adjudicated: dict[str, dict] = {}
    for p in args.adjudicated:
        for oid, r in load(ROOT_DIR / p).items():
            if r.get("status") == "ok":
                adjudicated[oid] = r
    declined = load(ROOT_DIR / args.declined) if (ROOT_DIR / args.declined).exists() else {}
    scryfall = (ROOT_DIR / args.scryfall).resolve()

    out_rows, problems = [], []
    sources: Counter = Counter()
    used_specs: Counter = Counter()
    tiers: Counter = Counter()
    seen = 0
    for card in iter_scryfall_cards(scryfall, vintage_only=True):
        seen += 1
        info = format_preprocessed_card(card)
        oid = info["oracle_id"]
        if oid in declined:
            sources["declined"] += 1
            continue
        rows = [r.get(oid) for r in runs]
        if any(r is None for r in rows):
            problems.append(f"{info['card_name']}: missing from a run")
            continue
        if any(r["abilities"] != info["abilities"] or r["costs"] != info["costs"] for r in rows):
            problems.append(f"{info['card_name']}: card text changed since annotation")
            continue
        if queue_lines(rows):
            row = adjudicated.get(oid)
            if row is None:
                problems.append(f"{info['card_name']}: queued but not adjudicated")
                continue
            source = "repaired" if row.get("repaired") else "adjudicated"
        else:
            row = rows[0]
            source = "no_candidates" if row.get("no_candidates") else "consensus"
        ann = row["annotation"]
        if ann.get("verdict") != "ok":
            problems.append(f"{info['card_name']}: final verdict {ann.get('verdict')}")
            continue
        errs = validate_schema(ann) or validate_grounding(ann, info)[0]
        if errs:
            problems.append(f"{info['card_name']}: {errs[:2]}")
            continue
        sources[source] += 1
        used_specs[row.get("spec_sha") or "unknown"] += 1
        tiers[row.get("tier") or "none"] += 1
        out_rows.append({
            "card_name": info["card_name"], "oracle_id": oid,
            "abilities": info["abilities"], "costs": info["costs"],
            "mentions": [{k: m[k] for k in MENTION_FIELDS} for m in ann["mentions"]],
            "source": source,
            "repaired": row.get("repaired"),
        })

    print(f"cards seen: {seen}; written: {len(out_rows)}; by source: {dict(sources)}")
    if problems:
        print(f"{len(problems)} problem(s); reference not written:", file=sys.stderr)
        for p in problems[:40]:
            print("  " + p, file=sys.stderr)
        sys.exit(1)

    out_path = ROOT_DIR / args.output
    if out_path.suffix == ".xz":
        f = lzma.open(out_path, "wt", encoding="utf-8", preset=9 | lzma.PRESET_EXTREME)
    else:
        f = gzip.open(out_path, "wt", encoding="utf-8")
    with f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "name": "OnlySpans reference",
        "cards": len(out_rows),
        "mentions": sum(len(r["mentions"]) for r in out_rows),
        "by_source": dict(sources),
        "declined": [{"card_name": r["card_name"], "oracle_id": oid, "reason": r.get("reason")}
                     for oid, r in declined.items()],
        "annotation_spec_shas": dict(used_specs),
        "spec_sha_at_assembly": spec_sha(),
        "scryfall_snapshot": load_scryfall_snapshot(scryfall),
        "annotators": sorted({r.get("tier") for run in runs for r in run.values()} - {None, "none"}),
        "final_annotation_by_tier": dict(tiers),
    }
    manifest_path = out_path.with_name(out_path.name.split(".")[0] + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path.relative_to(ROOT_DIR)} ({out_path.stat().st_size / 1e6:.1f} MB) "
          f"and {manifest_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
