#!/usr/bin/env python3
"""Adjudicate queued cards: Opus writes the final annotation from three independent runs.

Input is an adjudication queue (one JSON line per queued mention, grouped here by
card) plus the three runs' final annotations. Opus sees the card, all three full
annotations labelled neutrally, the queued mentions with why each was queued, and
the formal resolver's binding where one exists. It does not see which model wrote
which annotation, or any analyst's provisional verdict.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import random
import subprocess
import sys

import run_pipeline as rp
from build_agents import spec_sha
from preprocess import format_preprocessed_card, iter_scryfall_cards  # noqa: E402

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent

INSTRUCTIONS = """\
--- Adjudication ---
{n} annotators worked on this card independently under the same specification. Their final annotations follow, numbered. Some of their mentions were queued for review, listed below with the reason each was queued. For some mentions a separate formal resolver's binding is also given.

Write the final annotation for the whole card, every mention, under the specification:
- Decide each mention by meaning. Agreement between annotators is evidence, not proof, and the formal resolver can be wrong too.
- Include a mention any annotator found if it is a mention under the specification, and leave out any that is not. Handle every candidate (annotate or reject).
- Set confident: false only if, after careful analysis, two readings remain genuinely plausible. Name both in uncertainty_reason. Those cards go to a human, so do not use the flag as a formality.
- The card's official rulings follow the annotations. A ruling that says what a pronoun, "that <noun>" or another mention refers to settles that mention.{rules_cmd}
"""


def card_rulings(name: str) -> str:
    """The card's official rulings, from the mtg-rules skill's local data."""
    if not rp.MTG_RULES_DIR:
        return "(rulings unavailable)"
    proc = subprocess.run([str(rp.MTG_RULES_DIR / "scripts" / "rulings"), name],
                          capture_output=True, text=True, timeout=60, check=False)
    lines = [l for l in proc.stdout.splitlines() if l.strip() and not l.startswith(("note:", "# "))]
    return "\n".join(lines) if lines else "(no official rulings)"


def render_prompt(info: dict, annotations: list[dict], queued: list[dict]) -> str:
    rules_cmd = (f" If a resolution depends on a Comprehensive Rules section, look it up with "
                 f"`{rp.MTG_RULES_DIR}/scripts/rule <number>` (by that absolute path, with no cd or pipes)."
                 if rp.MTG_RULES_DIR else "")
    parts = [info["prompt_text"], "", INSTRUCTIONS.format(n=len(annotations), rules_cmd=rules_cmd)]
    for i, ann in enumerate(annotations, 1):
        slim = {"verdict": ann.get("verdict"), "uncertainty_reason": ann.get("uncertainty_reason"),
                "mentions": ann.get("mentions", []), "rejected": ann.get("rejected", [])}
        parts.append(f"--- Annotation {i} ---\n{json.dumps(slim, ensure_ascii=False, indent=1)}\n")
    parts.append(f"--- Official rulings ---\n{card_rulings(info['card_name'])}\n")
    parts.append("--- Queued mentions ---")
    for q in queued:
        line = f"- [{q['ability_id']}] {q['mention_text']!r} #{q['occurrence']}: queued for {', '.join(q['reasons'])}"
        if q.get("lean"):
            line += f"; formal resolver binds it to: {q['lean'].get('antecedent')!r}"
        parts.append(line)
    return "\n".join(parts) + "\n"


def load_runs(paths: list[pathlib.Path]) -> list[dict[str, dict]]:
    runs = []
    for p in paths:
        rows: dict[str, dict] = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["oracle_id"]] = r
        runs.append(rows)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--queue", required=True)
    ap.add_argument("--split", required=True, help="Queue split to adjudicate (canon or held)")
    ap.add_argument("--runs", nargs="+", required=True, help="Run JSONL files (two or more), in any order")
    ap.add_argument("--output", required=True)
    ap.add_argument("--scryfall", default="../deckmaste.rs/data/scryfall/oracle-cards.jsonl")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--effort", default="high")
    args = ap.parse_args()

    if rp.stale_agents():
        sys.exit("Stale agent files; run src/build_agents.py first.")
    queue: dict[str, list[dict]] = {}
    for line in pathlib.Path(args.queue).read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            if q["split"] == args.split:
                queue.setdefault(q["oracle_id"], []).append(q)
    runs = load_runs([ROOT_DIR / p for p in args.runs])
    out_path = ROOT_DIR / args.output
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("status") == "ok":
                    done.add(r["oracle_id"])
    infos = {c["oracle_id"]: format_preprocessed_card(c)
             for c in iter_scryfall_cards((ROOT_DIR / args.scryfall).resolve(), vintage_only=True)
             if c["oracle_id"] in queue}
    todo = [oid for oid in queue if oid not in done]
    print(f"{len(queue)} queued cards, {len(done)} already adjudicated, {len(todo)} to go", flush=True)

    def work(oid: str) -> dict:
        info = infos[oid]
        anns = [r[oid].get("annotation") or {} for r in runs]
        random.Random(oid).shuffle(anns)  # no fixed position per model
        prompt = render_prompt(info, anns, queue[oid])
        res = rp.run_agent_on_card(info, rp.TIER2, args.effort, prompt_override=prompt)
        return {"card_name": info["card_name"], "oracle_id": oid, "abilities": info["abilities"],
                "costs": info["costs"], "spec_sha": spec_sha(), "queued": queue[oid], **res}

    with open(out_path, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(args.concurrency) as pool:
        for fut in cf.as_completed([pool.submit(work, oid) for oid in todo]):
            row = fut.result()
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            ann = row.get("annotation") or {}
            cost = (row.get("meta") or {}).get("total_cost_usd")
            print(f"  [{row['status']}] {row['card_name']} | verdict={ann.get('verdict')} | "
                  f"${cost or 0:.3f} | {row.get('grounding_errors') or ''}", flush=True)


if __name__ == "__main__":
    main()
