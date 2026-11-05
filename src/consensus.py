#!/usr/bin/env python3
"""Compare independent annotation runs and write an adjudication queue of their disagreements.

A card is accepted as-is only if every run annotated the same mentions with the
same referent, confidently, with no grounding failures. "Same referent" is a
mechanical check: identical antecedent kind, and antecedent texts where one
contains the other (a longer or shorter span of the same phrase). Anything else
is queued, one line per mention, in the format src/adjudicate.py reads.
"""

from __future__ import annotations

import argparse
import json
import re
import pathlib
from collections import Counter

from preprocess import self_names

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent


def load(path: pathlib.Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["oracle_id"]] = r
    return rows


def _norm(t: str | None) -> str:
    return " ".join((t or "").lower().replace('"', "").split()).rstrip(".")


SELF_RE = re.compile(r"^this (?:creature|artifact|enchantment|spell|card|aura|equipment|vehicle|permanent|land"
                     r"|saga|class|case|room|door|spacecraft|siege|battle|planeswalker|token)$")


def is_self(ant: dict, names: set[str]) -> bool:
    """The antecedent is the card itself: kind "source", or a self-reference span."""
    if ant.get("kind") == "source":
        return True
    t = _norm(ant.get("text"))
    return ant.get("kind") == "text_span" and (bool(SELF_RE.match(t)) or t in names)


def same_referent(a: dict, b: dict, names: set[str] = frozenset()) -> bool:
    x, y = a["antecedent"], b["antecedent"]
    if is_self(x, names) and is_self(y, names):
        return True
    if x.get("kind") != y.get("kind"):
        return False
    if x.get("kind") in ("source", "performer"):
        return True
    p, q = _norm(x.get("text")), _norm(y.get("text"))
    return bool(p and q) and (p in q or q in p)


def key(m: dict) -> tuple:
    return (m.get("ability"), (m.get("mention_text") or "").lower(), m.get("occurrence"))


def queue_lines(rows: list[dict], split: str = "vintage") -> list[dict]:
    """Queue lines for one card given each run's row; empty if every run agrees confidently."""
    oid = rows[0]["oracle_id"]
    names = {_norm(n) for n in self_names({"name": rows[0]["card_name"]})}
    lines: dict[tuple, dict] = {}

    def queue(k: tuple, reason: str) -> None:
        q = lines.setdefault(k, {"split": split, "card": rows[0]["card_name"], "oracle_id": oid,
                                 "ability_id": k[0], "mention_text": k[1], "occurrence": k[2],
                                 "reasons": []})
        if reason not in q["reasons"]:
            q["reasons"].append(reason)

    card_level = []
    for i, r in enumerate(rows):
        if r.get("status") != "ok" or not r.get("annotation"):
            card_level.append(f"run{i + 1}_status:{r.get('status')}")
        elif r["annotation"].get("verdict") == "uncertain":
            card_level.append(f"run{i + 1}_uncertain")
    mentions = [{key(m): m for m in (r.get("annotation") or {}).get("mentions", [])} for r in rows]
    for k in set().union(*mentions):
        present = [m.get(k) for m in mentions]
        if any(p is None for p in present):
            queue(k, "missing:" + ",".join(f"run{i + 1}" for i, p in enumerate(present) if p is None))
            continue
        if not all(same_referent(present[0], p, names) for p in present[1:]):
            queue(k, "referent_differs")
        if any(not p.get("confident", True) for p in present):
            queue(k, "confident_false")
    if card_level:
        for k in set().union(*mentions) or {("A0", "(card)", 1)}:
            for reason in card_level:
                queue(k, reason)
    return list(lines.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="vintage", help="Label written to each queue line")
    ap.add_argument("--queue", required=True, help="Output adjudication queue JSONL")
    args = ap.parse_args()

    runs = [load(ROOT_DIR / p) for p in args.runs]
    common = set.intersection(*(set(r) for r in runs))
    missing_cards = set.union(*(set(r) for r in runs)) - common
    reasons_count: Counter = Counter()
    queued_cards = 0
    accepted = 0
    with open(ROOT_DIR / args.queue, "w", encoding="utf-8") as out:
        for oid in sorted(common):
            lines = queue_lines([r[oid] for r in runs], args.split)
            if lines:
                queued_cards += 1
                for q in lines:
                    reasons_count.update(q["reasons"])
                    out.write(json.dumps(q, ensure_ascii=False) + "\n")
            else:
                accepted += 1
    print(f"cards in all runs: {len(common)}; accepted: {accepted}; queued: {queued_cards}; "
          f"in only some runs: {len(missing_cards)}")
    print("queue reasons (mentions):", dict(reasons_count.most_common()))


if __name__ == "__main__":
    main()
