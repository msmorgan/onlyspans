#!/usr/bin/env python3
"""Add character offsets to the release (after locate.py and fixups.py).

Every offset is ability-local and half-open: [start, end] slices the text of
`abilities[ability]` (or of `costs[...]`), so no card-level serialization is
assumed.

- mention.span: the `occurrence`-th whole-word, case-insensitive match of
  `mention_text` in its ability (the validator's counting).
- antecedent.spans: one {"ability", "span"} per part, from `occurrences`;
  null for `source` and `performer`.
- antecedent.secondary_span: {"ability", "span"} for `secondary_text`, or null.
  `secondary_text` has no position of its own, so it is placed by the spec's
  definition (the latest re-mention): the last occurrence that ends before the
  mention in reading order (ability order, then offset); if none precedes the
  mention, the first one after it.

The release is rewritten in place, and the manifest records how each
secondary span was placed. Every offset is checked against the text it slices.
"""

from __future__ import annotations

import argparse
import json
import lzma
import pathlib
import re
import sys
from collections import Counter

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent


def ability_order(aid: str) -> tuple:
    return tuple(int(n) if n.isdigit() else n for n in re.findall(r"\d+|[A-Za-z]+", aid))


def mention_span(text: str, mention: dict) -> list[int]:
    hits = list(re.finditer(r"(?<!\w)" + re.escape(mention["mention_text"]) + r"(?!\w)", text, flags=re.IGNORECASE))
    m = hits[mention["occurrence"] - 1]
    return [m.start(), m.end()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--release", default="release/onlyspans.jsonl.xz")
    args = ap.parse_args()

    release = ROOT_DIR / args.release
    rows = [json.loads(line) for line in lzma.open(release, "rt", encoding="utf-8")]
    placed: Counter = Counter()
    for row in rows:
        src = {**row["costs"], **row["abilities"]}
        order = sorted(src, key=lambda k: (k not in row["costs"], ability_order(k)))
        for m in row["mentions"]:
            text = src[m["ability"]]
            m["span"] = mention_span(text, m)
            if text[m["span"][0]:m["span"][1]].lower() != m["mention_text"].lower():
                sys.exit(f"{row['card_name']}: mention span mismatch")
            a = m["antecedent"]
            if a["text"] is None:
                a["spans"] = None
            else:
                parts = a["text"].split(" + ") if a["kind"] == "split" else [a["text"]]
                aid = a["ability"]
                aids = aid.split(" + ") if a["kind"] == "split" and " + " in aid else [aid] * len(parts)
                a["spans"] = []
                for part, pid, occ in zip(parts, aids, a["occurrences"]):
                    hit = list(re.finditer(re.escape(part), src[pid]))[occ - 1]
                    a["spans"].append({"ability": pid, "span": [hit.start(), hit.end()]})
            sec = a.get("secondary_text")
            if not sec:
                a["secondary_span"] = None
                continue
            hits = [(k, h.start(), h.end()) for k in order for h in re.finditer(re.escape(sec), src[k])]
            if not hits:
                sys.exit(f"{row['card_name']}: secondary_text {sec!r} not on the card")
            here = (order.index(m["ability"]), m["span"][0])
            before = [h for h in hits if (order.index(h[0]), h[2]) <= here]
            if len(hits) == 1:
                k, s, e = hits[0]
                placed["unique"] += 1
            elif before:
                k, s, e = before[-1]
                placed["latest_before_mention"] += 1
            else:
                k, s, e = hits[0]
                placed["first_after_mention"] += 1
            a["secondary_span"] = {"ability": k, "span": [s, e]}

    with lzma.open(release, "wt", encoding="utf-8", preset=9 | lzma.PRESET_EXTREME) as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest_path = release.with_name(release.name.split(".")[0] + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["secondary_spans"] = dict(placed)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"spans added; secondary spans placed: {dict(placed)}")


if __name__ == "__main__":
    main()
