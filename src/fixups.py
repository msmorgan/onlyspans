#!/usr/bin/env python3
"""Apply mechanical corrections to the release, after locate.py.

Rules (each checked against the card text, so a rebuild can rerun them):
- player_type: an antecedent that is a plain player phrase ("a player",
  "Target opponent", "each player", ...) gets referent_type "player".
- verb_crop: an object antecedent that starts with the verb that introduced
  it ("Create a 1/1 ... token", "discard a card", "Choose one") is cropped to
  the noun phrase. "Copy ..." spans are left alone: the spec makes the copying
  clause the antecedent of a copy introduced by a verb.
- TYPE_CORRECTIONS: individual referent_type fixes no safe rule covers.

The release is rewritten in place, and every change is listed in the manifest
under "fixups".
"""

from __future__ import annotations

import argparse
import json
import lzma
import pathlib
import re
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent

PLAYER_NP = re.compile(
    r"^(?:a|an|any|another|target|each|each other|that|the chosen|up to one target|one target)"
    r" (?:player|opponent)s?$", re.IGNORECASE)
VERB = re.compile(r"^(?:create|discard|choose) ", re.IGNORECASE)
# (card name, mention text, correct referent_type)
TYPE_CORRECTIONS = [
    ("Said // Done", "their", "object"),  # "their controller's next untap step": the tapped creatures
]


def relocate(src: str, old: str, old_occ: int, new: str) -> int:
    """Occurrence number of `new` at the position it had inside the old span."""
    start = [m.start() for m in re.finditer(re.escape(old), src)][old_occ - 1] + old.index(new)
    starts = [m.start() for m in re.finditer(re.escape(new), src)]
    return starts.index(start) + 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--release", default="release/onlyspans.jsonl.xz")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    release = ROOT_DIR / args.release
    rows = [json.loads(line) for line in lzma.open(release, "rt", encoding="utf-8")]
    corrections = {(c, t): v for c, t, v in TYPE_CORRECTIONS}
    log = []
    for row in rows:
        src_by_id = {**row["abilities"], **row["costs"]}
        for m in row["mentions"]:
            a = m["antecedent"]
            text = a["text"]
            if text is None:
                continue
            fix = corrections.get((row["card_name"], m["mention_text"]))
            if fix and a["referent_type"] != fix:
                log.append({"card": row["card_name"], "mention": m["mention_text"], "rule": "type_correction",
                            "from": a["referent_type"], "to": fix})
                a["referent_type"] = fix
            elif a["kind"] == "text_span" and PLAYER_NP.match(text) and a["referent_type"] != "player":
                log.append({"card": row["card_name"], "mention": m["mention_text"], "rule": "player_type",
                            "from": a["referent_type"], "to": "player"})
                a["referent_type"] = "player"
            if (a["kind"] == "text_span" and a["referent_type"] == "object" and VERB.match(text)
                    and m["category"] != "lookback_or_followup"):
                new = text.split(" ", 1)[1]
                src = src_by_id[a["ability"]]
                occ = relocate(src, text, a["occurrences"][0], new)
                log.append({"card": row["card_name"], "mention": m["mention_text"], "rule": "verb_crop",
                            "from": text, "to": new, "occurrences": [occ]})
                a["text"], a["occurrences"] = new, [occ]
    for entry in log:
        print(json.dumps(entry, ensure_ascii=False))
    print(f"{len(log)} change(s)", file=sys.stderr)
    if args.dry_run:
        return
    with lzma.open(release, "wt", encoding="utf-8", preset=9 | lzma.PRESET_EXTREME) as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest_path = release.with_name(release.name.split(".")[0] + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fixups"] = log
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
