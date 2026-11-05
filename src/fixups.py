#!/usr/bin/env python3
"""Apply mechanical corrections to the release, after locate.py.

Rules (each checked against the card text, so a rebuild can rerun them):
- player_type: an antecedent that is a plain player phrase ("a player",
  "Target opponent", "each player", ...) gets referent_type "player".
- verb_crop: an object antecedent that starts with the verb that introduced
  it ("Create a 1/1 ... token", "discard a card", "Choose one") is cropped to
  the noun phrase. "Copy ..." spans are left alone: the spec makes the copying
  clause the antecedent of a copy introduced by a verb.
- TYPE_CORRECTIONS and CARD_FIXES: individual corrections no safe rule covers.

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


# Individual corrections: (card, ability, mention text, mention occurrence) ->
# fields to set on the mention ("mention") and its antecedent ("antecedent").
# A changed antecedent text gets its occurrence recomputed from where it sat
# inside the old span, or must be unique in its ability.
CARD_FIXES = [
    # The antecedent span contained the mention itself.
    ("Paradox Surveyor", "A1", "its", 1, {"antecedent": {"text": "a card"}}),
    ("Corrosion", "A1", "it", 1, {"antecedent": {"text": "each artifact"}}),
    # "the other creature" after "enchanted creature deals damage to a creature": as on Venom.
    ("Venomous Fangs", "A1", "the other", 1,
     {"antecedent": {"text": "enchanted creature", "secondary_text": "a creature"}}),
    ("Charisma", "A1", "the other", 1,
     {"antecedent": {"text": "enchanted creature", "secondary_text": "a creature"}}),
    # "You and that opponent each gain control of all creatures the other controls": distributive.
    ("Reins of Power", "A0", "the other", 1, {"antecedent": {"kind": "distributive_member"}}),
    # Participial readback missing its action clause.
    ("Lobelia Sackville-Baggins", "A2", "the exiled card", 1, {"antecedent": {
        "secondary_text": "exile target creature card from an opponent's graveyard that was put there from the "
                          "battlefield this turn"}}),
    # The mention is the whole readback noun phrase.
    ("Frantic Scapegoat", "A2", "the other", 1, {"mention": {
        "mention_text": "the other creatures", "occurrence": 1,
        "context_snippet": "you may suspect one of [the other creatures]"}}),
    # Inside a quoted ability, the object that has it is the outer phrase (as for pronouns).
    ("Hardy Outlander", "A0", "another", 1, {"antecedent": {
        "text": "Commander creatures you own", "secondary_text": "this creature"}}),
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
    by_card = {r["card_name"]: r for r in rows}
    for card, aid, mtext, mocc, change in CARD_FIXES:
        row = by_card[card]
        found = [m for m in row["mentions"] if m["ability"] == aid
                 and m["mention_text"].lower() == mtext.lower() and m["occurrence"] == mocc]
        if not found:
            if any(m["ability"] == aid and m["mention_text"] == change.get("mention", {}).get("mention_text")
                   for m in row["mentions"]):
                continue  # already applied
            sys.exit(f"{card}: no mention {mtext!r}#{mocc} in {aid}")
        m, a = found[0], found[0]["antecedent"]
        before = {part: {k: (m if part == "mention" else a).get(k) for k in fields}
                  for part, fields in change.items()}
        if before == change:
            continue  # already applied
        old_text, old_occ = a["text"], (a["occurrences"] or [None])[0]
        m.update(change.get("mention", {}))
        a.update(change.get("antecedent", {}))
        if a["text"] != old_text:
            src = {**row["abilities"], **row["costs"]}[a["ability"]]
            if old_text and a["text"] in old_text:
                a["occurrences"] = [relocate(src, old_text, old_occ, a["text"])]
            elif len(re.findall(re.escape(a["text"]), src)) == 1:
                a["occurrences"] = [1]
            else:
                sys.exit(f"{card}: can't place {a['text']!r} in {a['ability']}")
        log.append({"card": card, "mention": mtext, "rule": "card_fix", "from": before,
                    "to": {**change, "occurrences": a["occurrences"]}})
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
    manifest["fixups"] = manifest.get("fixups", []) + log  # earlier runs' entries are kept
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
