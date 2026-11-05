#!/usr/bin/env python3
"""Preprocess Scryfall oracle-cards JSONL into ability-indexed text for DRT endophora annotation."""

from __future__ import annotations

import gzip
import json
import pathlib
import re
from typing import Any, Iterator

# Scryfall "Oracle Cards" bulk data, as JSON Lines (.jsonl or .jsonl.gz) or Scryfall's JSON array.
DEFAULT_SCRYFALL = "data/oracle-cards.jsonl.gz"

REMINDER_RE = re.compile(r"\s*\([^()]*\)")  # Oracle parentheses are always reminder text


def strip_reminder_text(text: str) -> str:
    """Remove parenthesized reminder text from Oracle text and normalize whitespace."""
    prev = None
    curr = text
    while curr != prev:
        prev = curr
        curr = REMINDER_RE.sub("", curr)
    lines: list[str] = []
    for raw_line in curr.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        line = re.sub(r"\s+([,.;:])", r"\1", line)
        if line:
            lines.append(line)
    return "\n".join(lines)


MODE_LINE_RE = re.compile(r"^(?:•|\+ \{|\{P\})")  # modal bullet, spree "+ {cost} —", pawprint "{P} —"
# Die-roll result rows: "1—9 | ", "20 | ", "15+ | ", "3 or less | ", and six-sided "4 — " / "1 or 2 — ".
ROW_RE = re.compile(r"^(?:\d+(?:[—–-]\d+|\+| or less)? \||\d+(?: or \d+)? — )")
STATION_ROW_RE = re.compile(r"^\d+\+ \| ")
LEVEL_HEADER_RE = re.compile(r"^LEVEL (\d+(?:[-–]\d+|\+))$")
PT_LINE_RE = re.compile(r"^\*?\d*\*?/\*?\d*\*?$")
ANCHOR_CHOICE_RE = re.compile(r"[Cc]hoose ([A-Z][\w']*) or ([A-Z][\w']*)\.$")


def group_ability_lines(lines: list[str]) -> list[str]:
    """Group Oracle text lines into abilities.

    One line is one ability, except:
    - modal bullets, spree and pawprint modes, and die-roll result rows join the
      ability above them (six-sided "N — " rows only directly under a roll);
    - anchor-word bullets ("choose Khans or Dragons. • Khans — ...") are each
      their own ability, since they are conditional grants, not modes;
    - level-up bands become a prefix ("LEVEL 2-6 (3/3) | ") on each ability in
      the band, and station bands ("9+ | ") carry onto the band's later lines.
    """
    abilities: list[str] = []
    level: str | None = None
    level_pt: str | None = None
    station: str | None = None
    has_station = any(l.startswith("Station") for l in lines)
    anchors: tuple[str, str] | None = None
    for line in lines:
        is_bullet = line.startswith("•")
        anchor_bullet = bool(anchors) and is_bullet and any(line.startswith(f"• {w} — ") for w in anchors)
        if not is_bullet:
            m_anchor = ANCHOR_CHOICE_RE.search(line)
            anchors = (m_anchor.group(1), m_anchor.group(2)) if m_anchor else None
        is_row = bool(ROW_RE.match(line))
        if is_row and has_station and STATION_ROW_RE.match(line):
            is_row = False
        if is_row and re.match(r"^\d+(?: or \d+)? — ", line) and not (
                abilities and re.search(r"\broll\b", abilities[-1], re.IGNORECASE)):
            is_row = False
        if abilities and not anchor_bullet and (MODE_LINE_RE.match(line) or is_row):
            abilities[-1] = f"{abilities[-1]} {line}"
            continue
        m = LEVEL_HEADER_RE.match(line)
        if m:
            level, level_pt = m.group(1), None
            continue
        if level is not None and level_pt is None and PT_LINE_RE.match(line):
            level_pt = line
            continue
        if level is not None:
            pt = f" ({level_pt})" if level_pt else ""
            line = f"LEVEL {level}{pt} | {line}"
        elif has_station:
            m_st = STATION_ROW_RE.match(line)
            if m_st:
                station = m_st.group(0)
            elif station and not line.startswith("Station"):
                line = f"{station}{line}"
        abilities.append(line)
    return abilities


PRONOUNS = {
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "it's", "they're",
    "he", "him", "his", "she", "her", "yourself",
}
_CANDIDATE_PATTERNS = [
    r"it|its|itself|they|them|their|theirs|themselves|they're|it's|he|him|his|she|her|yourself",
    # "that player or permanent" stays one candidate; "this creature or another" does not.
    r"(?:that|those|this|these)\s+[a-z-]+"
    r"(?:\s+or\s+(?:that\s+)?(?:permanent|planeswalker|player|spell|ability|creature|card|battle|token)\b)?",
    r"the (?:last )?chosen [a-z]+",
    r"another|other(?!\s+than\b)",
    r"that much|that many",
    r"X|Y|Z",
    r"if you do|if you don't|if you can't|when you do|if they do|if they don't|if they can't|when they do"
    r"|otherwise|if it doesn't",
    r"this way|the same|the rest"
    r"|the other(?: (?:creatures?|cards?|permanents?|players?|spells?|piles?|ones?|opponents?)\b)?"
    r"|each other(?=\s*[.,;]|$)|one another",
    r"which|who|whose|whom",
    r"the (?:exiled|sacrificed|revealed|discarded|tapped|returned|milled|copied|destroyed|noted|cycled|discovered"
    r"|drafted|exploited|targeted|attacking|blocking|blocked|modified|blighted) [a-z]+",
    r"the (?:result|first result|second result|highest result|original spell|new targets?)",
    r"the copy|the copies|the token|the tokens|the vote|the difference|the total|the piles?|the card|the cards"
    r"|the creature|the creatures|the spell|the player|the damage",
    r"one(?!\s+(?:or|of|more|target|mana|other|color|creature|plus|spell|basic|at random|that hasn't)\b|\s*[—,])"
    r"(?<!choose one)(?<!plus one)",
    r"again|noted",
    r"chosen(?= this way| at random)",
]
# Listed separately so they can overlap the main candidates: "If the player
# does" is a lookback that also contains the readback "the player", and
# "exiled with <name>" marks a participial readback whose head noun varies.
_EXTRA_PATTERNS = [
    r"if (?:a|the|that) player (?:does|doesn't|can't)",
    r"(?:exiled|put onto the battlefield|returned to the battlefield|turned face up|sacrificed) with",
]
EXTRA_CANDIDATE_RE = re.compile(r"(?<!\w)(?:" + "|".join(_EXTRA_PATTERNS) + r")(?!\w)", re.IGNORECASE)
CANDIDATE_RE = re.compile(r"(?<![\w{])(?:" + "|".join(_CANDIDATE_PATTERNS) + r")(?![\w}])", re.IGNORECASE)
# Never mentions (spec: "What is not a mention"); dropped from the candidate list.
CANDIDATE_SKIP_RE = re.compile(
    r"^(?:this (?:creature|artifact|enchantment|spell|card|aura|equipment|vehicle|permanent|land|ability|turn|one"
    r"|phase|step|planeswalker|saga|battle|siege|class|room|token|emblem|door|spacecraft|case|combat|game|main"
    r"|draft|booster|is)"
    r"|that (?:is|are|was|were|would|could|can|has|have|had|you|a|an|the|it|its|enters?|entered|deals|dealt"
    r"|shares?|attacks?|attacked|blocks?|blocked|died|targets?|each|crewed|isn|aren|wasn|weren|hasn|haven|doesn"
    r"|don|didn|wouldn)"
    r"|these|this)$",
    re.IGNORECASE,
)
# A leading flavor/ability-word label ("Bring it Down! — ", "Landfall — ").
MODE_LABEL_RE = re.compile(r"• ([^—,.:{\"•]{1,40}?) — ")
LABEL_RE = re.compile(r"^(?:LEVEL [^|]*\| |\d+\+ \| )?([^—,.:{\"]{1,40}?) — ")


def _word_count(haystack: str, needle: str) -> int:
    return len(re.findall(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack, flags=re.IGNORECASE))


def self_names(card: dict[str, Any]) -> list[str]:
    """The card's name, its face names, and their comma-short forms ("Zhang He")."""
    names = [card["name"]] + [f.get("name", "") for f in card.get("card_faces") or []]
    names += [n.split(",")[0] for n in names if "," in n]
    names += [p for n in list(names) for p in n.split(" // ") if " // " in n]
    return sorted({n for n in names if n}, key=len, reverse=True)


def _mask(text: str, names: list[str]) -> str:
    """Blank out the card's own names and a leading label, keeping positions."""
    for n in names:
        text = re.sub(r"(?<!\w)" + re.escape(n) + r"(?!\w)", lambda m: "#" * len(m.group(0)), text)
    m = LABEL_RE.match(text)
    if m and len(m.group(1).split()) <= 5:
        text = text[: m.start(1)] + "#" * (m.end(1) - m.start(1)) + text[m.end(1):]
    # Mode labels inside a modal ability ("• Charge Them — ") are flavor words too.
    for m in MODE_LABEL_RE.finditer(text):
        if len(m.group(1).split()) <= 5:
            text = text[: m.start(1)] + "#" * (m.end(1) - m.start(1)) + text[m.end(1):]
    return text


def extract_candidates(text: str, names: list[str] | tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """Words and phrases in one ability that might be mentions, with occurrence numbers.

    Mechanical and deliberately over-inclusive: the annotator must annotate or
    reject each one. Matching runs on a copy with the card's own names and any
    leading label blanked out, but `occurrence` counts on the real text, the
    way the validator does, so a candidate's id matches a mention's.
    """
    masked = _mask(text, list(names))
    out = []
    matches = list(CANDIDATE_RE.finditer(masked)) + list(EXTRA_CANDIDATE_RE.finditer(masked))
    for m in sorted(matches, key=lambda m: (m.start(), -m.end())):
        phrase = text[m.start():m.end()]
        if CANDIDATE_SKIP_RE.match(phrase):
            continue
        occ = _word_count(text[: m.end()], phrase)
        out.append({"text": phrase, "occurrence": occ, "start": m.start(), "end": m.end(),
                    "pronoun": phrase.lower() in PRONOUNS})
    return out


def quoted_abilities(text: str) -> list[str]:
    """Quoted granted abilities inside an ability's text (not quoted names like "legend rule")."""
    return [q for q in re.findall(r'"([^"]+)"', text)
            if re.search(r"[:.]", q) or re.match(r"(?:When|Whenever|At|As long as|If)\b", q)]


def _ability_lines(aid: str, text: str, names: list[str]) -> list[str]:
    lines = [f"[{aid}] {text}"]
    for q in quoted_abilities(text):
        lines.append(f'    quoted ability: "{q}"')
    cands = extract_candidates(text, names)
    if cands:
        lines.append("    candidates: " + ", ".join(f"{c['text']}#{c['occurrence']}" for c in cands))
    return lines


def is_vintage_playable(card: dict[str, Any]) -> bool:
    """Return True if the card is legal or restricted in Vintage."""
    legalities = card.get("legalities") or {}
    return legalities.get("vintage") in ("legal", "restricted")


def format_preprocessed_card(card: dict[str, Any]) -> dict[str, Any]:
    """Produce structured ability list and formatted prompt text for a Scryfall card."""
    name = card["name"]
    faces = card.get("card_faces")
    costs: dict[str, str] = {}
    names = self_names(card)

    if faces and "oracle_text" not in card:
        face_blocks: list[str] = [f"Card: {name}"]
        abilities_map: dict[str, str] = {}
        for f_idx, face in enumerate(faces):
            f_name = face.get("name", f"Face {f_idx}")
            mana_cost = face.get("mana_cost") or "(none)"
            type_line = face.get("type_line") or "(none)"
            cleaned = strip_reminder_text(face.get("oracle_text") or "")
            ab_lines = group_ability_lines(cleaned.splitlines())
            costs[f"F{f_idx}.cost"] = face.get("mana_cost") or ""
            face_blocks.append(f"\nFace {f_idx}: {f_name}")
            face_blocks.append(f"[F{f_idx}.cost] Mana Cost: {mana_cost}")
            face_blocks.append(f"Type Line: {type_line}")
            face_blocks.append("Abilities:")
            if not ab_lines:
                face_blocks.append("  (none)")
            else:
                for a_idx, ab in enumerate(ab_lines):
                    aid = f"F{f_idx}.A{a_idx}"
                    abilities_map[aid] = ab
                    face_blocks.extend(_ability_lines(aid, ab, names))
        prompt_text = "\n".join(face_blocks)
    else:
        mana_cost = card.get("mana_cost") or "(none)"
        type_line = card.get("type_line") or "(none)"
        cleaned = strip_reminder_text(card.get("oracle_text") or "")
        ab_lines = group_ability_lines(cleaned.splitlines())
        abilities_map = {}
        costs["cost"] = card.get("mana_cost") or ""
        blocks = [
            f"Card: {name}",
            f"[cost] Mana Cost: {mana_cost}",
            f"Type Line: {type_line}",
            "Abilities:",
        ]
        if not ab_lines:
            blocks.append("  (none)")
        else:
            for a_idx, ab in enumerate(ab_lines):
                aid = f"A{a_idx}"
                abilities_map[aid] = ab
                blocks.extend(_ability_lines(aid, ab, names))
        prompt_text = "\n".join(blocks)

    return {
        "oracle_id": card.get("oracle_id"),
        "scryfall_id": card.get("id"),
        "card_name": name,
        "mana_cost": card.get("mana_cost"),
        "type_line": card.get("type_line"),
        "abilities": abilities_map,
        "costs": costs,
        "candidates": {aid: extract_candidates(t, names) for aid, t in abilities_map.items()},
        "prompt_text": prompt_text,
    }


def iter_scryfall_cards(
    path: str | pathlib.Path,
    vintage_only: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield Scryfall card dicts from a JSON Lines file or a JSON array, optionally gzipped."""
    p = pathlib.Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        cards = json.load(f) if first == "[" else (json.loads(l) for l in f if l.strip())
        for card in cards:
            if vintage_only and not is_vintage_playable(card):
                continue
            yield card


def load_card_names(path: str | pathlib.Path) -> list[str]:
    """Card names from a text file, one per line; blank lines and #-comments ignored.

    A meld pair written as one line ("A // B") also selects each half, since
    Scryfall lists meld halves as separate cards.
    """
    names: list[str] = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(line)
            names.extend(p for p in line.split(" // ") if p != line)
    return names


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preview preprocessed MTG cards.")
    parser.add_argument(
        "--scryfall",
        default=DEFAULT_SCRYFALL,
        help="Path to Scryfall oracle-cards .jsonl or .jsonl.gz",
    )
    parser.add_argument(
        "--card",
        action="append",
        help="Specific card name(s) to preview",
    )
    args = parser.parse_args()

    wanted = set(args.card or ["Ad Nauseam", "Green Sun's Twilight", "Stuffy Doll"])
    for c in iter_scryfall_cards(args.scryfall, vintage_only=True):
        if c["name"] in wanted:
            info = format_preprocessed_card(c)
            print(info["prompt_text"])
            print("-" * 60)
