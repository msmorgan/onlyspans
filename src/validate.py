"""Validate endophora annotations: schema shape, then grounding against the card text.

Schema checks are a backstop; `claude -p --json-schema` already constrains the
output. Grounding checks catch what a schema cannot: spans that are not verbatim
substrings of the card, unknown ability IDs, and occurrence indices past the
number of matches.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT_DIR / "schemas" / "card_endophora.schema.json"
SCHEMA: dict[str, Any] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

_MENTION = SCHEMA["properties"]["mentions"]["items"]
_ANTECEDENT = _MENTION["properties"]["antecedent"]
TOP_REQUIRED: list[str] = SCHEMA["required"]
MENTION_REQUIRED: list[str] = _MENTION["required"]
ANTECEDENT_REQUIRED: list[str] = _ANTECEDENT["required"]
VALID_VERDICTS = set(SCHEMA["properties"]["verdict"]["enum"])
VALID_CATEGORIES = set(_MENTION["properties"]["category"]["enum"])
VALID_KINDS = set(_ANTECEDENT["properties"]["kind"]["enum"])
VALID_REF_TYPES = set(_ANTECEDENT["properties"]["referent_type"]["enum"])

SPAN_REQUIRED_KINDS = {"text_span", "distributive_member", "split"}


def validate_schema(data: dict[str, Any]) -> list[str]:
    """Structural checks mirroring schemas/card_endophora.schema.json."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["annotation is not an object"]
    for req in TOP_REQUIRED:
        if req not in data:
            errors.append(f"missing top-level field '{req}'")
    if data.get("verdict") not in VALID_VERDICTS:
        errors.append(f"invalid verdict {data.get('verdict')!r}")
    mentions = data.get("mentions")
    if not isinstance(mentions, list):
        errors.append("'mentions' must be a list")
        return errors
    for idx, m in enumerate(mentions):
        if not isinstance(m, dict):
            errors.append(f"mention[{idx}] is not an object")
            continue
        for req in MENTION_REQUIRED:
            if req not in m:
                errors.append(f"mention[{idx}] missing '{req}'")
        if m.get("category") not in VALID_CATEGORIES:
            errors.append(f"mention[{idx}] invalid category {m.get('category')!r}")
        occ = m.get("occurrence")
        if not isinstance(occ, int) or isinstance(occ, bool) or occ < 1:
            errors.append(f"mention[{idx}] invalid occurrence {occ!r}")
        ant = m.get("antecedent")
        if not isinstance(ant, dict):
            errors.append(f"mention[{idx}].antecedent is not an object")
            continue
        for req in ANTECEDENT_REQUIRED:
            if req not in ant:
                errors.append(f"mention[{idx}].antecedent missing '{req}'")
        if ant.get("kind") not in VALID_KINDS:
            errors.append(f"mention[{idx}].antecedent invalid kind {ant.get('kind')!r}")
        if ant.get("referent_type") not in VALID_REF_TYPES:
            errors.append(f"mention[{idx}].antecedent invalid referent_type {ant.get('referent_type')!r}")
    return errors


def count_word_matches(haystack: str, needle: str) -> int:
    """Whole-word, case-insensitive occurrences of `needle` in `haystack`."""
    pattern = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
    return len(re.findall(pattern, haystack, flags=re.IGNORECASE))


def validate_grounding(data: dict[str, Any], card_info: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Check every span and index against the preprocessed card.

    Returns (errors, warnings). Errors mean the annotation cannot be trusted as
    written; warnings flag cosmetic drift (e.g. a lightly edited context snippet).
    """
    errors: list[str] = []
    warnings: list[str] = []
    abilities: dict[str, str] = card_info["abilities"]
    costs: dict[str, str] = card_info.get("costs", {})
    sources = {**abilities, **costs}
    all_text = list(sources.values())

    if data.get("card_name") != card_info["card_name"]:
        warnings.append(f"card_name {data.get('card_name')!r} != {card_info['card_name']!r}")

    for idx, m in enumerate(data.get("mentions") or []):
        tag = f"mention[{idx}] {m.get('mention_text')!r}"
        aid = m.get("ability")
        text = abilities.get(aid)
        if text is None:
            errors.append(f"{tag}: unknown ability {aid!r}")
        else:
            mt = m.get("mention_text") or ""
            n = count_word_matches(text, mt)
            occ = m.get("occurrence")
            if n == 0:
                errors.append(f"{tag}: not found in {aid}")
            elif isinstance(occ, int) and occ > n:
                errors.append(f"{tag}: occurrence {occ} but only {n} match(es) in {aid}")

            snippet = m.get("context_snippet") or ""
            bracketed = re.findall(r"\[([^\]]*)\]", snippet)
            if not any(b.lower() == mt.lower() for b in bracketed):
                warnings.append(f"{tag}: context_snippet does not bracket the mention")
            if snippet.replace("[", "").replace("]", "") not in text:
                warnings.append(f"{tag}: context_snippet is not verbatim from {aid}")

        ant = m.get("antecedent") or {}
        kind = ant.get("kind")
        a_aid, a_text = ant.get("ability"), ant.get("text")
        if kind in SPAN_REQUIRED_KINDS and (a_aid is None or a_text is None):
            errors.append(f"{tag}: antecedent kind {kind!r} needs ability and text")
        if a_text is not None:
            parts = a_text.split(" + ") if kind == "split" else [a_text]
            # A split whose parts are in different abilities lists one ability
            # per part, joined the same way ("A0 + A1").
            aids = (a_aid.split(" + ") if kind == "split" and a_aid and " + " in a_aid
                    else [a_aid] * len(parts))
            if len(aids) != len(parts):
                errors.append(f"{tag}: {len(parts)} antecedent parts but {len(aids)} abilities")
            for part, aid in zip(parts, aids):
                src = sources.get(aid) if aid is not None else None
                if src is None:
                    errors.append(f"{tag}: antecedent ability {aid!r} unknown")
                elif part not in src:
                    errors.append(f"{tag}: antecedent text {part!r} not verbatim in {aid}")
        # An antecedent can't contain its own mention ("a card with {X} in its mana cost" for "its").
        if a_text is not None and kind == "text_span" and a_aid == aid and text is not None:
            hits = list(re.finditer(r"(?<!\w)" + re.escape(m.get("mention_text") or "") + r"(?!\w)", text,
                                    flags=re.IGNORECASE))
            occ = m.get("occurrence")
            if isinstance(occ, int) and 0 < occ <= len(hits):
                ms, me = hits[occ - 1].span()
                spans = [x.span() for x in re.finditer(re.escape(a_text), text)]
                if spans and all(s0 <= ms and me <= e0 for s0, e0 in spans):
                    errors.append(f"{tag}: antecedent text {a_text!r} contains the mention itself")
        sec = ant.get("secondary_text")
        if sec is not None and not any(sec in t for t in all_text):
            errors.append(f"{tag}: secondary_text {sec!r} not verbatim on the card")

    # Coverage: each mechanically extracted candidate must be annotated (a
    # mention overlapping it) or explicitly rejected. Unhandled pronouns are
    # errors, since every pronoun is a mention; anything else is a warning.
    reasons = {(r.get("ability"), (r.get("candidate") or "").lower(), r.get("occurrence")): r.get("reason") or ""
               for r in data.get("rejected") or []}
    rejected = set(reasons)
    spans: dict[str, list[tuple[int, int]]] = {}
    for m in data.get("mentions") or []:
        text = abilities.get(m.get("ability"))
        mt, occ = m.get("mention_text") or "", m.get("occurrence")
        if text is None or not mt or not isinstance(occ, int):
            continue
        hits = list(re.finditer(r"(?<!\w)" + re.escape(mt) + r"(?!\w)", text, flags=re.IGNORECASE))
        if 0 < occ <= len(hits):
            spans.setdefault(m["ability"], []).append((hits[occ - 1].start(), hits[occ - 1].end()))
    for aid, cands in (card_info.get("candidates") or {}).items():
        for c in cands:
            if (aid, c["text"].lower(), c["occurrence"]) in rejected:
                if c["pronoun"] and not (c["text"].lower() in ("it", "it's") and reasons.get(
                        (aid, c["text"].lower(), c["occurrence"]), "").lower().startswith("expletive")):
                    errors.append(f"pronoun candidate {c['text']}#{c['occurrence']} in {aid} rejected")
                continue
            # A candidate nested inside another ("the player" in "If the player
            # does") is not covered by a mention that swallows the outer one.
            outer = [(o["start"], o["end"]) for o in cands
                     if o is not c and o["start"] <= c["start"] and c["end"] <= o["end"]]
            if any(s0 < c["end"] and c["start"] < e0
                   and not any(s0 <= os_ and oe <= e0 for os_, oe in outer)
                   for s0, e0 in spans.get(aid, [])):
                continue
            msg = f"candidate {c['text']}#{c['occurrence']} in {aid} neither annotated nor rejected"
            (errors if c["pronoun"] else warnings).append(msg)

    return errors, warnings
