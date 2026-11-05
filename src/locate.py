#!/usr/bin/env python3
"""Give every antecedent span an exact position: `antecedent.occurrences`.

`occurrences` has one entry per span part ([n] for an ordinary span, one per
part for a split): the n-th non-overlapping, exact, case-sensitive occurrence of
that part's text in its ability. It is null for `source` and `performer`.

Most positions are decided mechanically:
- unique: the text occurs once;
- same-referent: a self-reference ("this creature", the card's name) or a
  variable ({X}), where every occurrence is the same thing (the first is used);
- before-mention: exactly one occurrence precedes the mention in the same
  ability (an antecedent is the introducing mention, so it comes first).
The rest are put to a model, one call per card, and its answers are cached in
--choices so a rerun doesn't pay again. The release is rewritten in place and
the manifest records how many positions each method decided.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import lzma
import pathlib
import re
import subprocess
import sys
import time
from collections import Counter

from preprocess import self_names

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
SELF_RE = re.compile(r"^this (?:creature|artifact|enchantment|spell|card|aura|equipment|vehicle|permanent|land"
                     r"|saga|class|case|room|door|spacecraft|siege|battle|planeswalker|token)$", re.IGNORECASE)
VAR_RE = re.compile(r"^\{[XYZ]\}$")

SYSTEM_PROMPT = """\
You locate antecedents in Magic: The Gathering card text. Each task gives a card, a mention (a referring \
expression), and the phrase already chosen as the mention's antecedent: the phrase that first introduced \
what the mention refers to. That phrase occurs more than once in the ability, and each occurrence is \
listed with its number and context. Pick the occurrence that is the antecedent, by meaning: the \
occurrence that introduces the thing the mention refers to. Return the numbers through the \
StructuredOutput tool."""

CHOICE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["choices"],
    "properties": {"choices": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["task", "occurrence"],
        "properties": {"task": {"type": "integer"}, "occurrence": {"type": "integer", "minimum": 1}}}}},
}


def matches(text: str, part: str) -> list[re.Match]:
    return list(re.finditer(re.escape(part), text))


def mention_start(text: str, mention: dict) -> int | None:
    hits = list(re.finditer(r"(?<!\w)" + re.escape(mention["mention_text"]) + r"(?!\w)", text, flags=re.IGNORECASE))
    occ = mention["occurrence"]
    return hits[occ - 1].start() if 0 < occ <= len(hits) else None


def parts_of(ant: dict) -> list[tuple[str, str]]:
    parts = ant["text"].split(" + ") if ant["kind"] == "split" else [ant["text"]]
    aid = ant["ability"] or ""
    aids = aid.split(" + ") if ant["kind"] == "split" and " + " in aid else [aid] * len(parts)
    return list(zip(parts, aids))


def ability_order(aid: str) -> tuple:
    return tuple(int(n) if n.isdigit() else n for n in re.findall(r"\d+|[A-Za-z]+", aid))


def decide(row: dict, mention: dict, part: str, aid: str) -> tuple[int | None, str, list[re.Match]]:
    """(occurrence or None, method, candidate matches) for one span part."""
    src = {**row["abilities"], **row["costs"]}[aid]
    found = matches(src, part)
    if len(found) == 1:
        return 1, "unique", found
    names = {n.lower() for n in self_names({"name": row["card_name"]})}
    if SELF_RE.match(part) or part.lower() in names or VAR_RE.match(part):
        return 1, "same_referent", found
    if aid == mention["ability"]:
        start = mention_start(src, mention)
        before = [i for i, m in enumerate(found, 1) if start is not None and m.end() <= start]
        if len(before) == 1:
            return before[0], "before_mention", found
    elif ability_order(aid) > ability_order(mention["ability"]):
        pass  # antecedent in a later ability: leave it to the model
    return None, "model", found


def render_task(row: dict, idx: int, mention: dict, part: str, aid: str, found: list[re.Match]) -> str:
    src = {**row["abilities"], **row["costs"]}[aid]
    lines = [f"Task {idx}: mention {mention['mention_text']!r} (occurrence {mention['occurrence']} in "
             f"{mention['ability']}, context: {mention['context_snippet']!r}); antecedent phrase {part!r} in {aid}:"]
    for n, m in enumerate(found, 1):
        left, right = src[max(0, m.start() - 60):m.start()], src[m.end():m.end() + 60]
        lines.append(f"  occurrence {n}: …{left}[{m.group(0)}]{right}…")
    return "\n".join(lines)


def ask_model(prompt: str, model: str, effort: str, n_tasks: int) -> tuple[dict[int, int] | None, dict]:
    cmd = ["claude", "-p", "--system-prompt", SYSTEM_PROMPT, "--tools", "StructuredOutput",
           "--setting-sources", "", "--strict-mcp-config", "--settings", '{"disableAllHooks": true}',
           "--disable-slash-commands", "--model", model, "--effort", effort, "--output-format", "json",
           "--json-schema", json.dumps(CHOICE_SCHEMA), "--no-session-persistence"]
    try:
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=300, check=False)
        env = json.loads(proc.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return None, {"error": repr(exc)}
    meta = {"models": sorted((env.get("modelUsage") or {}).keys()), "cost": env.get("total_cost_usd")}
    out = env.get("structured_output")
    if not isinstance(out, dict):
        return None, {**meta, "error": str(env.get("result"))[:300]}
    choices = {c["task"]: c["occurrence"] for c in out.get("choices", [])}
    return (choices if len(choices) == n_tasks else None), meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--release", default="release/onlyspans.jsonl.xz")
    ap.add_argument("--choices", default="output/antecedent-choices.jsonl", help="Cache of model answers")
    ap.add_argument("--model", default="opus")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="Report what would go to the model; change nothing")
    args = ap.parse_args()

    release = ROOT_DIR / args.release
    rows = [json.loads(line) for line in lzma.open(release, "rt", encoding="utf-8")]
    cache_path = ROOT_DIR / args.choices
    cache: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c = json.loads(line)
                if c.get("choices") is not None:
                    cache[c["oracle_id"]] = c

    methods: Counter = Counter()
    pending: dict[str, list] = {}  # oracle_id -> [(mention, part_index, part, aid, found)]
    for row in rows:
        for m in row["mentions"]:
            ant = m["antecedent"]
            if ant["text"] is None:
                ant["occurrences"] = None
                methods["no_span"] += 1
                continue
            occs = []
            for i, (part, aid) in enumerate(parts_of(ant)):
                occ, method, found = decide(row, m, part, aid)
                occs.append(occ)
                if occ is None:
                    pending.setdefault(row["oracle_id"], []).append((m, i, part, aid, found))
                else:
                    methods[method] += 1
            ant["occurrences"] = occs

    by_id = {r["oracle_id"]: r for r in rows}
    todo = [oid for oid in pending if oid not in cache]
    n_tasks = sum(len(v) for v in pending.values())
    print(f"decided mechanically: {dict(methods)}; for the model: {n_tasks} spans on {len(pending)} cards "
          f"({len(pending) - len(todo)} cached)", flush=True)
    if args.dry_run:
        return

    def work(oid: str) -> dict:
        row, tasks = by_id[oid], pending[oid]
        text = "\n".join(f"[{aid}] {t}" for aid, t in {**row["costs"], **row["abilities"]}.items())
        prompt = (f"Card: {row['card_name']}\n{text}\n\n"
                  + "\n\n".join(render_task(row, i, m, p, a, f) for i, (m, _, p, a, f) in enumerate(tasks, 1)))
        choices, meta = ask_model(prompt, args.model, args.effort, len(tasks))
        if choices is not None and not all(1 <= choices.get(i, 0) <= len(t[4]) for i, t in enumerate(tasks, 1)):
            choices, meta = None, {**meta, "error": "occurrence out of range"}
        return {"oracle_id": oid, "card_name": row["card_name"], "choices": choices, "meta": meta,
                "tasks": [{"mention": m["mention_text"], "mention_occurrence": m["occurrence"],
                           "mention_ability": m["ability"], "part": p, "ability": a, "candidates": len(f)}
                          for m, _, p, a, f in tasks],
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with open(cache_path, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(args.concurrency) as pool:
        for fut in cf.as_completed([pool.submit(work, oid) for oid in todo]):
            res = fut.result()
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            ok = res["choices"] is not None
            failures = 0 if ok else failures + 1
            print(f"  [{'ok' if ok else 'FAIL'}] {res['card_name']} | ${res['meta'].get('cost') or 0:.3f}"
                  + ("" if ok else f" | {res['meta'].get('error')}"), flush=True)
            if ok:
                cache[res["oracle_id"]] = res
            if failures >= 20:
                pool.shutdown(wait=False, cancel_futures=True)
                sys.exit("Stopping: 20 consecutive failures")

    missing = [by_id[o]["card_name"] for o in pending if o not in cache]
    if missing:
        sys.exit(f"{len(missing)} card(s) without model answers; rerun to retry: {missing[:10]}")
    models: Counter = Counter()
    for oid, tasks in pending.items():
        choices = {int(k): v for k, v in cache[oid]["choices"].items()}
        models.update(cache[oid]["meta"].get("models") or [])
        for i, (m, part_index, *_rest) in enumerate(tasks, 1):
            m["antecedent"]["occurrences"][part_index] = choices[i]
            methods["model"] += 1

    # Validate: every position must exist.
    for row in rows:
        src = {**row["abilities"], **row["costs"]}
        for m in row["mentions"]:
            ant = m["antecedent"]
            if ant["text"] is None:
                continue
            for (part, aid), occ in zip(parts_of(ant), ant["occurrences"]):
                if not (isinstance(occ, int) and 1 <= occ <= len(matches(src[aid], part))):
                    sys.exit(f"{row['card_name']}: bad occurrence {occ} for {part!r} in {aid}")

    with lzma.open(release, "wt", encoding="utf-8", preset=9 | lzma.PRESET_EXTREME) as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest_path = release.with_name(release.name.split(".")[0] + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["antecedent_occurrences"] = {"by_method": dict(methods), "model": args.model,
                                          "models": sorted(models), "effort": args.effort}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {release.relative_to(ROOT_DIR)}; occurrences by method: {dict(methods)}")


if __name__ == "__main__":
    main()
