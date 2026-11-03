#!/usr/bin/env python3
"""Annotate preprocessed MTG cards with the Tier-1 Haiku agent and escalate to Tier-2 Opus.

Output is an append-only JSONL log; the last row per oracle_id is the current
result. A rerun skips cards whose current row is final, reruns failures, and
re-escalates when Tier-2 previously failed (without paying for Tier-1 again).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import re
import subprocess
import sys
import time
from typing import Any

from build_agents import spec_sha, stale_agents
from preprocess import format_preprocessed_card, iter_scryfall_cards, load_canon_card_names
from validate import SCHEMA, validate_grounding, validate_schema

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
SCHEMA_ARG = json.dumps(SCHEMA, separators=(",", ":"))

TIER1 = {"agent": "endophora-haiku", "model": "haiku", "timeout_sec": 180, "allowed_tools": None}
# Read-only lookup scripts of the mtg-rules skill that Tier-2 may run. Data setup
# scripts (setup-data, ensure-data) are deliberately excluded; in -p mode any
# shell command not allowed here is refused without a prompt.
MTG_RULES_SCRIPTS = [
    "rule", "rule-search", "define", "keyword", "lookup", "card", "rulings",
    "corpus", "ref", "underdetermined", "classify", "mtr", "health", "data-path",
]


def mtg_rules_skill_dirs() -> list[pathlib.Path]:
    """Every installed mtg-rules skill directory.

    The plugin can be cached under more than one marketplace and version, and the
    CLI tells the agent which one it loaded, so the scripts are allowed in all of them.
    """
    base = pathlib.Path.home() / ".claude" / "plugins" / "cache"
    return sorted(d for d in base.glob("*/mtg-rules/*/skills/mtg-rules") if (d / "scripts" / "rule").exists())


def newest_mtg_rules_skill_dir() -> pathlib.Path | None:
    def version(d: pathlib.Path) -> tuple[int, ...]:
        try:
            return tuple(int(x) for x in d.parents[1].name.split("."))
        except ValueError:
            return ()

    dirs = mtg_rules_skill_dirs()
    return max(dirs, key=version) if dirs else None


def tier2_allowed_tools() -> list[str]:
    # No Skill tool: enabling it injects the listing of every installed skill
    # (~10k tokens) into the first user message, after the last cacheable
    # point, so every escalation paid to re-write it. Tier-2 runs the
    # mtg-rules scripts directly instead (see build_prompt).
    # Read is limited to the mtg-rules skill: Tier-2 must not see the canon
    # .syl encodings or earlier outputs, or it stops being an independent annotator.
    tools = []
    dirs = mtg_rules_skill_dirs()
    if not dirs:
        print("warning: mtg-rules plugin not found; Tier-2 runs without CR lookups", file=sys.stderr)
    for skill in dirs:
        tools.append(f"Read(/{skill}/**)")
        tools += [f"Bash({skill / 'scripts' / name}:*)" for name in MTG_RULES_SCRIPTS]
    return tools


MTG_RULES_DIR = newest_mtg_rules_skill_dir()
TIER2 = {"agent": "endophora-opus", "model": "opus", "timeout_sec": 600, "allowed_tools": tier2_allowed_tools()}

# Representative canon cards. Many of these are worked examples in the spec, so
# the spike checks that the spec is followed, not that it generalizes.
SPIKE_CARDS = [
    "Ad Nauseam",
    "Adaptive Automaton",
    "Atris, Oracle of Half-Truths",
    "Binding Agony",
    "Black Vise",
    "Blizzard Brawl",
    "Blood Spatter Analysis",
    "Chrome Mox",
    "Collected Company",
    "Coveted Falcon",
    "Green Sun's Twilight",
    "Last Chance",
    "Mox Diamond",
    "Stuffy Doll",
    "Afflicted Deserter // Werewolf Ransacker",
]

# 40 Vintage-legal cards outside canon (random.Random(20261104) over cards with
# at least one candidate mention). The startup guard checks that no 9-word run
# of their text, or of any canon card's text, appears in the spec.
HELDOUT_CARDS = [
    "Fatal Attraction", "Charging Tuskodon", "Jor Kadeen, First Goldwarden", "Mtenda Lion",
    "H.E.R.B.I.E., Lovable Robot", "Retreat to Hagra", "Frost Breath", "Merchant's Dockhand",
    "Kinetic Ooze", "Elvish Champion", "Spit Flame", "Savage Beating", "Quiet Disrepair",
    "Apothecary Initiate", "Expedition Lookout", "Lethal Scheme", "Pillage", "Meteor Crater",
    "Shadow, Mysterious Assassin", "Crossover Collaboration", "Cloudform", "Knight of Wundagore",
    "Lurking Skirge", "Echoes of Eternity", "Soulblast", "Skulking Fugitive", "Demolition Field",
    "Nexus of Fate", "Mages' Contest", "Swordsworn Cavalier", "Cloaked Cadet", "Dream Tides",
    "Bojuka Brigand", "Painter's Studio // Defaced Gallery", "Jaya, Fiery Negotiator", "Redcap Raiders",
    "Virtue of Knowledge // Vantress Visions", "Mark of Mutiny", "Pooling Venom", "Yue, the Moon Spirit",
]


def load_scryfall_snapshot(path: pathlib.Path) -> dict[str, Any] | None:
    """Identify the Scryfall bulk snapshot from its sibling metadata file, if present."""
    meta_path = path.parent / f"{path.name.split('.')[0]}.metadata.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {k: meta.get(k) for k in ("descriptor_updated_at", "jsonl_sha256")}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9{}'+/-]+", text.lower())


def spec_leaks(spec_text: str, scryfall_path: pathlib.Path, names: set[str], n: int = 9) -> list[tuple[str, str]]:
    """(card, phrase) pairs where n consecutive words of a guarded card's text appear in the spec."""
    sw = _words(spec_text)
    spec_grams = {tuple(sw[i:i + n]) for i in range(len(sw) - n + 1)}
    leaks = []
    for card in iter_scryfall_cards(scryfall_path, vintage_only=False):
        if card["name"] not in names and not any(p in names for p in card["name"].split(" // ")):
            continue
        texts = [card.get("oracle_text") or ""] + [f.get("oracle_text") or "" for f in card.get("card_faces") or []]
        for t in texts:
            w = _words(t)
            hit = next((g for g in (tuple(w[i:i + n]) for i in range(len(w) - n + 1)) if g in spec_grams), None)
            if hit:
                leaks.append((card["name"], " ".join(hit)))
                break
    return leaks


def extract_structured_output(raw_stdout: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (annotation, metadata) from a `claude -p --output-format json` envelope."""
    envelope = json.loads(raw_stdout)
    meta = {
        "duration_ms": envelope.get("duration_ms"),
        "duration_api_ms": envelope.get("duration_api_ms"),
        "num_turns": envelope.get("num_turns"),
        "total_cost_usd": envelope.get("total_cost_usd"),
        "is_error": envelope.get("is_error", False),
        "usage": {k: (envelope.get("usage") or {}).get(k) for k in (
            "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")},
    }
    structured = envelope.get("structured_output")
    if isinstance(structured, dict):
        return structured, meta
    raise ValueError(
        f"no structured_output in envelope (subtype={envelope.get('subtype')!r}, "
        f"result={str(envelope.get('result'))[:300]!r})"
    )


def build_prompt(card_info: dict[str, Any], prior: dict[str, Any] | None) -> str:
    prompt = card_info["prompt_text"]
    if prior is None:
        return prompt
    reasons = []
    if prior.get("annotation", {}).get("uncertainty_reason"):
        reasons.append(f"Tier-1 uncertainty: {prior['annotation']['uncertainty_reason']}")
    for err in prior.get("grounding_errors") or []:
        reasons.append(f"Grounding error: {err}")
    return (
        f"{prompt}\n\n"
        f"--- Escalation reasons ---\n" + "\n".join(f"- {r}" for r in reasons) + "\n\n"
        f"--- Tier-1 tentative output ---\n{json.dumps(prior.get('annotation'), indent=2)}\n"
        + (f"\nmtg-rules scripts directory: {MTG_RULES_DIR}\n" if MTG_RULES_DIR else "")
    )


def run_agent_on_card(
    card_info: dict[str, Any],
    tier: dict[str, Any],
    effort: str,
    prior: dict[str, Any] | None = None,
    prompt_override: str | None = None,
) -> dict[str, Any]:
    """Invoke `claude -p` for one card. Never raises: failures become a status."""
    row: dict[str, Any] = {"tier": f"{tier['agent']}:{tier['model']}:{effort}"}
    cmd = [
        "claude", "-p",
        "--agent", tier["agent"],
        "--model", tier["model"],
        "--effort", effort,
        "--output-format", "json",
        "--json-schema", SCHEMA_ARG,
        "--no-session-persistence",
        # Keeps cwd/env/VCS status out of the system prompt so the spec stays in
        # the shared cache prefix; without it every Tier-2 call re-wrote ~13k tokens.
        "--exclude-dynamic-system-prompt-sections",
    ]
    if tier["allowed_tools"]:
        # The working directory holds outputs and the spec's history; deny it
        # explicitly, since reads inside the cwd are otherwise allowed.
        cmd += ["--disallowed-tools", "Read(./output/**)", "Read(./src/**)", "Read(./.claude/**)"]
        cmd += ["--allowed-tools", *tier["allowed_tools"]]
    # The prompt goes on stdin: --allowed-tools is variadic and would swallow a
    # trailing positional prompt.
    prompt = prompt_override if prompt_override is not None else build_prompt(card_info, prior)

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, cwd=str(ROOT_DIR), input=prompt, capture_output=True, text=True,
            timeout=tier["timeout_sec"], check=False,
        )
    except subprocess.TimeoutExpired:
        return {**row, "status": "timeout", "error": f"exceeded {tier['timeout_sec']}s",
                "wall_ms": int((time.monotonic() - t0) * 1000)}
    except Exception as exc:  # e.g. claude not on PATH
        return {**row, "status": "error", "error": repr(exc), "wall_ms": int((time.monotonic() - t0) * 1000)}
    row["wall_ms"] = int((time.monotonic() - t0) * 1000)

    if proc.returncode != 0:
        return {**row, "status": "error", "returncode": proc.returncode,
                "stderr": proc.stderr.strip()[-2000:], "stdout": proc.stdout.strip()[:1000]}

    try:
        annotation, meta = extract_structured_output(proc.stdout)
    except Exception as exc:
        return {**row, "status": "parse_error", "error": str(exc), "stdout": proc.stdout.strip()[:1000]}

    schema_errors = validate_schema(annotation)
    grounding_errors, grounding_warnings = ([], []) if schema_errors else validate_grounding(annotation, card_info)
    if schema_errors:
        status = "schema_error"
    elif grounding_errors:
        status = "grounding_error"
    else:
        status = "ok"
    return {
        **row,
        "status": status,
        "schema_errors": schema_errors,
        "grounding_errors": grounding_errors,
        "grounding_warnings": grounding_warnings,
        "meta": meta,
        "annotation": annotation,
    }


def needs_escalation(res: dict[str, Any]) -> bool:
    if res["status"] == "grounding_error":
        return True
    return res["status"] == "ok" and res["annotation"].get("verdict") == "uncertain"


def next_stage(row: dict[str, Any], escalate: bool) -> str:
    """For a card's current row, return 'done', 'tier1', 'tier2', or 'skip'."""
    status = row.get("status")
    if status == "escalation_error":
        return "tier2" if escalate else "skip"
    if row.get("tier1") is not None:  # already adjudicated by Tier-2
        return "done" if status == "ok" else "tier1"
    if escalate and needs_escalation(row):
        return "tier2"
    return "done" if status == "ok" else "tier1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MTG endophora annotation pipeline.")
    parser.add_argument("--scryfall", default="../deckmaste.rs/data/scryfall/oracle-cards.jsonl",
                        help="Path to Scryfall oracle-cards .jsonl or .jsonl.gz")
    parser.add_argument("--canon-dir", default="../deckmaste.rs/plugins/canon/cards",
                        help="Path to canon .syl cards directory")
    sel = parser.add_mutually_exclusive_group()
    sel.add_argument("--card", action="append", help="Run on specific card name(s)")
    sel.add_argument("--spike", action="store_true", help="Run on the 15-card canon spike set")
    sel.add_argument("--heldout", action="store_true", help="Run on the held-out set absent from the spec")
    sel.add_argument("--canon-only", action="store_true", help="Restrict to cards present in canon-dir")
    parser.add_argument("--limit", type=int, default=None, help="Max number of cards to queue")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent Tier-1 invocations")
    parser.add_argument("--escalation-concurrency", type=int, default=2, help="Concurrent Tier-2 invocations")
    parser.add_argument("--tier1-model", default="haiku", help="Model for the Tier-1 agent (e.g. haiku, sonnet)")
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"],
                        help="Effort level for Tier-1")
    parser.add_argument("--escalation-effort", default="high", choices=["low", "medium", "high", "xhigh", "max"],
                        help="Effort level for Tier-2")
    parser.add_argument("--escalate-uncertain", action="store_true",
                        help="Escalate uncertain verdicts and grounding errors to Tier-2")
    parser.add_argument("--accept-stale", action="store_true",
                        help="Treat rows from an older spec or different card text as done")
    parser.add_argument("--skip-empty", action="store_true",
                        help="Record cards with no candidate mentions as empty without calling a model")
    parser.add_argument("--also-done", action="append", default=[],
                        help="Another output JSONL whose finished cards count as done (for a parallel run)")
    parser.add_argument("--reverse", action="store_true", help="Process cards in reverse order")
    parser.add_argument("--output", default="output/spike.jsonl", help="Output JSONL path")
    parser.add_argument("--overwrite", action="store_true", help="Truncate output instead of resuming")
    args = parser.parse_args()

    TIER1["model"] = args.tier1_model
    stale = stale_agents()
    if stale:
        sys.exit(f"Stale agent files ({', '.join(stale)}); run src/build_agents.py first.")
    spec = spec_sha()
    spec_text = (ROOT_DIR / "prompts" / "endophora_spec.md").read_text(encoding="utf-8")
    scryfall_path = (ROOT_DIR / args.scryfall).resolve()
    canon_dir = ROOT_DIR / args.canon_dir
    guarded = set(HELDOUT_CARDS) | (load_canon_card_names(canon_dir) if canon_dir.exists() else set())
    leaked = spec_leaks(spec_text, scryfall_path, guarded)
    if leaked:
        sys.exit("Spec quotes evaluation cards (9-word overlap): "
                 + "; ".join(f"{n}: {g!r}" for n, g in leaked[:10]))

    snapshot = load_scryfall_snapshot(scryfall_path)
    out_path = ROOT_DIR / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    current: dict[str, dict[str, Any]] = {}
    sources = [ROOT_DIR / p for p in args.also_done]
    if out_path.exists() and not args.overwrite:
        sources.append(out_path)
    for src in sources:
        if not src.exists():
            continue
        with open(src, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a parallel run may be mid-write
                    current[rec["oracle_id"]] = rec

    wanted_names: list[str] | set[str] | None = None
    if args.card:
        wanted_names = args.card
    elif args.spike:
        wanted_names = SPIKE_CARDS
    elif args.heldout:
        wanted_names = HELDOUT_CARDS
    elif args.canon_only:
        wanted_names = sorted(load_canon_card_names(ROOT_DIR / args.canon_dir))
        if not wanted_names:
            sys.exit(f"No .syl cards in {ROOT_DIR / args.canon_dir}; pass --canon-dir.")
    wanted_set = set(wanted_names) if wanted_names is not None else None

    jobs: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = []
    empty_rows: list[dict[str, Any]] = []
    counts = {"done": 0, "skip": 0, "stale": 0}
    seen: set[str] = set()
    for card in iter_scryfall_cards(scryfall_path, vintage_only=True):
        if wanted_set is not None and card["name"] not in wanted_set:
            continue
        info = format_preprocessed_card(card)
        seen.add(info["card_name"])
        prev = current.get(info["oracle_id"])
        stage = "tier1"
        if prev is not None:
            fresh = (prev.get("spec_sha") == spec and prev.get("abilities") == info["abilities"]
                     and prev.get("costs") == info["costs"])
            if fresh or args.accept_stale:
                stage = next_stage(prev, args.escalate_uncertain)
            else:
                counts["stale"] += 1
        if stage in ("done", "skip"):
            counts[stage] += 1
            continue
        if args.skip_empty and stage == "tier1" and not any(info["candidates"].values()):
            empty_rows.append(info)
            continue
        prior = None
        if stage == "tier2":
            prior = prev["tier1"] if prev.get("status") == "escalation_error" else prev
        jobs.append((stage, info, prior))

    if wanted_names is not None and not args.canon_only:  # canon also lists meld halves
        missing = sorted(set(wanted_names) - seen)
        if missing:
            print(f"Not found among Vintage-legal cards: {', '.join(missing)}", file=sys.stderr)
    if isinstance(wanted_names, list):
        order = {name: i for i, name in enumerate(wanted_names)}
        jobs.sort(key=lambda j: order.get(j[1]["card_name"], len(order)))
    if args.reverse:
        jobs.reverse()
    # --limit takes the first N in the selection's order (alphabetical for canon),
    # not Scryfall file order.
    if args.limit:
        jobs = jobs[: args.limit]

    n_t2 = sum(1 for j in jobs if j[0] == "tier2")
    print(
        f"Queued {len(jobs)} cards ({n_t2} straight to Tier-2); already done: {counts['done']}; "
        f"stale reruns: {counts['stale']}; escalation failures skipped (pass --escalate-uncertain): "
        f"{counts['skip']} -> {out_path}",
        flush=True,
    )
    base = {"spec_sha": spec, "scryfall_snapshot": snapshot}
    if empty_rows:
        print(f"Recording {len(empty_rows)} cards with no candidate mentions without a model call", flush=True)
        with open(out_path, "w" if args.overwrite else "a", encoding="utf-8") as f:
            for info in empty_rows:
                row = {"card_name": info["card_name"], "oracle_id": info["oracle_id"], "abilities": info["abilities"],
                       "costs": info["costs"], **base, "tier": "none", "status": "ok", "no_candidates": True,
                       "schema_errors": [], "grounding_errors": [], "grounding_warnings": [],
                       "annotation": {"card_name": info["card_name"], "verdict": "ok", "uncertainty_reason": None,
                                      "mentions": [], "rejected": []}}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        args.overwrite = False
    if not jobs:
        return

    def card_fields(info: dict[str, Any]) -> dict[str, Any]:
        return {"card_name": info["card_name"], "oracle_id": info["oracle_id"],
                "abilities": info["abilities"], "costs": info["costs"]}

    def tier1_view(row: dict[str, Any]) -> dict[str, Any]:
        keep = ("tier", "status", "grounding_errors", "grounding_warnings", "wall_ms", "meta", "annotation")
        return {k: row[k] for k in keep if k in row}

    mode = "w" if args.overwrite else "a"
    with (
        open(out_path, mode, encoding="utf-8") as out_f,
        cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool1,
        cf.ThreadPoolExecutor(max_workers=args.escalation_concurrency) as pool2,
    ):
        pending: dict[cf.Future, tuple[str, dict[str, Any], dict[str, Any] | None]] = {}

        def submit(stage: str, info: dict[str, Any], prior: dict[str, Any] | None) -> None:
            if stage == "tier1":
                fut = pool1.submit(run_agent_on_card, info, TIER1, args.effort)
            else:
                fut = pool2.submit(run_agent_on_card, info, TIER2, args.escalation_effort, prior)
            pending[fut] = (stage, info, prior)

        for stage, info, prior in jobs:
            submit(stage, info, prior)

        while pending:
            finished, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for fut in finished:
                stage, info, prior = pending.pop(fut)
                res = fut.result()
                name = info["card_name"]

                if stage == "tier1" and args.escalate_uncertain and needs_escalation(res):
                    why = res["annotation"].get("uncertainty_reason") or "; ".join(res["grounding_errors"])
                    print(f"  [escalating] {name}: {why}", flush=True)
                    submit("tier2", info, res)
                    continue

                if stage == "tier2":
                    t1 = tier1_view(prior)
                    if res["status"] == "ok":
                        res = {**res, "tier1": t1}
                    else:
                        res = {"status": "escalation_error", "tier1": t1, "tier2": res}

                row = {**card_fields(info), **base, **res}
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()

                status = row["status"]
                if status == "ok":
                    ann = row["annotation"]
                    cost = (row.get("meta") or {}).get("total_cost_usd")
                    cost_str = f"${cost:.4f}" if isinstance(cost, (int, float)) else "?"
                    warn = f" | {len(row['grounding_warnings'])} warning(s)" if row["grounding_warnings"] else ""
                    print(
                        f"  [done] {name} | {row['tier']} | verdict={ann.get('verdict')} | "
                        f"mentions={len(ann.get('mentions', []))} | {row['wall_ms']}ms | {cost_str}{warn}",
                        flush=True,
                    )
                else:
                    detail = row.get("tier2", row)
                    why = (detail.get("schema_errors") or detail.get("grounding_errors")
                           or detail.get("error") or detail.get("stderr"))
                    print(f"  [ERR]  {name} | status={status} | {why}", flush=True)


if __name__ == "__main__":
    main()
