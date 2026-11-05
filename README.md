# OnlySpans

Endophoric reference annotations for every Vintage-legal Magic: The Gathering card: each expression in a card's rules text that refers back to something else on the card ("it", "that creature", "the chosen color", "another", "that much", X, "if you do", "the exiled card", …), with the span it refers to.

## The reference

[`release/onlyspans.jsonl.xz`](https://github.com/msmorgan/onlyspans/raw/main/release/onlyspans.jsonl.xz) — one JSON object per card, as xz-compressed JSON Lines. [`release/onlyspans.manifest.json`](release/onlyspans.manifest.json) gives the counts, the Scryfall snapshot it was built from, the spec versions used, and the cards left out.

32,027 cards and 34,937 mentions (16,876 cards have at least one). One card is declined ([`release/declined.jsonl`](release/declined.jsonl)): Lantern Flare, whose X has two referents depending on whether it is cast with cleave.

Each line:

```json
{
  "card_name": "Juxtapose",
  "oracle_id": "a52ccb40-…",
  "abilities": {"A0": "You and target player exchange control of … their controller chooses one of them."},
  "costs": {"cost": "{3}{U}"},
  "mentions": [
    {"ability": "A0", "mention_text": "their", "occurrence": 1,
     "context_snippet": "[their] controller chooses one of them",
     "category": "pronoun",
     "antecedent": {"kind": "text_span", "referent_type": "object", "ability": "A0",
                    "text": "two or more permanents a player controls", "secondary_text": null},
     "confident": true}
  ],
  "source": "adjudicated",
  "repaired": null
}
```

- **Spans are verbatim substrings of `abilities`**, which is the card's Oracle text after preprocessing: reminder text removed, each ability on one line (`A0`, `A1`, … or `F0.A0` on multi-face cards), modal modes, spree modes and die-roll tables kept inside their ability, and level-up bands prefixed (`LEVEL 2-6 (3/3) | `). Mana costs are addressed as `cost` (or `F0.cost`).
- **A mention is located by** `ability`, `mention_text` and `occurrence`: the occurrence-th whole-word, case-insensitive match of `mention_text` in that ability's text.
- **The antecedent** is the phrase that first introduced the referent. `kind` is `text_span`; `split` (several separately named things, parts joined with `" + "`, and `ability` likewise when the parts are in different abilities); `source` (the card itself, not named in that ability); `performer` (the implicit "you"); or `distributive_member`. `secondary_text` holds the latest re-mention or zone-move clause in a chain, the whole collection for "the rest", or the action clause for a participial readback.
- **Categories:** `pronoun`, `reflexive`, `demonstrative`, `definite_readback`, `participial_description`, `contrastive`, `complement_or_reciprocal`, `amount_or_outcome`, `lookback_or_followup`, `relative`. The full definitions are in [`prompts/endophora_spec.md`](prompts/endophora_spec.md).
- **`source`** says how the card was settled: `no_candidates` (nothing in the text could refer back), `consensus` (both annotators agreed, confidently), `adjudicated`, or `repaired` (adjudicated, then an antecedent's ability ID corrected mechanically; `repaired` lists the edits).
- **Known gap:** an antecedent has no occurrence number, so when its phrase appears twice in one ability, the text alone doesn't say which.

## How it was made

Annotations are by meaning (what a careful player understands each expression to refer to) under the spec in `prompts/endophora_spec.md`, which deliberately contains no resolution procedure.

1. `src/preprocess.py` turns Scryfall's Oracle text into ability-indexed prompts and lists the candidate expressions in each ability. The annotator must annotate or reject every candidate.
2. Two independent annotators label every card with candidates: Claude Haiku (high effort) and Claude Sonnet (medium effort), via `src/run_pipeline.py`.
3. `src/consensus.py` accepts a card when both agree on every mention, confidently, with valid spans, and queues everything else.
4. `src/adjudicate.py` has Claude Opus write the final annotation for each queued card (17% of cards with candidates), given both annotations and the card's official rulings. `src/repair.py` fixes antecedents cited in the wrong ability.
5. `src/assemble.py` re-validates every annotation against the card text (`src/validate.py`) and writes the reference.

This is model-produced data with cross-checks, not human-verified ground truth.

### This release

- **Models:** `claude-haiku-5-5` (high effort) and `claude-sonnet-5-5` (medium effort) annotated; `claude-opus-5-5` (high effort) adjudicated. Claude Code 2.1.317 called them by alias (`haiku`, `sonnet`, `opus`). New runs record the exact model IDs in each row.
- **Spec:** 32,020 cards were annotated under spec `49c2466ccc84` and 7 under `f2e08463b7a4` (both in `prompts/history/`). The current spec adds only the note on per-part ability IDs for split antecedents.
- **Card data:** Scryfall Oracle Cards of 2026-10-03, `https://data.scryfall.io/oracle-cards/oracle-cards-20261003090153.jsonl.gz` (uncompressed SHA-256 `49dc966b…`, full hash in the manifest).
- **Environment:** the release was built through Claude Code agents running under the owner's Claude Code setup, so its prompts also carried that setup's CLAUDE.md instructions and plugin hook output (unrelated to the task). The current code isolates every call from the local setup instead.
- **Volume and cost:** 2,866 cards queued: 2,855 adjudicated, 10 adjudicated and then repaired, 1 declined. About $490 in API-equivalent terms (Haiku ~$28, Sonnet ~$290, Opus ~$170), run on a Claude subscription. Wall time was roughly 3 h for Haiku, 2 h for Sonnet (in parallel) and 1 h for adjudication.

## Rebuilding

Requires Python 3.10+ (standard library only; tested on 3.14) and the [Claude Code](https://claude.com/claude-code) CLI, logged in (built with 2.1.317). Run everything from the repository root. Inputs are not pinned and model runs are not deterministic, so a rebuild follows the same process on current data and models; it won't reproduce this release exactly.

Every model call runs `claude -p` with the spec as its whole system prompt and is isolated from your own Claude Code setup (`--setting-sources ''`, no MCP servers, hooks, skills or slash commands).

1. **Download inputs** into `data/` (ignored), about 1 GB:

   ```sh
   python3 src/setup_data.py
   ```

   This fetches Scryfall's current Oracle Cards bulk file and the current [mtg-rules](https://github.com/msmorgan/mtg-rules) plugin, then runs the plugin's own data setup (Comprehensive Rules, Scryfall catalogs, MTGJSON for official rulings). `data/sources.json` records what was fetched, and new rows record it as provenance. mtg-rules is optional: without it, adjudication runs without rulings or rules lookups (with a warning). You can also point `MTG_RULES_DIR` and `MTG_RULES_DATA` at another copy.

2. **Smoke test** one card before spending on a full run:

   ```sh
   python3 src/run_pipeline.py --card Juxtapose --output output/smoke.jsonl --overwrite
   ```

3. **Two independent annotations** of every card that has candidate mentions (about 17,000; the rest are recorded as empty without a model call):

   ```sh
   python3 src/run_pipeline.py --skip-empty --effort high --concurrency 16 --output output/haiku.jsonl
   python3 src/run_pipeline.py --skip-empty --tier1-model sonnet --effort medium --concurrency 12 --output output/sonnet.jsonl
   ```

   Model names can be aliases or full model IDs. Runs resume where they stopped; a row is redone if the spec or card text changed since it was written (unless `--accept-stale`); a run stops after 20 consecutive call failures. `--card NAME`, `--cards-file FILE` (one name per line), `--heldout` and `--spike` restrict the cards.

4. **Consensus and adjudication.** Cards where both runs agree on every mention, confidently and with valid spans, are accepted. Everything else is queued for Opus:

   ```sh
   python3 src/consensus.py --runs output/haiku.jsonl output/sonnet.jsonl --queue output/queue.jsonl
   python3 src/adjudicate.py --queue output/queue.jsonl \
       --runs output/haiku.jsonl output/sonnet.jsonl --output output/adjudicated.jsonl --concurrency 12
   python3 src/repair.py --adjudicated output/adjudicated.jsonl --output output/repaired.jsonl
   ```

   `repair.py` fixes adjudicated cards whose only fault is an antecedent cited in the wrong ability (linked abilities). A card that still fails validation needs another `adjudicate.py` pass (rerunning it retries only cards that aren't `ok`). A card that can't be annotated in this format goes in `release/declined.jsonl` by hand: one JSON object per line with `oracle_id`, `card_name` and `reason`.

5. **Assemble.** `assemble.py` re-validates every card and refuses to write anything if a card is missing, invalid or still uncertain:

   ```sh
   python3 src/assemble.py --runs output/haiku.jsonl output/sonnet.jsonl \
       --adjudicated output/adjudicated.jsonl output/repaired.jsonl
   ```

   The first `--runs` file supplies the accepted annotations. Later `--adjudicated` files take precedence.

## License and notice

The code and the annotations (mentions, antecedent spans, categories) are MIT-licensed; see [`LICENSE`](LICENSE).

Card names and Oracle text are the property of Wizards of the Coast and are not covered by that license. This project is unofficial Fan Content, not approved or endorsed by Wizards. Card data comes from [Scryfall](https://scryfall.com).
