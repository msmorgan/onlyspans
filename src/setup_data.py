#!/usr/bin/env python3
"""Download everything a build needs into data/ (ignored by version control).

- Scryfall's current Oracle Cards bulk file -> data/oracle-cards.jsonl.gz
- the current mtg-rules plugin (github.com/msmorgan/mtg-rules) -> data/mtg-rules/,
  then its own `setup-data --rulings`, which fetches the Comprehensive Rules,
  Scryfall catalogs and MTGJSON (official rulings) into data/mtg-rules/data/

What was fetched (URLs, dates, SHA-256, plugin commit) is written to
data/sources.json, which run_pipeline.py and assemble.py record as provenance.
Versions are not pinned: a rebuild uses whatever is current.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SOURCES = DATA_DIR / "sources.json"
SCRYFALL_BULK = "https://api.scryfall.com/bulk-data/oracle-cards"
MTG_RULES_TARBALL = "https://github.com/msmorgan/mtg-rules/archive/HEAD.tar.gz"
HEADERS = {"User-Agent": "OnlySpans-setup/1.0", "Accept": "application/json;q=0.9,*/*;q=0.8"}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=300) as r:
        return r.read()


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def setup_scryfall() -> dict:
    meta = json.loads(fetch(SCRYFALL_BULK))
    # Scryfall serves gzipped JSON Lines; older API versions only gave a JSON array.
    uri = meta.get("jsonl_download_uri") or meta.get("download_uri")
    if not uri:
        sys.exit(f"Scryfall bulk-data response has no download URI: {sorted(meta)}")
    print(f"Scryfall Oracle Cards, updated {meta.get('updated_at')}: {uri}", flush=True)
    body = fetch(uri)
    target = DATA_DIR / ("oracle-cards.jsonl.gz" if uri.endswith(".jsonl.gz") else pathlib.Path(uri).name)
    target.write_bytes(body)
    raw = gzip.decompress(body) if target.suffix == ".gz" else body
    return {"path": str(target.relative_to(ROOT_DIR)), "download_uri": uri, "updated_at": meta.get("updated_at"),
            "sha256": hashlib.sha256(body).hexdigest(), "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
            "fetched_at": now()}


def setup_mtg_rules() -> dict:
    print(f"mtg-rules: {MTG_RULES_TARBALL}", flush=True)
    archive = tarfile.open(fileobj=io.BytesIO(fetch(MTG_RULES_TARBALL)), mode="r:gz")
    top = archive.getmembers()[0].name.split("/")[0]  # "mtg-rules-<commit sha>"
    target = DATA_DIR / "mtg-rules"
    if target.exists():
        shutil.rmtree(target)
    staging = DATA_DIR / ".mtg-rules-unpack"
    if staging.exists():
        shutil.rmtree(staging)
    try:
        archive.extractall(staging, filter="data")
    except TypeError:  # Python without tarfile extraction filters
        archive.extractall(staging)
    (staging / top).rename(target)
    shutil.rmtree(staging)
    setup = target / "skills" / "mtg-rules" / "scripts" / "setup-data"
    env = {**os.environ, "MTG_RULES_DATA": str(target / "data")}
    # api.academyruins.com refuses Python's default User-Agent and Scryfall's API
    # requires Accept, so run the plugin's own setup with both headers installed.
    bootstrap = (
        "import runpy, sys, urllib.request\n"
        "opener = urllib.request.build_opener()\n"
        f"opener.addheaders = {list(HEADERS.items())!r}\n"
        "urllib.request.install_opener(opener)\n"
        "sys.argv = [sys.argv[1], '--rulings']\n"
        "runpy.run_path(sys.argv[0], run_name='__main__')\n"
    )
    subprocess.run([sys.executable, "-c", bootstrap, str(setup)], env=env, check=True)
    return {"path": str(target.relative_to(ROOT_DIR)), "repo": "msmorgan/mtg-rules",
            "commit": top.removeprefix("mtg-rules-"), "fetched_at": now()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-scryfall", action="store_true")
    ap.add_argument("--skip-rules", action="store_true", help="Skip the mtg-rules plugin and its data")
    args = ap.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    sources = json.loads(SOURCES.read_text(encoding="utf-8")) if SOURCES.exists() else {}
    if not args.skip_scryfall:
        sources["scryfall"] = setup_scryfall()
        SOURCES.write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    if not args.skip_rules:
        sources["mtg_rules"] = setup_mtg_rules()
        SOURCES.write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(sources, indent=2))


if __name__ == "__main__":
    main()
