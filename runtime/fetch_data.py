#!/usr/bin/env python
"""Download the pinned MATH dataset of area "agentic RL" (host step, network on, no GPU).

Only the pinned repository at the pinned revision is fetched (the maintainers' decisions "Area 10
decisions", not published): `EleutherAI/hendrycks_math` @ 21a5633873b6a120296cce3e2df9d5550074f4a3 (MIT, not gated; README "Pins":
7 subject configs, 14 parquet files plus the card, 4,890,090 bytes). Every file the Hub lists for that revision is
downloaded to `raw/hendrycks_math/` as plain files (input of `make_data.py`, never mounted into a run);
huggingface_hub's metadata goes to `hf-meta/raw-hendrycks_math/`. Token handling as fetch_models.py: `token=False`,
`HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, fresh `HF_HOME`, Xet off, token variables removed before import (never read).
`raw/RAW.json` records repo, revision, license, every file's SHA-256 and bytes; `--verify` re-hashes.

Usage (host, any venv with huggingface_hub, e.g. a Python 3.13 venv):
  env -u HF_TOKEN <venv>/bin/python fetch_data.py
  <venv>/bin/python fetch_data.py --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fetch_models import hub_env, move_meta, plain_files  # noqa: E402

RAW = HERE / "raw"
REPO = "EleutherAI/hendrycks_math"
REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
LICENSE = "mit"
N_PARQUET = 14
README_BYTES = 4_890_090        # README "Pins" (Hub API with blobs, 2026-11-14)


def download() -> int:
    hub_env()
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION, token=False))
    parquet = [f for f in files if f.endswith(".parquet")]
    if len(parquet) != N_PARQUET:
        raise SystemExit(f"{REPO}@{REVISION} lists {len(parquet)} parquet files, expected {N_PARQUET}: {files}")
    dest = RAW / "hendrycks_math"
    for f in files:
        hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION, local_dir=dest, token=False)
    move_meta(dest, "raw-hendrycks_math")
    got = plain_files(dest)
    if sorted(got) != files:
        raise SystemExit(f"{dest} holds {sorted(got)}, not {files}")
    total = sum(f["bytes"] for f in got.values())
    raw = {"schema": "ptc_mm.agentic_raw", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
           "license": LICENSE, "datasets": {"hendrycks_math": {
               "repo": REPO, "revision": REVISION, "files": got, "bytes": total,
               "bytes_listed_in_readme": README_BYTES}}}
    (RAW / "RAW.json").write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"hendrycks_math: {len(got)} files ({len(parquet)} parquet), {total} bytes "
          f"(README listing {README_BYTES})", flush=True)
    return 0


def verify() -> int:
    raw = json.loads((RAW / "RAW.json").read_text(encoding="utf-8"))
    bad = 0
    for name, d in raw["datasets"].items():
        got = plain_files(RAW / name)
        ok = got == d["files"]
        bad += not ok
        print(f"{name}: {'ok' if ok else 'MISMATCH'} ({len(got)} files)", flush=True)
    return 1 if bad else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--verify", action="store_true")
    a = p.parse_args(argv)
    return verify() if a.verify else download()


if __name__ == "__main__":
    sys.exit(main())
