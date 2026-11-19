#!/usr/bin/env python
"""Download the base model of area "agentic RL" (host step, network on, no GPU).

The model goes to `models/qwen3-1.7b/` as plain files (no symbolic links, MM-20). huggingface_hub's own download
metadata (`<local_dir>/.cache/huggingface/`) is moved out to `hf-meta/<name>/`. No token is used or read:
`token=False` on every call, `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, and `HF_HOME` points to the fresh `hf-home/` of this
directory; a token in the environment is removed from this process's environment before huggingface_hub is
imported (its value is never read). Xet transfer is disabled (plain HTTPS). Run it as `env -u HF_TOKEN ...` anyway.

Only the pinned repository at the pinned revision is fetched (the maintainers' decisions "Area 10
decisions", not published): `Qwen/Qwen3-1.7B` @ 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e, Apache-2.0, all 12 files, 4,079,450,110
bytes (README "Pins"). `models/MODELS.json` records repo, revision, license, every file's SHA-256 and bytes;
`--verify` re-hashes and compares. Adapted from area 7's fetch_models.py (copied, not imported).

Usage (host, any venv with huggingface_hub, e.g. a Python 3.13 venv):
  env -u HF_TOKEN <venv>/bin/python fetch_models.py            # download, write MODELS.json
  <venv>/bin/python fetch_models.py --verify                     # re-hash every file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"

# Revision resolved 2026-11-14 from the Hub API (README "Pins"): gated false, license apache-2.0.
QWEN3_FILES = [".gitattributes", "LICENSE", "README.md", "config.json", "generation_config.json", "merges.txt",
               "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
               "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"]
MODEL_SPECS = {
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", QWEN3_FILES, "apache-2.0",
                   4_079_450_110),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def plain_files(root: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise SystemExit(f"{p} is a symbolic link; the directory must be plain (MM-20)")
        if p.is_file():
            out[p.relative_to(root).as_posix()] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return out


def hub_env() -> None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH"):
        os.environ.pop(name, None)          # removed, never read
    os.environ["HF_HOME"] = str(HERE / "hf-home")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    (HERE / "hf-home").mkdir(parents=True, exist_ok=True)


def move_meta(dest: Path, name: str) -> None:
    meta = dest / ".cache"
    if meta.exists():
        target = HERE / "hf-meta" / name
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(meta), str(target))


def download() -> int:
    hub_env()
    from huggingface_hub import hf_hub_download

    record = {"schema": "ptc_mm.agentic_models", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
              "models": {}}
    for name, (repo, revision, files, lic, want_bytes) in MODEL_SPECS.items():
        dest = MODELS / name
        for f in files:
            hf_hub_download(repo, f, revision=revision, local_dir=dest, token=False)
        move_meta(dest, name)
        got = plain_files(dest)
        if sorted(got) != sorted(files):
            raise SystemExit(f"{dest} holds {sorted(got)}, not {sorted(files)}")
        total = sum(f["bytes"] for f in got.values())
        if total != want_bytes:
            raise SystemExit(f"{dest}: {total} bytes, the Hub listing said {want_bytes}")
        record["models"][name] = {"repo": repo, "revision": revision, "license": lic, "files": got, "bytes": total}
        print(f"{name}: {len(got)} files, {total} bytes", flush=True)
    (MODELS / "MODELS.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def verify() -> int:
    bad = 0
    record = json.loads((MODELS / "MODELS.json").read_text(encoding="utf-8"))
    for name, m in record["models"].items():
        got = plain_files(MODELS / name)
        ok = got == m["files"]
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
