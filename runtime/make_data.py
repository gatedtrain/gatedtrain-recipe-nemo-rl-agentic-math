#!/usr/bin/env python
"""Export the pinned MATH dataset of area "agentic RL" to training rows and gate items (host step, CPU).

Input: `raw/hendrycks_math/<subject>/{train,test}-00000-of-00001.parquet` (fetch_data.py; RAW.json pins repo,
revision and SHA-256; columns problem, level, type, solution). Outputs:
- `data/shards/shard-0000{0..3}.jsonl` (500 rows each) and their concatenation `data/train.jsonl` (NeMo RL reads it in
  file order, `data.shuffle=false`). Row: `{row_id, input, messages, answer, subject, level, source}`; `messages` is
  `[{role: user, content: <problem>}, {role: assistant, content: <integer answer>}]` (ResponseDataset keeps all columns
  when `messages` exists, response_dataset.py:64-72; math_hf_data_processor reads the problem from messages[0] and the
  ground truth from messages[1], processors.py:561-563); `input` is the overlap payload (the problem text).
- `items-test/items.jsonl` (500 gate items) and `item_set_meta.json`. Item: `{item, text, problem, answer, subject,
  level, source}`; `text` is the problem text (overlap payload).
- `data/data_meta.json`: counts, the integer filter's yield per split, strata, overlap checks, file hashes.

Reference answer: the content of the last `\\boxed{...}` (balanced braces; `\\fbox{...}` and `\\boxed x` also
accepted) of the source `solution`. Integer rule (`integer_answer`, exact): strip whitespace; delete `$`, `\\!`,
`\\,`, `\\;`, `\\ ` and spaces; `{,}` becomes `,`; a comma-grouped number (`-?d{1,3}(,ddd)+`) loses its commas; the
result must then match `[+-]?\\d+` optionally followed by `.0...` (a zero fraction); the answer is `str(int(...))`.
Anything else (fractions, radicals, units, `\\text{}`, degrees, tuples, expressions) is excluded.

Selection (seeded, deterministic; `--seed`, default 20261115):
- Items first: the integer-answer problems of the `test` split, stratified by (subject, level): 500 allocated to the
  35 cells in proportion to their sizes (largest remainder, ties by cell name), a seeded sample in each cell
  (`random.Random("<seed>:items:<subject>:<level>")`), written in (subject, level, source index) order.
- Training rows: the integer-answer problems of the `train` split whose exact text and normalized text (lower case,
  `$` and all whitespace removed) differ from every `test`-split problem (all 5,000, not only the 500 items),
  stratified by subject: 2,000 allocated by largest remainder, a seeded sample per subject
  (`"<seed>:train:<subject>"`), then permuted once (`"<seed>:order:train"`).
- Overlap checks (0 required, refused otherwise): training rows vs items by exact and normalized text; duplicate
  texts inside the item set.

Usage (host): <venv with pyarrow>/bin/python -I make_data.py [--seed 20261115] [--tokenizer models/qwen3-1.7b]
  `--tokenizer` (needs transformers) adds prompt token counts (chat template, enable_thinking=False) to data_meta.
  Without it, the counts of the existing `data/data_meta.json` are kept when everything else in it is reproduced byte
  for byte (they are a function of the rows, the prompt file and the pinned tokenizer, all unchanged then); otherwise
  data_meta has no `prompt_tokens`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
DATA = HERE / "data"
ITEMS = HERE / "items-test"
PROMPT_FILE = HERE / "prompt_math_code.txt"
ROWS_PER_SHARD = 500
N_TRAIN = 2000
N_ITEMS = 500
_INT = re.compile(r"[+-]?\d+(?:\.0+)?")
_GROUPED = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+")


def sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def last_boxed(solution: str) -> str | None:
    """Content of the last \\boxed{...} / \\fbox{...} (balanced braces), or of `\\boxed x` up to a `$`."""
    best = None
    for m in re.finditer(r"\\(?:boxed|fbox)\s*", solution):
        i = m.end()
        if i < len(solution) and solution[i] == "{":
            depth, j = 0, i
            while j < len(solution):
                if solution[j] == "{":
                    depth += 1
                elif solution[j] == "}":
                    depth -= 1
                    if depth == 0:
                        best = solution[i + 1:j]
                        break
                j += 1
        elif m.group(0).startswith("\\boxed") and m.group(0).endswith(" "):
            best = solution[i:].split("$")[0].strip()
    return best


def integer_answer(ans: str | None) -> str | None:
    if ans is None:
        return None
    t = ans.strip()
    for s in ("$", "\\!", "\\,", "\\;", "\\ ", " "):
        t = t.replace(s, "")
    t = t.replace("{,}", ",")
    if _GROUPED.fullmatch(t):
        t = t.replace(",", "")
    if not _INT.fullmatch(t):
        return None
    return str(int(t.split(".")[0]))


def norm_text(s: str) -> str:
    return re.sub(r"\s+", "", s.replace("$", "")).lower()


def read_split(split: str) -> list[dict]:
    import pyarrow.parquet as pq

    rows = []
    for f in sorted((RAW / "hendrycks_math").glob(f"*/{split}-*.parquet")):
        subject = f.parent.name
        for i, r in enumerate(pq.read_table(f).to_pylist()):
            boxed = last_boxed(r["solution"])
            rows.append({"problem": r["problem"], "level": r["level"], "type": r["type"], "subject": subject,
                         "boxed": boxed, "answer": integer_answer(boxed),
                         "source": {"file": f"hendrycks_math/{subject}/{f.name}", "index": i}})
    return rows


def allocate(sizes: dict, total: int) -> dict:
    """Largest-remainder proportional allocation; ties broken by key."""
    n = sum(sizes.values())
    quota = {k: total * v / n for k, v in sizes.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    rest = total - sum(alloc.values())
    for k in sorted(sizes, key=lambda k: (-(quota[k] - alloc[k]), str(k)))[:rest]:
        alloc[k] += 1
    for k in alloc:
        if alloc[k] > sizes[k]:
            raise SystemExit(f"stratum {k}: {alloc[k]} allocated but only {sizes[k]} rows")
    return alloc


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


def prompt_tokens(tokenizer_dir: Path, problems: list[str]) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    tmpl = PROMPT_FILE.read_text(encoding="utf-8")
    lens = []
    for p in problems:
        text = tok.apply_chat_template([{"role": "user", "content": tmpl.format(p)}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        lens.append(len(tok(text, add_special_tokens=False)["input_ids"]))
    lens.sort()
    return {"n": len(lens), "min": lens[0], "median": lens[len(lens) // 2], "p99": lens[int(0.99 * (len(lens) - 1))],
            "max": lens[-1], "template_ends_with_empty_think": text.endswith("<think>\n\n</think>\n\n")}


def render(meta: dict) -> bytes:
    return (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode("utf-8")


def kept_prompt_tokens(meta: dict, committed: bytes | None):
    """The `prompt_tokens` of the `committed` data_meta.json if, with them added, `meta` renders to exactly its bytes;
    else None."""
    try:
        old = json.loads(committed) if committed is not None else None
    except ValueError:
        return None
    if not isinstance(old, dict) or "prompt_tokens" not in old:
        return None
    return old["prompt_tokens"] if render({**meta, "prompt_tokens": old["prompt_tokens"]}) == committed else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seed", type=int, default=20261115)
    p.add_argument("--tokenizer", type=Path)
    a = p.parse_args(argv)
    raw_meta = json.loads((RAW / "RAW.json").read_text(encoding="utf-8"))
    src = raw_meta["datasets"]["hendrycks_math"]
    for rel, f in src["files"].items():
        if sha256_file(RAW / "hendrycks_math" / rel) != f["sha256"]:
            raise SystemExit(f"raw/hendrycks_math/{rel}: SHA-256 differs from RAW.json")
    try:
        committed = (DATA / "data_meta.json").read_bytes()
    except OSError:
        committed = None
    for d in (DATA, ITEMS):
        if d.exists():
            shutil.rmtree(d)

    test_all, train_all = read_split("test"), read_split("train")
    yield_ = {}
    for name, rows in (("test", test_all), ("train", train_all)):
        yield_[name] = {"rows": len(rows), "with_boxed": sum(r["boxed"] is not None for r in rows),
                        "integer": sum(r["answer"] is not None for r in rows),
                        "integer_by_subject": dict(sorted(Counter(r["subject"] for r in rows
                                                                  if r["answer"] is not None).items()))}

    # ---- items: stratified by (subject, level) over the integer-answer test problems
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in test_all:
        if r["answer"] is not None:
            cells[(r["subject"], r["level"])].append(r)
    alloc = allocate({k: len(v) for k, v in cells.items()}, N_ITEMS)
    picked = []
    for (subject, level), rows in sorted(cells.items()):
        k = alloc[(subject, level)]
        picked += random.Random(f"{a.seed}:items:{subject}:{level}").sample(rows, k)
    picked.sort(key=lambda r: (r["subject"], r["level"], r["source"]["index"]))
    items = [{"item": f"math-test-{k:04d}", "text": r["problem"], "problem": r["problem"], "answer": r["answer"],
              "subject": r["subject"], "level": r["level"], "source": r["source"]} for k, r in enumerate(picked)]

    # ---- training rows: integer-answer train problems, none equal (exact or normalized) to any test problem
    test_exact = {r["problem"] for r in test_all}
    test_norm = {norm_text(r["problem"]) for r in test_all}
    cand = [r for r in train_all if r["answer"] is not None]
    kept = [r for r in cand if r["problem"] not in test_exact and norm_text(r["problem"]) not in test_norm]
    by_subj: dict[str, list[dict]] = defaultdict(list)
    for r in kept:
        by_subj[r["subject"]].append(r)
    talloc = allocate({k: len(v) for k, v in by_subj.items()}, N_TRAIN)
    chosen = []
    for subject, rows in sorted(by_subj.items()):
        chosen += random.Random(f"{a.seed}:train:{subject}").sample(rows, talloc[subject])
    chosen.sort(key=lambda r: (r["subject"], r["source"]["index"]))
    random.Random(f"{a.seed}:order:train").shuffle(chosen)
    train = [{"row_id": f"math-train-{k:05d}", "input": r["problem"],
              "messages": [{"role": "user", "content": r["problem"]}, {"role": "assistant", "content": r["answer"]}],
              "answer": r["answer"], "subject": r["subject"], "level": r["level"], "source": r["source"]}
             for k, r in enumerate(chosen)]

    item_exact = {it["text"] for it in items}
    item_norm = {norm_text(it["text"]) for it in items}
    checks = {
        "train_vs_items_exact_overlap": sum(r["input"] in item_exact for r in train),
        "train_vs_items_normalized_overlap": sum(norm_text(r["input"]) in item_norm for r in train),
        "duplicate_item_texts": len(items) - len(item_exact),
        "train_candidates_dropped_as_test_duplicates": len(cand) - len(kept),
    }
    if checks["train_vs_items_exact_overlap"] or checks["train_vs_items_normalized_overlap"] \
            or checks["duplicate_item_texts"]:
        raise SystemExit(f"overlap checks failed: {checks}")

    (DATA / "shards").mkdir(parents=True)
    for s in range(0, len(train), ROWS_PER_SHARD):
        write_jsonl(DATA / "shards" / f"shard-{s // ROWS_PER_SHARD:05d}.jsonl", train[s:s + ROWS_PER_SHARD])
    with open(DATA / "train.jsonl", "wb") as f:
        for sh in sorted((DATA / "shards").glob("shard-*.jsonl")):
            f.write(sh.read_bytes())
    ITEMS.mkdir()
    write_jsonl(ITEMS / "items.jsonl", items)
    imeta = {"schema": "ptc_mm.agentic_items", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
             "env_id": "math_code", "n_items": len(items), "seed": a.seed, "split": "test",
             "selection": "integer-answer test problems; 500 by largest-remainder allocation over (subject, level); "
                          "random.Random('<seed>:items:<subject>:<level>').sample per cell",
             "strata": {f"{s}|{l}": n for (s, l), n in sorted(alloc.items())},
             "source": {"repo": src["repo"], "revision": src["revision"]}, "license": raw_meta["license"],
             "answer_rule": "integer_answer() of make_data.py (README section 0)",
             "items_sha256": sha256_file(ITEMS / "items.jsonl")}
    (ITEMS / "item_set_meta.json").write_text(json.dumps(imeta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    meta = {"schema": "ptc_mm.agentic_data", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
            "seed": a.seed, "source": {"repo": src["repo"], "revision": src["revision"]},
            "license": raw_meta["license"], "integer_filter_yield": yield_,
            "counts": {"train": len(train), "items-test": len(items),
                       "train_by_subject": dict(sorted(Counter(r["subject"] for r in train).items())),
                       "train_by_level": dict(sorted(Counter(r["level"] for r in train).items())),
                       "items_by_subject": dict(sorted(Counter(r["subject"] for r in items).items())),
                       "items_by_level": dict(sorted(Counter(r["level"] for r in items).items()))},
            "checks": checks,
            "files": {"data/train.jsonl": sha256_file(DATA / "train.jsonl"),
                      **{f"data/shards/{s.name}": sha256_file(s) for s in sorted((DATA / "shards").glob("*.jsonl"))},
                      "items-test/items.jsonl": imeta["items_sha256"]},
            "prompt_file_sha256": sha256_file(PROMPT_FILE)}
    if a.tokenizer:
        meta["prompt_tokens"] = {"train": prompt_tokens(a.tokenizer, [r["input"] for r in train]),
                                 "items-test": prompt_tokens(a.tokenizer, [it["text"] for it in items])}
    else:
        kept = kept_prompt_tokens(meta, committed)
        if kept is not None:
            meta["prompt_tokens"] = kept
            print("data_meta.json reproduced; kept its prompt_tokens (pass --tokenizer to recompute them)", flush=True)
    (DATA / "data_meta.json").write_bytes(render(meta))
    print(json.dumps({"counts": meta["counts"], "checks": checks, "yield": {k: {x: v[x] for x in ("rows", "integer")}
                                                                             for k, v in yield_.items()},
                      "prompt_tokens": meta.get("prompt_tokens")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
