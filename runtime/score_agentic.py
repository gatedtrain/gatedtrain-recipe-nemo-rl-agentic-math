#!/usr/bin/env python
"""Scoring half of area "agentic RL"'s evaluator `agentic.math_code_items@0.1` (CPU, standard library only, no model,
runs no generated code).

Deciding metric `exact_match` (higher is better, binary, [0, 1]) with this script's own answer parser - not
math_verify, which gives the training reward (math_code_env.py) - so the metric is T3 under R15-1 (05 section 6.1:
an answer parser is a tool) with a construct-alignment disclosure (both reward final-answer correctness) and a
shared-annotation disclosure (R16-1: the references are MATH `\\boxed{}` answers written by the same process as the
training answers).

Parser (`parse_answer`, applied to the final assistant turn's text only, never to tool results):
1. The candidate is the content of the last `\\boxed{...}` or `\\fbox{...}` (balanced braces); none: no answer.
2. Normalize: delete `$`, `\\!`, `\\,`, `\\;`, `\\:`, `\\ `, `~` and whitespace; unwrap one `\\text{...}` /
   `\\mathrm{...}` / `\\mathbf{...}` around the whole candidate; `{,}` becomes `,`; drop one trailing `.`; a
   comma-grouped number (`-?d{1,3}(,ddd)+`) loses its commas; a leading `+` is dropped.
3. The candidate must then be `-?\\d+` optionally followed by `.0...`; its value as an integer is the answer;
   anything else (fractions, expressions, units, `x=5`) is no answer.
4. Score 1 when the answer equals the item's reference integer, else 0.

Outcomes (MM-4; score 0 unless `correct`, all the policy's): `correct`, `wrong_answer`, `no_answer`, `max_turns`
(no answer and the trajectory ended at the turn limit), `overflow` (no answer and the context was exhausted). Code
that raised or timed out is part of the transcript, not an error. Errors (evaluator's fault, unpaired):
`missing_output` (no generation for a selected (item, seed)), `sandbox_failure` (the generator recorded an executor
fault for the item), `scorer_error` (a reference that is not an integer, or a malformed generation file).

Sub-metrics (finite numbers, MM-1): `turns_used`, `tool_calls` (executed code cells), `tool_errors` (cells with
status error, syntax_error, timeout or crash), `tool_timeouts`, `answer_found`, `truncated` (a turn stopped at the
token budget or the trajectory overflowed), `model_tokens`, `env_tokens`.

Outputs (`--out`): `outputs/<item>__r<k>.json`, `scores.jsonl`, `summary.json` (statistics, MM-38 degeneracy check,
outcome counts), `env-runtime.json`, a byte copy of `gen_meta.json`, and `arm-exact_match/` with MM-1 `items.jsonl`,
MM-2 `eval_meta.json` and hard-linked `outputs/`.

Usage (CPU, any Python >= 3.10, e.g. `/opt/nemo_rl_venv/bin/python -I` in the image):
  score_agentic.py --gen DIR --items DIR --out DIR [--eval-seeds 11] [--limit N] [--model-hash H]
                   [--base-model-hash H] [--item-set-hash H]
  score_agentic.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EVALUATOR_ID = "agentic.math_code_items@0.1"
ADAPTER = "nemo_rl_agentic"
SCORER_VERSION = "math_int_boxed_v1"
_BOX = re.compile(r"\\(?:boxed|fbox)\s*\{")
_WRAP = re.compile(r"^\\(?:text|mathrm|mathbf|textbf)\{(.*)\}$", re.S)
_GROUPED = re.compile(r"-?\d{1,3}(?:,\d{3})+")
_INT = re.compile(r"-?\d+(?:\.0+)?")
TOOL_ERRORS = ("error", "syntax_error", "timeout", "crash")


def sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(Path(p).read_bytes())


def canonical(doc) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def write_json(path: Path, doc) -> None:
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def last_boxed(text: str) -> str | None:
    found = None
    for m in _BOX.finditer(text):
        depth, j = 1, m.end()
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            found = text[m.end():j - 1]
    return found


def normalize_int(cand: str) -> int | None:
    t = cand
    for s in ("$", "\\!", "\\,", "\\;", "\\:", "\\ ", "~"):
        t = t.replace(s, "")
    t = re.sub(r"\s+", "", t)
    m = _WRAP.match(t)
    if m:
        t = m.group(1)
    t = t.replace("{,}", ",")
    if t.endswith("."):
        t = t[:-1]
    if t.startswith("+"):
        t = t[1:]
    if _GROUPED.fullmatch(t):
        t = t.replace(",", "")
    if not _INT.fullmatch(t):
        return None
    return int(t.split(".")[0])


def parse_answer(text: str | None) -> int | None:
    if not text:
        return None
    cand = last_boxed(text)
    return None if cand is None else normalize_int(cand)


def reference_int(ref: str) -> int | None:
    return int(ref) if isinstance(ref, str) and re.fullmatch(r"-?\d+", ref) else None


def select_items(items: list[dict], limit: int | None) -> list[dict]:
    """Same rule as gen_agentic.select_items (copied: the scorer imports nothing of the generator)."""
    if not limit or limit >= len(items):
        return items
    return [items[(i * len(items)) // limit] for i in range(limit)]


def stats(vals) -> dict:
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if not n:
        return {"n": 0}
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else None
    return {"n": n, "mean": mean, "sd": sd}


def score_doc(doc: dict, ref: int) -> tuple[float, str, dict, int | None]:
    turns = doc["turns"]
    answer = parse_answer(doc.get("final_text"))
    statuses = [t.get("tool_status") for t in turns if t.get("tool_status") is not None]
    truncated = int(doc.get("end_reason") == "overflow" or any(t.get("finish_reason") == "length" for t in turns))
    sub = {"turns_used": len(turns), "tool_calls": len(statuses),
           "tool_errors": sum(s in TOOL_ERRORS for s in statuses),
           "tool_timeouts": sum(s == "timeout" for s in statuses),
           "answer_found": int(answer is not None), "truncated": truncated,
           "model_tokens": int(doc["model_tokens"]), "env_tokens": int(doc["env_tokens"])}
    if answer is not None:
        outcome = "correct" if answer == ref else "wrong_answer"
    elif doc.get("end_reason") in ("max_turns", "overflow"):
        outcome = doc["end_reason"]
    else:
        outcome = "no_answer"
    return float(outcome == "correct"), outcome, sub, answer


def self_test() -> int:
    cases = [("so \\boxed{42}", 42), ("\\boxed{ -7 }", -7), ("$\\boxed{1,000}$", 1000), ("\\boxed{1{,}234}", 1234),
             ("\\boxed{\\text{12}}", 12), ("\\boxed{5.}", 5), ("\\boxed{5.00}", 5), ("\\boxed{+3}", 3),
             ("\\boxed{3} then \\boxed{4}", 4), ("\\boxed{\\frac{1}{2}}", None), ("\\boxed{x=5}", None),
             ("the answer is 5", None), ("\\boxed{5\\text{ cm}}", None), ("\\boxed{2^{3}}", None),
             ("\\boxed{12", None), ("\\fbox{9}", 9), ("\\boxed{0}", 0), ("\\boxed{007}", 7), ("", None),
             ("<code>print(1)</code><result>\n\\boxed{3}\n</result>\\boxed{8}", 8), ("\\boxed{\\!1\\,000}", 1000)]
    bad = [(t, parse_answer(t), w) for t, w in cases if parse_answer(t) != w]
    refs = [("12", 12), ("-3", -3), ("1/2", None), ("", None)]
    bad += [(r, reference_int(r), w) for r, w in refs if reference_int(r) != w]
    doc = {"turns": [{"text": "<code>x</code>", "tool_status": "error", "finish_reason": "stop", "n_tokens": 5},
                     {"text": "\\boxed{4}", "tool_status": None, "finish_reason": "stop", "n_tokens": 3}],
           "final_text": "\\boxed{4}", "end_reason": "final", "model_tokens": 8, "env_tokens": 9}
    got = score_doc(doc, 4)
    if got[:2] != (1.0, "correct") or got[2]["tool_errors"] != 1 or got[2]["tool_calls"] != 1:
        bad.append(("score_doc correct", got, None))
    doc2 = {**doc, "final_text": "<code>y</code>", "end_reason": "max_turns"}
    if score_doc(doc2, 4)[:2] != (0.0, "max_turns"):
        bad.append(("score_doc max_turns", score_doc(doc2, 4), None))
    print("self-test", "ok" if not bad else f"FAILED {bad}")
    return 1 if bad else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gen", type=Path, help="directory with gen/ and gen_meta.json (gen_agentic.py)")
    p.add_argument("--items", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--eval-seeds", default="11")
    p.add_argument("--limit", type=int)
    p.add_argument("--evaluator-id", default=EVALUATOR_ID)
    p.add_argument("--model-hash")
    p.add_argument("--base-model-hash")
    p.add_argument("--item-set-hash")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.gen and a.items and a.out):
        p.error("--gen, --items and --out are required")
    t0 = time.time()
    out = a.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    (out / "outputs").mkdir(parents=True, exist_ok=True)
    gen_meta = json.loads((a.gen / "gen_meta.json").read_text(encoding="utf-8"))
    items = select_items([json.loads(x) for x in open(a.items / "items.jsonl", encoding="utf-8")], a.limit)
    seeds = [int(s) for s in a.eval_seeds.split(",")]
    if gen_meta["eval_seeds"] != seeds:
        raise SystemExit(f"gen_meta eval_seeds {gen_meta['eval_seeds']} != --eval-seeds {seeds}")
    if gen_meta["items_sha256"] != sha256_file(a.items / "items.jsonl"):
        raise SystemExit("gen_meta.json names another items.jsonl")
    if gen_meta.get("env_id") != "math_code":
        raise SystemExit(f"gen_meta env_id {gen_meta.get('env_id')!r} is not this evaluator's env math_code")
    if gen_meta["item_ids"] != [it["item"] for it in items]:
        raise SystemExit("gen_meta item_ids differ from this scorer's selection (--limit must match the generator's)")

    rows = []
    for it in items:
        ref = reference_int(it["answer"])
        for k, s in enumerate(seeds):
            gp = a.gen / "gen" / f"{it['item']}__s{s}.json"
            row = {"item": it["item"], "repeat": k, "seed": s, "error": None, "score": None, "sub_metrics": None,
                   "outcome": None}
            doc, answer = None, None
            if not gp.exists():
                row["error"] = {"class": "missing_output", "message": "no generation for this (item, seed)"}
            elif ref is None:
                row["error"] = {"class": "scorer_error", "message": f"reference {it['answer']!r} is not an integer"}
            else:
                try:
                    doc = json.loads(gp.read_text(encoding="utf-8"))
                    if doc.get("error"):
                        row["error"] = {"class": doc["error"]["class"], "message": doc["error"].get("message")}
                    else:
                        row["score"], row["outcome"], row["sub_metrics"], answer = score_doc(doc, ref)
                except (ValueError, KeyError, TypeError) as e:
                    row["error"] = {"class": "scorer_error", "message": f"{type(e).__name__}: {e}"}
            rel = f"outputs/{it['item']}__r{k}.json"
            write_json(out / rel, {"item": it["item"], "repeat": k, "seed": s, "problem": it["problem"],
                                   "reference": it["answer"], "response": doc.get("final_text") if doc else None,
                                   "transcript": doc.get("turns") if doc else None,
                                   "end_reason": doc.get("end_reason") if doc else None, "parsed_answer": answer,
                                   "score": row["score"], "outcome": row["outcome"],
                                   "sub_metrics": row["sub_metrics"], "error": row["error"]})
            row["output_ref"] = {"path": rel, "sha256": sha256_file(out / rel)}
            rows.append(row)
    write_jsonl(out / "scores.jsonl", rows)

    code_sha = {"score_agentic.py": sha256_file(Path(__file__).resolve())}
    scorer_code = sha256_bytes(canonical(code_sha))
    sampler = {**gen_meta["sampler"], "backend": gen_meta["backend"],
               "prompt_file_sha256": gen_meta["prompt_file_sha256"],
               "executor_code_sha256": (gen_meta.get("executor") or {}).get("code_sha256")}
    sampler_hash = sha256_bytes(canonical(sampler))
    v = gen_meta.get("versions") or {}
    arm = out / "arm-exact_match"
    arm.mkdir()
    lines = []
    for r in rows:
        link_or_copy(out / r["output_ref"]["path"], arm / r["output_ref"]["path"])
        lines.append({"item": r["item"], "repeat": r["repeat"], "seed": r["seed"], "score": r["score"],
                      "error": r["error"], "sub_metrics": r["sub_metrics"], "output_ref": r["output_ref"]})
    write_jsonl(arm / "items.jsonl", lines)
    write_json(arm / "eval_meta.json", {
        "schema": "gatedtrain.item_eval", "schema_version": "1.0", "evaluator_id": a.evaluator_id,
        "adapter": ADAPTER, "metric": "exact_match", "direction": "higher_is_better",
        "score_range": [0, 1], "checkpoint": {"model_hash": a.model_hash, "base_model_hash": a.base_model_hash},
        "normalizer_sha256": None, "env_version_sha256": None, "item_set_hash": a.item_set_hash,
        "eval_seeds": seeds, "sampler": sampler, "sampler_hash": sampler_hash,
        "scorer": {"name": "math_integer_boxed_own_parser",
                   "version": f"{SCORER_VERSION}; python {sys.version.split()[0]}",
                   "weights_sha256": None, "code_sha256": scorer_code},
        "framework": {"name": "vllm", "version": v.get("vllm")},
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})

    ok = [r for r in rows if not r["error"]]
    st = stats([r["score"] for r in ok])
    mean = st.get("mean")
    outcomes: dict[str, int] = {}
    for r in ok:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1

    def share(key):
        return sum(r["sub_metrics"][key] > 0 for r in ok) / len(ok) if ok else None

    def mean_of(key):
        return sum(r["sub_metrics"][key] for r in ok) / len(ok) if ok else None

    summary = {"items": len(items), "repeats": len(seeds), "responses": len(rows), "errors": len(rows) - len(ok),
               "error_classes": sorted({r["error"]["class"] for r in rows if r["error"]}), "exact_match": st,
               "outcomes": outcomes, "answer_found_share": share("answer_found"),
               "used_tool_share": share("tool_calls"),
               "truncated_share": share("truncated"), "mean_turns": mean_of("turns_used"),
               "mean_tool_calls": mean_of("tool_calls"), "mean_tool_errors": mean_of("tool_errors"),
               "mean_model_tokens": mean_of("model_tokens"), "mean_env_tokens": mean_of("env_tokens"),
               "degeneracy_mm38": {"exact_match": {"pass": mean is not None and 0 < mean < 1, "binary": True}},
               "wall_seconds": round(time.time() - t0, 1), "sampler_hash": sampler_hash,
               "gen_backend": gen_meta["backend"], "model_dir": gen_meta["model_dir"],
               "adapter": gen_meta.get("adapter"),
               "missing_hashes": [k for k, v_ in (("model_hash", a.model_hash), ("item_set_hash", a.item_set_hash))
                                  if v_ is None]}
    write_json(out / "env-runtime.json", {
        "python": sys.version.split()[0], "executable": sys.executable, "code_sha256": code_sha,
        "scorer_code_sha256": scorer_code, "items_sha256": sha256_file(a.items / "items.jsonl"),
        "gen_meta_sha256": sha256_file(a.gen / "gen_meta.json"), "uid": os.getuid(), "gid": os.getgid()})
    write_json(out / "summary.json", summary)
    shutil.copyfile(a.gen / "gen_meta.json", out / "gen_meta.json")
    print(json.dumps({k: summary[k] for k in ("items", "errors", "exact_match", "outcomes", "answer_found_share",
                                              "used_tool_share", "degeneracy_mm38")}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
