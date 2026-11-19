#!/usr/bin/env python
"""Generation half of area "agentic RL"'s evaluator `agentic.math_code_items@0.1`: one multi-turn trajectory per
(item, evaluation seed), with the same turn protocol and the same executor sidecar as training.

GPU run inside the NeMo RL image with its vLLM worker venv
(`/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python`, vLLM 0.29.0, as areas 6
and 7), paired with an executor sidecar (`executor/run-executor.sh serve`) through `/spool`. Generated code never
runs in this container. Per item:
- prompt exactly as training: the deployment prompt file `PROMPT_FILE` (`prompt_math_code_oneshot.txt`, the
  `text_oneshot` variant of probe_tool_use.py byte for byte once formatted, chosen after probe-tool-2, README 0.11)
  `.format(problem)` as one user message through the base tokenizer's chat template with
  `add_generation_prompt=True` and `enable_thinking=False`, then tokenized without special tokens (NeMo RL's
  math_hf_data_processor, processors.py:574-590);
- turns as NeMo RL's synchronous multi-turn rollout (rollouts.py:1084-1243): vLLM offline `LLM.generate` on the
  running token ids (`prompt_token_ids`), greedy, at most 1,024 new tokens capped to `max_model_len` (4,096) minus
  the input length (vllm_worker.py:92-102), stop `</code>` with `include_stop_str_in_output=True`
  (vllm_worker.py:891), no stop token ids beyond the model's EOS (grpo_math_1B.yaml `stop_token_ids: null`); the
  new token ids decoded with `skip_special_tokens=True` (rollouts.py:699: the decision is on the decoded token ids,
  not on vLLM's stop-truncated text, in both places); `agentic_core.MathCodeCore` decides (the same
  code as the training environment); the observation tokenized without special tokens and appended as raw tokens,
  cut when it would reach `max_model_len` (then the trajectory ends, `overflow`), as rollouts.py:1172-1196;
- all items advance turn by turn in one batch; the executor gets one batch of jobs per turn.

A cut (`--adapter`, a LoRA cut of ptc_agentic_hooks.py) is loaded as a vLLM `LoRARequest` on the base model (the
base-plus-LoRA path NeMo RL's native refit uses for rollouts); its `cut.json` file hashes and its `base_ref.json`
against the base directory are checked first, and its `sampler.json` must equal this file's SAMPLER in
`SAMPLER_MUST_MATCH` (prompt file hash, turn budget, context, turns, stop string; a cut of P1 p1-a2, trained with
prompt_math_code.txt and 512 / 2,048, is refused). A full-weight cut (memory rung c) is passed as `--model`.

Executor faults (`exec_protocol.ExecutorUnavailable`) are retried once per turn; if the retry fails too, the
affected items end with error class `sandbox_failure` (MM-4, evaluator's fault, unpaired) and the run continues.

Deployment sampler (the "SAMPLER" block, also written into every cut): greedy, 1 sample per (item, seed).

Outputs (`--out`): `gen/<item>__s<seed>.json` `{item, seed, repeat, prompt_sha256, prompt_tokens, turns: [{text,
n_tokens, finish_reason, code, tool_status, observation, obs_tokens}], end_reason, final_text, model_tokens,
env_tokens, total_tokens, error}`, `gen_meta.json` (model and adapter file hashes, sampler, items hash and selected
item ids, prompt and code hashes, executor identity, versions, timing).

Usage (container, vLLM worker venv):
  gen_agentic.py --model <area>/models/qwen3-1.7b [--adapter /inputs/cut] --items <area>/items-test --out /out
                 --socket /spool/mathexec.sock --eval-seeds 11 [--limit 20]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE / "executor"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import agentic_core  # noqa: E402
import exec_protocol  # noqa: E402

# Deployment prompt (README 0.11): probe_tool_use.TEXT_ONESHOT in NeMo RL's prompt_file form (`{}` = problem,
# literal braces doubled); prompt_math_code.txt stays as the record of P1 p1-a2 and of the probes' text_base arm.
PROMPT_FILE = HERE / "prompt_math_code_oneshot.txt"

# ---- SAMPLER (begin) -------------------------------------------------------------------------------------------
SAMPLER = {
    "strategy": "greedy",
    "temperature": 0.0,
    "top_p": 1.0,
    "samples_per_seed": 1,
    "max_new_tokens_per_turn": 1024,
    "max_model_len": 4096,
    "max_turns": agentic_core.MAX_TURNS,
    "stop": [agentic_core.STOP],
    "include_stop_str_in_output": True,
    "chat_template": "base tokenizer chat template, add_generation_prompt=True, enable_thinking=False",
    "messages": "[{role: user, content: <prompt_file>.format(problem)}]",
    "prompt_file": PROMPT_FILE.name,
    "prompt_file_sha256": "sha256:" + hashlib.sha256(PROMPT_FILE.read_bytes()).hexdigest(),
    "observation": "agentic_core.observation(), raw tokens without special tokens, obs_max_chars 512",
    "executor": "executor/executor_server.py sidecar, stateless replay, limits of executor_server.DEFAULT_LIMITS",
}
# ---- SAMPLER (end) ---------------------------------------------------------------------------------------------
# What a LoRA cut's sampler.json (the SAMPLER of its training run) must share with this evaluator: a cut trained with
# another prompt file, turn budget or stop string is refused (check_cut).
SAMPLER_MUST_MATCH = ("prompt_file_sha256", "max_new_tokens_per_turn", "max_model_len", "max_turns", "stop")
LORA_RANK = 32


def sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def select_items(items: list[dict], limit: int | None) -> list[dict]:
    """`--limit N`: N items evenly spaced over the file (the file is sorted by subject and level), so a smoke arm
    covers every subject."""
    if not limit or limit >= len(items):
        return items
    return [items[(i * len(items)) // limit] for i in range(limit)]


def check_cut(cut_dir: Path, base_dir: Path) -> dict:
    cut = json.loads((cut_dir / "cut.json").read_text(encoding="utf-8"))
    bad = [f for f, h in cut["files"].items() if sha256_file(cut_dir / f) != h]
    extra = sorted({f.name for f in cut_dir.iterdir()} - set(cut["files"]) - {"cut.json"})
    if bad or extra:
        raise SystemExit(f"cut {cut_dir}: hash mismatch {bad}, unlisted files {extra}")
    ref = json.loads((cut_dir / "base_ref.json").read_text(encoding="utf-8"))
    wrong = [f for f, d in ref["files"].items() if sha256_file(base_dir / f) != d["sha256"]]
    if wrong:
        raise SystemExit(f"base {base_dir} differs from the cut's base_ref.json in {wrong}")
    check_sampler(cut_dir)
    return cut


def check_sampler(cut_dir: Path) -> None:
    trained = json.loads((cut_dir / "sampler.json").read_text(encoding="utf-8"))
    differ = [k for k in SAMPLER_MUST_MATCH if trained.get(k) != SAMPLER[k]]
    if differ:
        raise SystemExit(f"cut {cut_dir} was trained with another prompt or turn budget: sampler.json differs in "
                         f"{differ} (cut {[trained.get(k) for k in differ]}, now {[SAMPLER[k] for k in differ]})")


def wait_for_executor(sock: str, deadline_s: float = 900.0) -> dict:
    t0, last = time.time(), None
    while time.time() - t0 < deadline_s:
        try:
            return exec_protocol.ping(sock, timeout=10.0)
        except exec_protocol.ExecutorUnavailable as e:
            last = e
            time.sleep(5)
    raise SystemExit(f"executor at {sock} did not answer within {deadline_s:.0f} s: {last}")


def run_trajectories(states: list[dict], generate, tokenize, core: agentic_core.MathCodeCore,
                     max_model_len: int) -> None:
    """The multi-turn loop (rollouts.py:1084-1243 semantics), framework-free so the CPU tests can drive it.
    `generate(list of token-id lists, list of max_tokens) -> list of (new token ids, text, finish_reason)`;
    `tokenize(text) -> token ids`."""
    for turn in range(1, core.max_turns + 1):
        active = [s for s in states if not s["done"]]
        if not active:
            return
        max_tokens = [min(SAMPLER["max_new_tokens_per_turn"], max_model_len - len(s["ids"])) for s in active]
        outs = generate([s["ids"] for s in active], max_tokens)
        input_lens = [len(s["ids"]) for s in active]
        for s, (ids, text, finish) in zip(active, outs):
            s["ids"] = s["ids"] + list(ids)
            s["turns"].append({"text": text, "n_tokens": len(ids), "finish_reason": finish})
        samples = [{"turn": turn, "text": s["turns"][-1]["text"], "state": {"cells": s["cells"]}} for s in active]
        try:
            res = core.step(samples)
        except exec_protocol.ExecutorUnavailable:
            try:
                res = core.step(samples)
            except exec_protocol.ExecutorUnavailable as e:
                for s in active:
                    s["done"] = True
                    s["error"] = {"class": "sandbox_failure", "message": str(e)[:500]}
                continue
        for s, r, n_in in zip(active, res, input_lens):
            t = s["turns"][-1]
            t.update({"code": r["code"], "tool_status": r["tool_status"], "observation": r["observation"]})
            obs_ids = list(tokenize(r["observation"])) if r["observation"] else []
            n_gen = t["n_tokens"]
            overflow = len(obs_ids) + n_gen + n_in >= max_model_len
            if overflow:
                obs_ids = obs_ids[:max(0, max_model_len - (n_gen + n_in))]
            t["obs_tokens"] = len(obs_ids)
            s["ids"] = s["ids"] + obs_ids
            s["cells"] = r["state"]["cells"]
            if r["terminated"]:
                s["done"], s["end_reason"] = True, r["end_reason"]
            elif overflow:
                s["done"], s["end_reason"] = True, "overflow"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", type=Path, required=True, help="base model directory (or a full-weight cut)")
    p.add_argument("--adapter", type=Path, help="LoRA cut directory (cut.json, adapter_model.safetensors)")
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--socket", required=True)
    p.add_argument("--eval-seeds", default="11")
    p.add_argument("--limit", type=int)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    a = p.parse_args(argv)
    t0 = time.time()
    out = a.out.resolve()
    if (out / "gen").exists() and any((out / "gen").iterdir()):
        raise SystemExit(f"{out}/gen exists and is not empty")
    (out / "gen").mkdir(parents=True, exist_ok=True)
    all_items = [json.loads(x) for x in open(a.items / "items.jsonl", encoding="utf-8")]
    items = select_items(all_items, a.limit)
    seeds = [int(s) for s in a.eval_seeds.split(",")]
    cut = None
    if a.adapter:
        cut = check_cut(a.adapter, a.model)
        if cut.get("kind") != "lora_adapter":
            raise SystemExit(f"{a.adapter} is a {cut.get('kind')} cut; pass a full-weight cut as --model")
    elif (a.model / "cut.json").exists():
        cut = json.loads((a.model / "cut.json").read_text(encoding="utf-8"))
        bad = [f for f, h in cut["files"].items() if sha256_file(a.model / f) != h]
        if bad:
            raise SystemExit(f"cut {a.model}: hash mismatch {bad}")
    executor = wait_for_executor(a.socket)
    core = agentic_core.MathCodeCore(execute=lambda jobs: exec_protocol.call_executor(a.socket, jobs, 900.0),
                                     verifier=None)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import vllm

    tok = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
    tmpl = PROMPT_FILE.read_text(encoding="utf-8")
    llm_kwargs = {"model": str(a.model), "dtype": "bfloat16", "seed": 0, "max_model_len": SAMPLER["max_model_len"],
                  "gpu_memory_utilization": a.gpu_memory_utilization}
    lora_req = None
    if a.adapter:
        from vllm.lora.request import LoRARequest
        llm_kwargs.update({"enable_lora": True, "max_lora_rank": LORA_RANK, "max_loras": 1})
        lora_req = LoRARequest("cut", 1, str(a.adapter))
    llm = LLM(**llm_kwargs)

    def tokenize(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    for k, seed in enumerate(seeds):
        def generate(id_lists, max_tokens):
            sps = [SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(m), stop=[agentic_core.STOP],
                                  include_stop_str_in_output=True, seed=seed) for m in max_tokens]
            outs = llm.generate([{"prompt_token_ids": ids} for ids in id_lists], sps, lora_request=lora_req,
                                use_tqdm=False)
            res = []
            for o in outs:
                ids = list(o.outputs[0].token_ids)
                res.append((ids, tok.decode(ids, skip_special_tokens=True), o.outputs[0].finish_reason))
            return res

        states = []
        for it in items:
            prompt = tok.apply_chat_template([{"role": "user", "content": tmpl.format(it["problem"])}],
                                             tokenize=False, add_generation_prompt=True, add_special_tokens=False,
                                             enable_thinking=False)
            ids = tokenize(prompt)
            states.append({"item": it["item"], "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                           "prompt_tokens": len(ids), "ids": ids, "turns": [], "cells": [], "done": False,
                           "end_reason": None, "error": None})
        run_trajectories(states, generate, tokenize, core, SAMPLER["max_model_len"])
        for s in states:
            model_t = sum(t["n_tokens"] for t in s["turns"])
            env_t = sum(t.get("obs_tokens", 0) for t in s["turns"])
            doc = {"item": s["item"], "seed": seed, "repeat": k, "prompt_sha256": s["prompt_sha256"],
                   "prompt_tokens": s["prompt_tokens"], "turns": s["turns"], "end_reason": s["end_reason"],
                   "final_text": s["turns"][-1]["text"] if s["turns"] else None, "model_tokens": model_t,
                   "env_tokens": env_t, "total_tokens": len(s["ids"]), "error": s["error"]}
            (out / "gen" / f"{s['item']}__s{seed}.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n",
                                                                   encoding="utf-8")
    import transformers
    model_files = {f.name: sha256_file(f) for f in sorted(a.model.iterdir())
                   if f.is_file() and (f.suffix in (".json", ".safetensors", ".txt", ".jinja"))}
    meta = {"backend": "vllm", "env_id": "math_code", "model_dir": str(a.model.resolve()), "model_files": model_files,
            "adapter": None if not a.adapter else {
                "dir": str(a.adapter.resolve()), "global_step": cut["global_step"],
                "cut_json_sha256": sha256_file(a.adapter / "cut.json"),
                "files": {f: h for f, h in cut["files"].items() if f in cut["deployable"]}},
            "cut": None if a.adapter or cut is None else {"global_step": cut["global_step"],
                                                          "cut_json_sha256": sha256_file(a.model / "cut.json")},
            "sampler": SAMPLER, "eval_seeds": seeds, "items": len(items), "item_ids": [it["item"] for it in items],
            "items_sha256": sha256_file(a.items / "items.jsonl"), "limit": a.limit,
            "prompt_file": PROMPT_FILE.name, "prompt_file_sha256": sha256_file(PROMPT_FILE),
            "code_sha256": {f: sha256_file(HERE / f) for f in ("gen_agentic.py", "agentic_core.py",
                                                               "executor/exec_protocol.py")},
            "executor": executor,
            "versions": {"vllm": vllm.__version__, "transformers": transformers.__version__, "dtype": "bfloat16",
                         "python": sys.version.split()[0]},
            "wall_seconds": round(time.time() - t0, 1),
            "seconds_per_trajectory": round((time.time() - t0) / max(1, len(items) * len(seeds)), 3)}
    (out / "gen_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("items", "wall_seconds", "seconds_per_trajectory", "versions")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
