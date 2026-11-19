#!/usr/bin/env python
"""Tool-use probe of area "agentic RL": how often does base Qwen3-1.7B call the Python tool, under which prompt or
protocol, before any training (GPU, vLLM offline, paired executor sidecar; nothing is trained).

Why: in P1 run p1-a2 (README 0.10) `<code>` appeared in 6, 2 and 1 of 128 replies per step and only one tool call
ran; GRPO cannot reinforce a behaviour the policy almost never samples. This script measures, per variant, the
share of trajectories with a tool call and the answer quality, on training problems only (never the held-out items).

Variants (`VARIANTS`; prompt texts below):
- `text_base`     the training prompt (prompt_math_code.txt), `<code>`/`<result>` protocol (control).
- `text_oneshot`  the same protocol, a stronger instruction ("compute with code, not by hand") and one worked
                  example of a full `<code>` -> `<result>` -> `\\boxed` exchange in the user turn.
- `text_system`   the training user prompt plus a system prompt with the tool rules and a code-first instruction.
- `native`        Qwen3's own tool calling: `tools=[python]` through the chat template (its system block "# Tools
                  ... <tool_call>{json}</tool_call>"), replies `<tool_call>{"name": "python", "arguments": {"code":
                  ...}}</tool_call>` ending at EOS (`<|im_end|>`, no stop string), observations rendered as the chat
                  template renders a `tool` message: `\\n<|im_start|>user\\n<tool_response>\\n...\\n</tool_response>
                  <|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n` appended as raw tokens.
- `native_fewshot` `native` plus one worked exchange as earlier chat turns (user, assistant with tool_calls, tool,
                  assistant with the boxed answer) before the problem.

Turn loop: as gen_agentic.run_trajectories and NeMo RL's synchronous multi-turn rollout (rollouts.py:1084-1243):
generation on the running token ids, at most `--max-new-tokens` per turn capped to `--max-model-len` minus the input,
new tokens decoded with skip_special_tokens=True, the decision on the decoded text, the observation tokenized without
adding special tokens (special-token strings in it such as `<|im_start|>` map to their ids, as NeMo RL's
`tokenizer(obs, add_special_tokens=False)` does, rollouts.py:1170-1176) and appended, cut at the context limit
(`overflow`). Text variants decide with `agentic_core.MathCodeCore` itself (the training environment's code). Native
variants use `NativeCore` (same executor, same stateless replay, same observation text inside `<tool_response>`;
several tool calls in one turn run in order, each replaying the earlier ones).

Samplers: `sampled` = the training sampler (temperature 1.0, top_p 1.0, no top_k; NeMo RL grpo_math_1B.yaml
policy.generation) with `--samples` (8) samples per problem, and `greedy` (1 per problem, the evaluator's sampler).

Metrics per (variant, sampler) in `summary.json` and printed: trajectories, `tool_share` (>= 1 tool call attempted),
`executed_share` (>= 1 call run in the executor), `tool_calls_mean`, `turns_mean`, `boxed_share` (final turn holds
`\\boxed`), `answer_share` (score_agentic.parse_answer finds an integer), `exact_match` (the evaluator's parser
against the row's integer answer: the deciding metric's construct), `exact_match_tool` / `exact_match_no_tool`,
`truncated_share` (a turn hit the token budget), `fake_result_share` (the model wrote its own `<result>` /
`<tool_response>`), end reasons, executor statuses (`error`, `syntax_error`, `timeout`, `crash`, `parse_error`),
mean prompt / model / environment tokens, `sandbox_failure` count.

Problems: `--n` rows evenly spaced over `--rows` (default data/shards/shard-00003.jsonl, training rows
math-train-01500..01999, the last shard), never items-test/.

Outputs (`--out`): `<variant>.jsonl` (one trajectory per line: row, sampler, sample, turns with text, finish reason,
tool statuses, end reason, final answer, correct), `summary.json`, `probe_meta.json` (args, model file hashes, code
and prompt hashes, versions, executor identity, timing).

Usage (container, vLLM worker venv, paired sidecar on /spool as gen_agentic.py):
  probe_tool_use.py --model <area>/models/qwen3-1.7b --out /out --socket /spool/mathexec.sock [--n 64] [--samples 8]
                    [--variants text_base,text_oneshot,text_system,native,native_fewshot]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import textwrap
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE / "executor"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import agentic_core  # noqa: E402
import exec_protocol  # noqa: E402
import score_agentic  # noqa: E402

TRAIN_PROMPT = (HERE / "prompt_math_code.txt").read_text(encoding="utf-8")
EXAMPLE_PROBLEM = "How many integers $n$ with $1 \\le n \\le 2026$ are divisible by 7 but not by 11?"
EXAMPLE_CODE = "sum(1 for n in range(1, 2027) if n % 7 == 0 and n % 11 != 0)"
EXAMPLE_RESULT = "263"
EXAMPLE_ANSWER = "Of the integers from 1 to 2026, \\boxed{263} are divisible by 7 but not by 11."

TEXT_ONESHOT = (
    "Solve the following math problem. You have a Python interpreter (numpy and sympy are available). To run code, "
    "write it between <code> and </code> and stop: the printed output and the value of the last line are returned "
    "between <result> and </result>, and you continue after it. Use code for every step that is not trivial "
    "(arithmetic, algebra with sympy, counting, checking cases) instead of computing by hand. You can run code at "
    "most 3 times. When you know the answer, write it as \\boxed{answer}.\n\n"
    "Example:\n"
    "Problem: " + EXAMPLE_PROBLEM + "\n"
    "I will count them with code.\n"
    "<code>\n" + EXAMPLE_CODE + "\n</code>\n"
    "<result>\n" + EXAMPLE_RESULT + "\n</result>\n"
    + EXAMPLE_ANSWER + "\n\n"
    "Now solve this problem in the same way.\n"
    "Problem: {problem}")

TEXT_SYSTEM = (
    "You are a careful math assistant with a Python interpreter (numpy and sympy are available). To run code, write "
    "it between <code> and </code> and stop; the output is returned between <result> and </result>. Prefer computing "
    "with code over mental arithmetic: write a one-line plan, then the code, then read the result. Run code at most "
    "3 times. End with the final answer as \\boxed{answer}.")

NATIVE_SYSTEM = (
    "You solve math problems. Use the python tool for every step that is not trivial (arithmetic, algebra with "
    "sympy, counting, checking cases) instead of computing by hand; you can call it at most 3 times. Keep the "
    "reasoning short. Put the final answer in \\boxed{}.")
NATIVE_USER = "Problem: {problem}"
PYTHON_TOOL = {"type": "function", "function": {
    "name": "python",
    "description": "Run Python 3 code (numpy and sympy are available) and return what it prints and the value of the "
                   "last line. Each call starts from the state left by the earlier successful calls.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to run"}},
                   "required": ["code"]}}}

VARIANTS = {
    "text_base": {"protocol": "text", "system": None, "user": None},          # None: the training prompt file
    "text_oneshot": {"protocol": "text", "system": None, "user": TEXT_ONESHOT},
    "text_system": {"protocol": "text", "system": TEXT_SYSTEM, "user": None},
    "native": {"protocol": "native", "system": NATIVE_SYSTEM, "user": NATIVE_USER, "shots": False},
    "native_fewshot": {"protocol": "native", "system": NATIVE_SYSTEM, "user": NATIVE_USER, "shots": True},
}

TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"
_TOOL_RE = re.compile(re.escape(TOOL_OPEN) + r"(.*?)" + re.escape(TOOL_CLOSE), re.S)
NATIVE_NEXT = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
TOOL_STATUSES = ("ok", "error", "syntax_error", "timeout", "crash", "parse_error")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


# ------------------------------------------------------------------------------------------------ prompts ---
def messages_for(variant: str, problem: str) -> tuple[list[dict], list[dict] | None]:
    v = VARIANTS[variant]
    if v["protocol"] == "text":
        user = TRAIN_PROMPT.format(problem) if v["user"] is None else v["user"].replace("{problem}", problem)
        msgs = ([{"role": "system", "content": v["system"]}] if v["system"] else []) + \
               [{"role": "user", "content": user}]
        return msgs, None
    msgs = [{"role": "system", "content": v["system"]}]
    if v.get("shots"):
        msgs += [{"role": "user", "content": v["user"].replace("{problem}", EXAMPLE_PROBLEM)},
                 {"role": "assistant", "content": "I will count them with code.",
                  "tool_calls": [{"type": "function", "function": {"name": "python",
                                                                    "arguments": {"code": EXAMPLE_CODE}}}]},
                 {"role": "tool", "content": EXAMPLE_RESULT},
                 {"role": "assistant", "content": EXAMPLE_ANSWER}]
    msgs.append({"role": "user", "content": v["user"].replace("{problem}", problem)})
    return msgs, [PYTHON_TOOL]


def render_prompt(tok, variant: str, problem: str) -> str:
    """Chat template as NeMo RL's math processor uses it (add_generation_prompt, enable_thinking=False)."""
    msgs, tools = messages_for(variant, problem)
    kw = {"tools": tools} if tools else {}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False, **kw)


# ----------------------------------------------------------------------------------------- native protocol ---
def parse_tool_calls(text: str) -> list[dict] | None:
    """None when the turn is not a tool call (no complete `<tool_call>...</tool_call>` at its end); else one entry per
    block: {"code": str} or {"code": None, "error": why}."""
    body = text.rstrip()
    if not body.endswith(TOOL_CLOSE):
        return None
    calls = []
    for raw in _TOOL_RE.findall(body):
        try:
            obj = json.loads(raw.strip())
            args = obj.get("arguments")
            if isinstance(args, str):
                args = json.loads(args)
            if obj.get("name") != "python" or not isinstance(args, dict) or not isinstance(args.get("code"), str):
                raise ValueError("expected {\"name\": \"python\", \"arguments\": {\"code\": \"...\"}}")
            calls.append({"code": textwrap.dedent(args["code"]).strip()})
        except (ValueError, AttributeError) as e:
            calls.append({"code": None, "error": f"{type(e).__name__}: {e}"[:200]})
    return calls or None


def tool_response_text(result: dict, limit: int) -> str:
    """The text inside `<tool_response>`: agentic_core.observation's content without the `<result>` wrapper."""
    obs = agentic_core.observation(result, limit)
    return obs[len("<result>\n"):-len("\n</result>\n")]


def native_observation(outputs: list[str]) -> str:
    """Appended after the model's `<|im_end|>`: what the chat template renders for consecutive `tool` messages plus
    the next generation prompt (enable_thinking=False)."""
    body = "".join(f"\n<tool_response>\n{o}\n</tool_response>" for o in outputs)
    return "\n<|im_start|>user" + body + NATIVE_NEXT


class NativeCore:
    """MathCodeCore's contract (same result keys) for the native `<tool_call>` protocol; evaluation only (no reward)."""

    def __init__(self, execute, max_turns: int = agentic_core.MAX_TURNS, obs_max_chars: int = agentic_core.OBS_MAX_CHARS):
        self.execute = execute
        self.max_turns = int(max_turns)
        self.obs_max_chars = int(obs_max_chars)
        self.calls = 0

    def step(self, samples: list[dict]) -> list[dict]:
        self.calls += 1
        out, pending = [], []
        for i, s in enumerate(samples):
            cells = list((s.get("state") or {}).get("cells") or [])
            calls = parse_tool_calls(s["text"])
            r = {"observation": "", "terminated": True, "end_reason": None, "reward": 0.0, "tool_status": None,
                 "tool_statuses": [], "code": None if calls is None else [c["code"] for c in calls],
                 "state": {**(s.get("state") or {}), "cells": cells}, "_outputs": []}
            if calls is None:
                r["end_reason"] = "final" if "\\boxed" in s["text"] else "no_answer"
            elif s["turn"] >= self.max_turns:
                r["end_reason"] = "max_turns"
            else:
                r["terminated"], r["reward"] = False, None
                pending.append((i, calls))
            out.append(r)
        rnd = 0
        while any(rnd < len(calls) for _, calls in pending):
            jobs, idx = [], []
            for i, calls in pending:
                if rnd >= len(calls):
                    continue
                c = calls[rnd]
                if c["code"] is None:
                    out[i]["tool_statuses"].append("parse_error")
                    out[i]["_outputs"].append(f"Error: the tool call could not be parsed ({c['error']})")
                    continue
                jobs.append({"id": f"{self.calls}:{i}:{rnd}",
                             "cells": agentic_core.replay_cells(out[i]["state"]["cells"]) + [c["code"]]})
                idx.append((i, c["code"]))
            for (i, code), res in zip(idx, self.execute(jobs) if jobs else []):
                r = out[i]
                r["tool_statuses"].append(res["status"])
                r["_outputs"].append(tool_response_text(res, self.obs_max_chars))
                r["state"] = {**r["state"], "cells": r["state"]["cells"] + [{"code": code, "status": res["status"]}]}
            rnd += 1
        for r in out:
            if not r["terminated"]:
                r["observation"] = native_observation(r["_outputs"])
                r["tool_status"] = r["tool_statuses"][0] if r["tool_statuses"] else None
            del r["_outputs"]
        return out


# --------------------------------------------------------------------------------------------- turn loop ---
def run_trajectories(states: list[dict], generate, tokenize, core, max_model_len: int, max_new_tokens: int) -> None:
    """gen_agentic.run_trajectories with a configurable budget, per-state sampler and NativeCore support.
    `generate(states, max_tokens) -> [(new ids, text, finish_reason)]`."""
    for turn in range(1, core.max_turns + 1):
        active = [s for s in states if not s["done"]]
        if not active:
            return
        max_tokens = [min(max_new_tokens, max_model_len - len(s["ids"])) for s in active]
        outs = generate(active, max_tokens)
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
                    s["done"], s["error"] = True, {"class": "sandbox_failure", "message": str(e)[:500]}
                continue
        for s, r, n_in in zip(active, res, input_lens):
            t = s["turns"][-1]
            statuses = r.get("tool_statuses") if "tool_statuses" in r else (
                [r["tool_status"]] if r["tool_status"] is not None else [])
            attempted = bool(statuses) or r["end_reason"] == "max_turns"
            t.update({"tool_call": attempted, "tool_statuses": statuses, "observation": r["observation"]})
            obs_ids = list(tokenize(r["observation"])) if r["observation"] else []
            overflow = len(obs_ids) + t["n_tokens"] + n_in >= max_model_len
            if overflow:
                obs_ids = obs_ids[:max(0, max_model_len - (t["n_tokens"] + n_in))]
            t["obs_tokens"] = len(obs_ids)
            s["ids"] = s["ids"] + obs_ids
            s["cells"] = r["state"]["cells"]
            if r["terminated"]:
                s["done"], s["end_reason"] = True, r["end_reason"]
            elif overflow:
                s["done"], s["end_reason"] = True, "overflow"


def trajectory_doc(s: dict, variant: str) -> dict:
    final = s["turns"][-1]["text"] if s["turns"] else ""
    ans = score_agentic.parse_answer(final)
    ref = score_agentic.reference_int(s["answer"])
    statuses = [x for t in s["turns"] for x in t.get("tool_statuses", [])]
    fake = "<result>" if VARIANTS[variant]["protocol"] == "text" else "<tool_response>"
    return {"variant": variant, "row_id": s["row_id"], "sampler": s["sampler"], "sample": s["sample"],
            "prompt_tokens": s["prompt_tokens"], "turns": s["turns"], "end_reason": s["end_reason"],
            "error": s["error"], "final_text": final, "answer": ans, "reference": ref,
            "correct": int(ans is not None and ans == ref),
            "tool_calls_attempted": sum(t.get("tool_call", False) for t in s["turns"]),
            "tool_calls_executed": sum(x != "parse_error" for x in statuses), "tool_statuses": statuses,
            "boxed": int("\\boxed" in final), "truncated": int(any(t["finish_reason"] == "length" for t in s["turns"])),
            "fake_result": int(any(fake in t["text"] for t in s["turns"])),
            "model_tokens": sum(t["n_tokens"] for t in s["turns"]),
            "env_tokens": sum(t.get("obs_tokens", 0) for t in s["turns"])}


def summarize(docs: list[dict]) -> dict:
    ok = [d for d in docs if d["error"] is None]
    n = len(ok)

    def share(f):
        return round(sum(1 for d in ok if f(d)) / n, 4) if n else None

    def mean(f):
        return round(sum(f(d) for d in ok) / n, 2) if n else None

    tool = [d for d in ok if d["tool_calls_attempted"]]
    notool = [d for d in ok if not d["tool_calls_attempted"]]
    ends: dict[str, int] = {}
    for d in ok:
        ends[str(d["end_reason"])] = ends.get(str(d["end_reason"]), 0) + 1
    return {
        "trajectories": n, "sandbox_failure": len(docs) - n,
        "tool_share": share(lambda d: d["tool_calls_attempted"] > 0),
        "executed_share": share(lambda d: d["tool_calls_executed"] > 0),
        "tool_calls_mean": mean(lambda d: d["tool_calls_attempted"]),
        "turns_mean": mean(lambda d: len(d["turns"])),
        "boxed_share": share(lambda d: d["boxed"]), "answer_share": share(lambda d: d["answer"] is not None),
        "exact_match": share(lambda d: d["correct"]),
        "exact_match_tool": round(sum(d["correct"] for d in tool) / len(tool), 4) if tool else None,
        "exact_match_no_tool": round(sum(d["correct"] for d in notool) / len(notool), 4) if notool else None,
        "truncated_share": share(lambda d: d["truncated"]), "fake_result_share": share(lambda d: d["fake_result"]),
        "end_reasons": ends,
        "tool_statuses": {k: sum(d["tool_statuses"].count(k) for d in ok) for k in TOOL_STATUSES},
        "prompt_tokens_mean": mean(lambda d: d["prompt_tokens"]), "model_tokens_mean": mean(lambda d: d["model_tokens"]),
        "env_tokens_mean": mean(lambda d: d["env_tokens"])}


def select_rows(rows: list[dict], n: int) -> list[dict]:
    if n >= len(rows):
        return rows
    return [rows[(i * len(rows)) // n] for i in range(n)]


# -------------------------------------------------------------------------------------------------- main ---
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--rows", type=Path, default=HERE / "data" / "shards" / "shard-00003.jsonl")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-turns", type=int, default=agentic_core.MAX_TURNS)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--socket", required=True)
    p.add_argument("--seed", type=int, default=10100)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    a = p.parse_args(argv)
    variants = [v for v in a.variants.split(",") if v]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; known {list(VARIANTS)}")
    if "items-test" in str(a.rows.resolve()):
        raise SystemExit("the probe runs on training rows only")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    t0 = time.time()
    rows = select_rows([json.loads(x) for x in open(a.rows, encoding="utf-8") if x.strip()], a.n)
    executor = exec_protocol.ping(a.socket, timeout=10.0)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import transformers
    import vllm

    tok = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
    llm = LLM(model=str(a.model), dtype="bfloat16", seed=0, max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_memory_utilization)

    def tokenize(text: str) -> list[int]:
        return tok(text, add_special_tokens=False)["input_ids"]

    summary = {}
    for variant in variants:
        native = VARIANTS[variant]["protocol"] == "native"
        stop = None if native else [agentic_core.STOP]
        execute = lambda jobs: exec_protocol.call_executor(a.socket, jobs, 1800.0)  # noqa: E731
        core = NativeCore(execute, a.max_turns) if native else \
            agentic_core.MathCodeCore(execute=execute, verifier=None, max_turns=a.max_turns)
        states = []
        for r_i, row in enumerate(rows):
            prompt = render_prompt(tok, variant, row["input"])
            ids = tokenize(prompt)
            for sampler, k in [("greedy", 0)] + [("sampled", k) for k in range(a.samples)]:
                states.append({"row_id": row["row_id"], "answer": row["answer"], "sampler": sampler, "sample": k,
                               "seed": a.seed + 1000 * r_i + k, "prompt_tokens": len(ids), "ids": list(ids),
                               "turns": [], "cells": [], "done": False, "end_reason": None, "error": None})

        def generate(active, max_tokens):
            sps = [SamplingParams(temperature=0.0 if s["sampler"] == "greedy" else a.temperature, top_p=1.0, top_k=-1,
                                  max_tokens=int(m), stop=stop, include_stop_str_in_output=True, seed=s["seed"])
                   for s, m in zip(active, max_tokens)]
            res = []
            for o in llm.generate([{"prompt_token_ids": s["ids"]} for s in active], sps, use_tqdm=False):
                new = list(o.outputs[0].token_ids)
                res.append((new, tok.decode(new, skip_special_tokens=True), o.outputs[0].finish_reason))
            return res

        tv = time.time()
        run_trajectories(states, generate, tokenize, core, a.max_model_len, a.max_new_tokens)
        docs = [trajectory_doc(s, variant) for s in states]
        (out / f"{variant}.jsonl").write_text("".join(json.dumps(d, sort_keys=True) + "\n" for d in docs),
                                              encoding="utf-8")
        summary[variant] = {smp: summarize([d for d in docs if d["sampler"] == smp]) for smp in ("sampled", "greedy")}
        summary[variant]["seconds"] = round(time.time() - tv, 1)
        print(json.dumps({"variant": variant, **{smp: {k: summary[variant][smp][k] for k in (
            "tool_share", "executed_share", "turns_mean", "boxed_share", "exact_match", "truncated_share")}
            for smp in ("sampled", "greedy")}}), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    model_files = {f.name: sha256_file(f) for f in sorted(a.model.iterdir())
                   if f.is_file() and f.suffix in (".json", ".safetensors", ".txt", ".jinja")}
    meta = {"args": {k: str(v) for k, v in vars(a).items()}, "variants": variants, "rows": [r["row_id"] for r in rows],
            "rows_file_sha256": sha256_file(a.rows), "model_files": model_files,
            "code_sha256": {f: sha256_file(HERE / f) for f in ("probe_tool_use.py", "agentic_core.py",
                                                               "score_agentic.py", "executor/exec_protocol.py")},
            "prompt_file_sha256": sha256_file(HERE / "prompt_math_code.txt"), "executor": executor,
            "versions": {"vllm": vllm.__version__, "transformers": transformers.__version__,
                         "python": sys.version.split()[0]},
            "wall_seconds": round(time.time() - t0, 1)}
    (out / "probe_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    hdr = ["variant", "sampler", "tool", "exec", "calls", "turns", "boxed", "EM", "EM|tool", "EM|no", "trunc", "fake"]
    print("\t".join(hdr))
    for v in variants:
        for smp in ("sampled", "greedy"):
            s = summary[v][smp]
            print("\t".join(str(x) for x in (v, smp, s["tool_share"], s["executed_share"], s["tool_calls_mean"],
                                             s["turns_mean"], s["boxed_share"], s["exact_match"], s["exact_match_tool"],
                                             s["exact_match_no_tool"], s["truncated_share"], s["fake_result_share"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
