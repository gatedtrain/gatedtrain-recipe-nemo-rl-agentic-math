"""Framework-free core of area "agentic RL"'s `math_code` environment (standard library only).

Shared, byte for byte, by the training environment (`math_code_env.py`, a NeMo RL Ray actor in the trainer
container) and the evaluation generator (`gen_agentic.py`, vLLM in the generation container), so a trajectory is
decided the same way in both (README section 0, "Turn protocol"). Generated code never runs here: tool calls go to
the executor sidecar through `executor/exec_protocol.py`.

Turn protocol (`<code>`/`<result>` text inside one assistant conversation; the maintainers' decisions "Area 10
decisions", D6 a):
- Every generation stops at `</code>` (NeMo RL's stop string, included in the output: vllm_worker.py:891) or at
  EOS / the token budget.
- A turn whose text ends with `</code>` (trailing whitespace ignored) and contains an opening `<code>` before it is a
  tool call: the code is the text between the last `<code>` and that final `</code>` (a Markdown fence inside is
  stripped, common indentation removed). Below `max_turns` it is executed (stateless replay of the trajectory's
  earlier replayable cells plus this one) and the observation `<result>\\n...\\n</result>\\n` is returned; the
  trajectory continues.
- Any other turn (EOS, budget exhausted, text after the last `</code>`, no `<code>`) is terminal: end reason
  `final` when its text contains `\\boxed` (an answer was given), else `no_answer` (no tool call and no answer:
  a reply cut by the token budget, an unclosed `<code>`, or prose without `\\boxed`; P1 p1-a2 had 179 of 384
  such replies, all logged as `final` before this split). The reward is the same either way (0 without `\\boxed`).
- A tool call at turn `max_turns` is not executed: end reason `max_turns`. Both terminal cases return an empty
  observation (NeMo RL appends it as zero tokens, rollouts.py:1172-1179).
- Reward (training only, `verifier` given): at the terminal turn, math_verify on the final assistant turn's text
  against the integer reference, and only when that text contains `\\boxed` (the prompt asks for it and the
  evaluator scores only a boxed answer); 0 elsewhere. Observations are cut to `obs_max_chars` (head and tail).
"""

from __future__ import annotations

import hashlib
import re
import textwrap
from typing import Callable

STOP = "</code>"
OPEN = "<code>"
MAX_TURNS = 4                   # grpo.max_rollout_turns: at most 3 executed tool calls, then an answer
OBS_MAX_CHARS = 512
NON_REPLAY = ("syntax_error", "timeout", "crash")   # cells not replayed later (never ran to a state)
END_REASONS = ("final", "no_answer", "max_turns", "overflow")
_FENCE_OPEN = re.compile(r"^\s*```[ \t]*(?:python|py)?[ \t]*\n", re.I)
_FENCE_CLOSE = re.compile(r"\n?```\s*$")


def sha256_text(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


def extract_code(text: str) -> str | None:
    """Code of a tool-call turn, or None when the turn is not a tool call."""
    body = text.rstrip()
    if not body.endswith(STOP):
        return None
    body = body[: -len(STOP)]
    start = body.rfind(OPEN)
    if start < 0:
        return None
    code = body[start + len(OPEN):]
    m = _FENCE_OPEN.match(code)
    if m:
        code = code[m.end():]
        code = _FENCE_CLOSE.sub("", code)
    return textwrap.dedent(code).strip()


def cut_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n...[{omitted} characters omitted]...\n{text[-tail:]}"


def observation(result: dict, limit: int = OBS_MAX_CHARS) -> str:
    """Observation text for one executor result (exec_protocol statuses)."""
    status = result["status"]
    out = result.get("output") or ""
    if result.get("output_chars", len(out)) > len(out):
        out = out + f"\n...[output truncated at {len(out)} of {result['output_chars']} characters]"
    if status == "timeout":
        text = (out + "\n" if out else "") + "TimeoutError: the code ran out of time"
    elif status == "crash":
        text = "Error: the code crashed the Python process"
    else:
        text = out.rstrip("\n") if out.strip() else "(no output)"
    return f"<result>\n{cut_middle(text, limit)}\n</result>\n"


def replay_cells(cells: list[dict]) -> list[str]:
    return [c["code"] for c in cells if c["status"] not in NON_REPLAY]


class MathCodeCore:
    """Decides every sample of one environment call.

    `execute(jobs) -> results` sends jobs to the executor (exec_protocol.call_executor bound to a socket);
    `verifier(text, ground_truth) -> float` is the training reward (None in evaluation).
    """

    def __init__(self, execute: Callable[[list[dict]], list[dict]], verifier: Callable[[str, str], float] | None = None,
                 max_turns: int = MAX_TURNS, obs_max_chars: int = OBS_MAX_CHARS):
        self.execute = execute
        self.verifier = verifier
        self.max_turns = int(max_turns)
        self.obs_max_chars = int(obs_max_chars)
        self.calls = 0

    def step(self, samples: list[dict]) -> list[dict]:
        """samples: [{"turn": number of assistant turns so far (1-based), "text": the last assistant turn,
        "state": {"cells": [{code, status}], "ground_truth": str?, ...}}]. Returns per sample
        {"observation", "terminated", "end_reason", "reward", "tool_status", "code", "state"}."""
        self.calls += 1
        out: list[dict] = []
        jobs, job_idx = [], []
        for i, s in enumerate(samples):
            cells = list((s.get("state") or {}).get("cells") or [])
            code = extract_code(s["text"])
            r = {"observation": "", "terminated": True, "end_reason": None, "reward": None, "tool_status": None,
                 "code": code, "state": {**(s.get("state") or {}), "cells": cells}}
            if code is None:
                r["end_reason"] = "final" if "\\boxed" in s["text"] else "no_answer"
            elif s["turn"] >= self.max_turns:
                r["end_reason"] = "max_turns"
            else:
                r["terminated"] = False
                jobs.append({"id": f"{self.calls}:{i}", "cells": replay_cells(cells) + [code]})
                job_idx.append(i)
            out.append(r)
        if jobs:
            results = self.execute(jobs)
            for i, res in zip(job_idx, results):
                r = out[i]
                r["tool_status"] = res["status"]
                r["observation"] = observation(res, self.obs_max_chars)
                cell = {"code": r["code"], "status": res["status"]}
                r["state"] = {**r["state"], "cells": r["state"]["cells"] + [cell]}
        for s, r in zip(samples, out):
            if r["terminated"]:
                r["reward"] = 0.0
                if self.verifier is not None and "\\boxed" in s["text"]:
                    r["reward"] = float(self.verifier(s["text"], str((s.get("state") or {})["ground_truth"])))
        return out
