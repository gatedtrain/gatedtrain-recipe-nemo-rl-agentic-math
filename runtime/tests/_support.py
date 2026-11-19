"""Shared helpers of area "agentic RL"'s CPU tests (host, no GPU, no docker, no network).

- Paths and the area modules on sys.path. `NEMO_RL`: a NeMo RL clone at 4f39fd8 (`GATEDTRAIN_NEMO_RL_DIR`);
  `OUTPUTS`: the directory holding the GPU runs' outputs (`GATEDTRAIN_RECIPE_NEMO_RL_AGENTIC_MATH_OUTPUTS`); tests that
  read a run skip when it is absent.
- `install_nemo_stubs()`: minimal stand-ins for `ray` and the few `nemo_rl` modules that NeMo RL's
  `sliding_puzzle.py` and this area's Ray actors import, so the pinned clone's real `SlidingPuzzleGameLogic`,
  `SlidingPuzzleRunner` and `SlidingPuzzleEnv` (loaded from the NeMo RL clone `NEMO_RL`, read only)
  and this area's `MathCodeEnvironment` / `LoggedSlidingPuzzleEnv` classes run in-process. `ray.remote` returns the
  class itself and sets `__ray_actor_class__`, as the real ActorClass exposes it. torch is the real one (area-4 venv).
- `fake_rollout()`: a re-implementation of the bookkeeping of NeMo RL's `run_multi_turn_rollout`
  (rollouts.py:1084-1243: per-turn generation for active samples, environment call with role/content only, rewards
  summed over turns, observation appended and cut at max_seq_len (sample ends), metadata None keeps the previous one,
  done = terminated or cut) plus the fields `grpo_train` logs to `train_data_step<N>.jsonl` (grpo.py:4667-4691),
  with a whitespace tokenizer. It is a test double, not NeMo RL.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Generic, NamedTuple, TypeVar

AREA = Path(__file__).resolve().parent.parent
NEMO_RL = Path(os.environ.get("GATEDTRAIN_NEMO_RL_DIR") or "/nonexistent/nemo-rl")
OUTPUTS = Path(os.environ.get("GATEDTRAIN_RECIPE_NEMO_RL_AGENTIC_MATH_OUTPUTS") or "/nonexistent/nemo-rl-agentic-math-outputs")
for _p in (AREA, AREA / "executor"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ------------------------------------------------------------------------------------------------- stubs ---
class _EnvironmentReturn(NamedTuple):
    observations: list
    metadata: list
    next_stop_strings: list
    rewards: object
    terminateds: object
    answers: object


_T = TypeVar("_T")


class _EnvironmentInterface(Generic[_T]):
    pass


def _remote(*args, **kwargs):
    def wrap(cls):
        cls.__ray_actor_class__ = cls
        return cls
    if args and isinstance(args[0], type) and not kwargs:
        return wrap(args[0])
    return wrap


class StubVerifyWorker:
    """Stands in for NeMo RL's HFVerifyWorker: 1.0 when the last \\boxed{...} equals the ground truth."""

    def verify(self, pred_responses, ground_truths, return_extracted_answer=False, **kwargs):
        import re
        if kwargs.get("math_verify_impl") != "hf_math_verify":
            return [0.0] * len(pred_responses)       # as the real one (math_environment.py:131-146)
        out = []
        for r, g in zip(pred_responses, ground_truths):
            m = re.findall(r"\\boxed\{([^{}]*)\}", r)
            out.append(float(bool(m) and m[-1].strip() == g))
        return out


StubVerifyWorker.__ray_actor_class__ = StubVerifyWorker


def install_nemo_stubs() -> None:
    if "nemo_rl.environments.interfaces" in sys.modules and getattr(sys.modules["ray"], "_gatedtrain_stub", False):
        return
    ray = types.ModuleType("ray")
    ray._gatedtrain_stub = True
    ray.remote = _remote
    sys.modules["ray"] = ray
    mods = {}
    for name in ("nemo_rl", "nemo_rl.data", "nemo_rl.data.interfaces", "nemo_rl.distributed",
                 "nemo_rl.distributed.batched_data_dict", "nemo_rl.environments", "nemo_rl.environments.interfaces",
                 "nemo_rl.environments.games", "nemo_rl.environments.math_environment"):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
        mods[name] = m
    mods["nemo_rl.data.interfaces"].LLMMessageLogType = list
    mods["nemo_rl.distributed.batched_data_dict"].BatchedDataDict = dict
    mods["nemo_rl.environments.interfaces"].EnvironmentInterface = _EnvironmentInterface
    mods["nemo_rl.environments.interfaces"].EnvironmentReturn = _EnvironmentReturn
    mods["nemo_rl.environments.math_environment"].HFVerifyWorker = StubVerifyWorker
    name = "nemo_rl.environments.games.sliding_puzzle"
    spec = importlib.util.spec_from_file_location(name, NEMO_RL / "nemo_rl/environments/games/sliding_puzzle.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)


def load_fresh(module_file: str, name: str):
    """Import an area module under another name (after install_nemo_stubs, so its Ray actor is defined)."""
    spec = importlib.util.spec_from_file_location(name, AREA / module_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------ rollout double ---
FAKE_EOS = 151645


def ntok(text: str) -> int:
    return len(text.split())


def fake_rollout(prompts: list[str], extra_env_info: list[dict], policy, env_step, max_turns: int, max_seq: int):
    """Returns per-sample train_data_step lines in the logged shape ({content, rewards, input_lengths, token_ids,
    token_loss_mask, sample_loss_mask} each a batch of one, plus idx)."""
    B = len(prompts)
    logs = [[{"role": "user", "content": p, "n": ntok(p)}] for p in prompts]
    md = [dict(e) for e in extra_env_info]
    total = [0.0] * B
    active = list(range(B))
    for turn in range(max_turns):
        if not active:
            break
        input_lens = {i: sum(m["n"] for m in logs[i]) for i in active}
        for i in active:
            text = policy(i, turn, logs[i])
            n = ntok(text)
            assert n >= 1
            # the double's turns end at EOS unless capped or stopped at the tool-call stop string
            eos = n <= max_seq - input_lens[i] and not text.rstrip().endswith("</code>")
            n = min(n, max_seq - input_lens[i])                  # cap_max_tokens_to_context
            text = " ".join(text.split()[:n])
            logs[i].append({"role": "assistant", "content": text, "n": n, "eos": eos})
        msgs = [[{"role": m["role"], "content": m["content"]} for m in logs[i]] for i in active]
        obs, md_out, _stops, rewards, terms = env_step(msgs, [md[i] for i in active])
        nxt = []
        for k, i in enumerate(active):
            total[i] += float(rewards[k])
            o = obs[k]
            n_o = ntok(o)
            gen_n = logs[i][-1]["n"]
            cut = n_o + gen_n + input_lens[i] >= max_seq
            if cut:
                n_o = max_seq - (gen_n + input_lens[i])
            logs[i].append({"role": "environment", "content": o, "n": n_o})
            if not (cut or terms[k]):
                nxt.append(i)
                if md_out[k] is not None:
                    md[i] = md_out[k]
        active = nxt
    lines = []
    for i in range(B):
        mask, ids = [], []
        for m in logs[i]:
            mask += [1 if m["role"] == "assistant" else 0] * m["n"]
            ids += [FAKE_EOS if m["role"] == "assistant" and k == m["n"] - 1 and m.get("eos") else 7
                    for k in range(m["n"])]
        # As logged by Logger.log_batched_dict_as_jsonl (logger.py:1427-1441): every field a batch of one, idx = line,
        # sequences right-padded (3 here).
        lines.append({"content": [[m["content"] for m in logs[i]]], "rewards": [total[i]],
                      "input_lengths": [len(mask)], "token_ids": [ids + [0] * 3],
                      "token_loss_mask": [mask + [0] * 3], "sample_loss_mask": [1.0], "idx": i})
    return lines


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
