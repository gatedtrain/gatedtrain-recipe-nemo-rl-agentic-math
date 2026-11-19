"""NeMo RL pieces of area "agentic RL" (MM-27: wrappers and registrations at run time; no NeMo RL source is edited).
Imported by run_agentic_nemo.py in the driver and by Ray in the environment actor (NeMo RL image, commit
4f39fd86289a2ffd5aaf5d202702f7fd7185c924, the image's system Python).

1. `math_code_rowid_processor` (registered as `math_code_rowid`; processors.py:1145 `register_processor`): NeMo RL's
   own `math_hf_data_processor` (processors.py:553-614: optional system prompt, `prompt.format(problem)`, the
   tokenizer's chat template with `add_generation_prompt=True` - the tokenizer's `apply_chat_template` is a partial
   with `enable_thinking=False` when `policy.tokenizer.chat_template_kwargs` sets it, algorithms/utils.py:492-495 -
   `extra_env_info = {"ground_truth": messages[1].content}`) plus `row_id` in `extra_env_info` and the per-datum
   stop string `</code>` (`DatumSpec.stop_strings`, data/interfaces.py:74, collated at collate_fn.py:231-232). Note:
   the synchronous rollout passes the batch's own `stop_strings` to every turn (rollouts.py:671-672 inside
   `generate_responses`), and vLLM merges all samples' stop strings into one set (vllm_worker.py:854-865), so the
   datum's `</code>` is what applies at every turn; the environment's `next_stop_strings` is the same value. This
   only works when vLLM has a tokenizer: the launcher therefore also sets `policy.generation.stop_strings`
   (run_agentic_nemo.py; without it NeMo RL sets skip_tokenizer_init=True and vLLM ignores stop strings, P1 p1-a2).
2. `MathCodeEnvironment`, the Ray actor registered as env `math_code` (`ENV_REGISTRY` entry and
   `ACTOR_ENVIRONMENT_REGISTRY[fqn] = PY_EXECUTABLES.SYSTEM`, environments/utils.py:111-137,
   distributed/ray_actor_environment_registry.py:64-80; text datasets are created under their own `env_name`,
   data/utils.py:198-203). Implements `EnvironmentInterface` (environments/interfaces.py:54-90) and returns
   `EnvironmentReturn` (:26-51). Per call it sees each active trajectory's whole message log (role/content only,
   rollouts.py:915-918); the turn decision, the executor call (stateless replay in the sidecar) and the reward are
   `agentic_core.MathCodeCore` (shared with the evaluator). The reward function is NeMo RL's own `HFVerifyWorker`
   class (math_environment.py:86-168) instantiated in-process (its Ray wrapper bypassed with `__ray_actor_class__`,
   as area 7's vlm_env_logged.py does with VLMEnvironment) and called with `math_verify_impl="hf_math_verify"`
   explicitly: without the keyword its `verify` raises ValueError internally and returns 0 for every sample
   (math_environment.py:131-146). Every call is logged by `traj_log.TrajectoryLogger` (env-agnostic).
   Config (`env.math_code`): `socket` (executor Unix socket, required), `log_dir` (env_calls.jsonl,
   trajectories.jsonl), `max_turns` (= grpo.max_rollout_turns), `obs_max_chars`, `timeout_s` (one executor batch).
   An executor fault raises (the GRPO step fails rather than training on a reward that was not the policy's, MM-4).

The host has neither ray nor nemo_rl: there only the processor's helper and `build_core` are importable; the CPU
tests drive `MathCodeCore` and `TrajectoryLogger` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE / "executor"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import agentic_core  # noqa: E402
import exec_protocol  # noqa: E402
import traj_log  # noqa: E402

ENV_ID = "math_code"


def math_code_rowid_processor(datum_dict, task_data_spec, tokenizer, max_seq_length, idx):
    from nemo_rl.data.processors import math_hf_data_processor

    out = math_hf_data_processor(datum_dict, task_data_spec, tokenizer, max_seq_length, idx)
    out["extra_env_info"]["row_id"] = datum_dict["row_id"]
    out["stop_strings"] = [agentic_core.STOP]
    return out


def build_core(cfg: dict, verifier=None) -> agentic_core.MathCodeCore:
    sock = cfg["socket"]
    timeout = float(cfg.get("timeout_s", 900))
    return agentic_core.MathCodeCore(execute=lambda jobs: exec_protocol.call_executor(sock, jobs, timeout),
                                     verifier=verifier, max_turns=int(cfg.get("max_turns", agentic_core.MAX_TURNS)),
                                     obs_max_chars=int(cfg.get("obs_max_chars", agentic_core.OBS_MAX_CHARS)))


def env_step(core: agentic_core.MathCodeCore, tlog: traj_log.TrajectoryLogger, message_log_batch, metadata):
    """Framework-free body of MathCodeEnvironment.step: returns (observations, metadata, next_stop_strings, rewards,
    terminateds) as plain lists."""
    samples = []
    for conv, md in zip(message_log_batch, metadata):
        assistant = [m for m in conv if m["role"] == "assistant"]
        state = {k: v for k, v in (md or {}).items() if k != traj_log.STAMP}
        samples.append({"turn": len(assistant), "text": str(assistant[-1]["content"]), "state": state})
    res = core.step(samples)
    rewards = [r["reward"] if r["reward"] is not None else 0.0 for r in res]
    terms = [r["terminated"] for r in res]
    obs = [r["observation"] for r in res]
    md_out = [None if r["terminated"] else r["state"] for r in res]
    md_out = tlog.record(message_log_batch, metadata, md_out, rewards, terms, obs,
                         [r["end_reason"] for r in res], [r["tool_status"] for r in res])
    stops = [None if t else [agentic_core.STOP] for t in terms]
    return obs, md_out, stops, rewards, terms


try:  # the Ray actor exists only inside the NeMo RL image
    import ray
    import torch
    from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
    from nemo_rl.environments.math_environment import HFVerifyWorker

    @ray.remote(max_restarts=0, max_task_retries=0)  # pragma: no cover
    class MathCodeEnvironment(EnvironmentInterface):
        def __init__(self, cfg: dict):
            cfg = dict(cfg)
            self._verify = HFVerifyWorker.__ray_actor_class__()
            self.core = build_core(cfg, verifier=self._reward)
            self.tlog = traj_log.TrajectoryLogger(cfg.get("log_dir"), ENV_ID)

        def _reward(self, text: str, ground_truth: str) -> float:
            return float(self._verify.verify([text], [ground_truth], math_verify_impl="hf_math_verify")[0])

        def step(self, message_log_batch, metadata):
            obs, md, stops, rewards, terms = env_step(self.core, self.tlog, message_log_batch, metadata)
            return EnvironmentReturn(
                observations=[{"role": "environment", "content": o} for o in obs], metadata=md,
                next_stop_strings=stops, rewards=torch.tensor(rewards, dtype=torch.float32),
                terminateds=torch.tensor(terms, dtype=torch.bool), answers=None)

        def global_post_process_and_metrics(self, batch):
            r = batch.get("total_reward")
            return batch, {"math_code_reward": float(r.float().mean().item()) if r is not None else 0.0}

        def shutdown(self):
            return None
except ImportError:  # host: no ray / nemo_rl
    MathCodeEnvironment = None
