#!/usr/bin/env python
"""NeMo RL launcher of area "agentic RL": stage `agentic_grpo` (`rl.nemo_rl_agentic@0.1`), multi-turn GRPO of
Qwen3-1.7B (LoRA r32) with a Python tool on MATH (env `math_code`).

Runs inside the derived NeMo RL image `gatedtrain-a6-nemo-rl:1` (as area 7: Tier B `nemo-rl:4f39fd8`, NeMo RL commit
4f39fd86289a2ffd5aaf5d202702f7fd7185c924), cwd /opt/nemo-rl, the image's system interpreter
`/opt/nemo_rl_venv/bin/python`, paired with the executor sidecar (executor/run-executor.sh) through the spool socket.
It records the run, pings the executor (stops if it does not answer), installs the cut writer of
ptc_agentic_hooks.py, registers the row-id processor `math_code_rowid` and the environment `math_code`
(math_code_env.py), builds the overrides of the profile and memory rung and calls the unmodified example entry point
`examples/run_grpo.py` `main()` with `examples/configs/grpo_math_1B.yaml`, so NeMo RL parses the overrides itself.

Trainer (verified in the pinned clone): the classic synchronous `grpo_train` (`data_plane.enabled=false`, area 7
probe-a5), `run_multi_turn_rollout` (grpo.py:3924-3936; the async variant only with `vllm_cfg.async_engine`, which
stays false, generation/interfaces.py:287-302), `grpo.max_rollout_turns` 4 (GRPOConfig default 1, grpo.py:362),
`policy.max_total_sequence_length` 4,096 as the rollout's max_seq_len (grpo.py:3929-3932) and vLLM's max_model_len
(grpo_math_1B.yaml interpolation), at most 1,024 new tokens per turn capped to the remaining context
(`vllm_cfg.cap_max_tokens_to_context`, vllm_worker.py:1177-1199; without it a late turn could ask vLLM for more than
the context). Prompt and budget come from gen_agentic.SAMPLER (one definition for training and evaluation): the
deployment prompt file `prompt_math_code_oneshot.txt` (README 0.11; probe-tool-2's `text_oneshot`), whose name and
SHA-256 go into run_args.json and every cut's sampler.json. KL 0 with the reference pass skipped (area 7), thinking off
(`policy.tokenizer.chat_template_kwargs.enable_thinking=false`, applied as a partial of apply_chat_template,
algorithms/utils.py:465-495), sequence packing off as in NeMo RL's Qwen3 LoRA recipe
(examples/configs/recipes/llm/grpo-qwen3-8B-base-1n8g-automodel-lora.yaml), no Automodel backend override (Qwen3 dense
loads as the transformers class, which rejects `backend`; P1 p1-a), micro batches of one sequence for training and
logprobs (area 7: fp32 logits; vocabulary 151,936: 4,096 x 151,936 x 4 B = 2.5 GB per fp32 copy of one sequence).

Memory rungs (README "Memory and time"; the first whose probe peaks at <= 22.5 GB is used):
- `a` (probe first): LoRA rank 32, alpha 64, all decoder projections (`*_proj`, triton kernels off), plain
  AdamW lr 2e-5 (the base is loaded in fp32 because AdamW keeps no fp32 master, automodel/setup.py:776-780),
  activation checkpointing, vLLM `gpu_memory_utilization` 0.35, `policy.logprob_chunk_size` 512 (added with the
  4,096 context: log-softmax and gather per 512-position chunk, recomputed per chunk in the loss,
  automodel/train.py:712-735 and :939-985; same values, bounds the fp32 log-softmax copies to 0.3 GB each instead of
  2.5 GB). LoRA weights reach vLLM by NeMo RL's native LoRA refit (`lora_refit_mode: native`, lora_utils.py; vLLM
  `enable_lora` is set by NeMo RL itself). P1 p1-a2 peaked at 11.0 GB with sequences of at most 858 tokens.
- `b`: `a` with TE FusedAdam fp32 master weights and bf16 moments (base loaded in bf16, about -4 GB; LoRA plus TE
  FusedAdam unverified), `policy.logprob_chunk_size` 512, vLLM 0.4.
- `c`: full fine-tune (no LoRA; outside the recipe's scope, on explicit request only), area 7's rung a optimizer,
  `max_total_sequence_length` 1,536, vLLM 0.15; expected to exceed 22.5 GB. Predates the one-shot prompt and the
  1,024 / 4,096 budget (the longest training prompt is 1,557 tokens > 1,536: NeMo RL's context cap would raise,
  vllm_worker.py:92-102); main() refuses it until it is re-planned.

Usage (container):
  run_agentic_nemo.py --profile probe|smoke|full --out /out --seed 10100 --base <area>/models/qwen3-1.7b
                      --memory a --socket /spool/mathexec.sock [--max-steps N]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "executor"))
import agentic_core  # noqa: E402
import exec_protocol  # noqa: E402
import gen_agentic  # noqa: E402  (SAMPLER and PROMPT_FILE only; importing it loads no model)
import ptc_agentic_hooks as hooks  # noqa: E402

DATA = HERE / "data" / "train.jsonl"
PROMPT = gen_agentic.PROMPT_FILE                          # the deployment prompt, as the evaluator's
MAX_NEW_TOKENS = gen_agentic.SAMPLER["max_new_tokens_per_turn"]     # 1,024 per turn
MAX_SEQ = gen_agentic.SAMPLER["max_model_len"]                      # 4,096: rollout max_seq_len = vLLM max_model_len
ENV_ID = "math_code"
MAX_TURNS = 4
PROFILES = {
    "probe": {"steps": 3, "prompts": 16, "generations": 8, "save_period": 3},
    "smoke": {"steps": 5, "prompts": 16, "generations": 8, "save_period": 5},
    "full": {"steps": 60, "prompts": 16, "generations": 8, "save_period": 20},
}
LORA = {"dim": 32, "alpha": 64}
# No Automodel backend override: in this image Qwen3 dense loads as transformers' Qwen3ForCausalLM, whose
# __init__ takes no `backend` keyword (P1 probe p1-a, 2026-11-15: TypeError in AutomodelPolicyWorker.__init__); the
# HF class has no TE layers, so area 7's `_extra_state` refit issue (an Automodel custom model) does not arise.
BACKEND: list[str] = []
LORA_OVERRIDES = [
    "policy.automodel_cfg.lora_cfg.enabled=true",
    f"policy.automodel_cfg.lora_cfg.dim={LORA['dim']}",
    f"policy.automodel_cfg.lora_cfg.alpha={LORA['alpha']}",
    # Every linear projection of the decoder blocks (q, k, v, o, gate, up, down: 1.25 M adapter parameters per layer
    # at rank 32, about 35 M in all) but not the tied lm_head, which `match_all_linear` would include (Automodel
    # module_matcher.py:118-124 warns that it breaks downstream tools; vLLM serves the adapter here).
    "policy.automodel_cfg.lora_cfg.match_all_linear=false",
    "policy.automodel_cfg.lora_cfg.target_modules=['*_proj']",
    "policy.automodel_cfg.lora_cfg.use_triton=false",
]
FUSED_ADAM = [
    "policy.optimizer.name=transformer_engine.pytorch.optimizers.fused_adam.FusedAdam",
    "++policy.optimizer.kwargs.master_weights=true",
    "++policy.optimizer.kwargs.store_param_remainders=true",
    "++policy.optimizer.kwargs.exp_avg_dtype=torch.bfloat16",
    "++policy.optimizer.kwargs.exp_avg_sq_dtype=torch.bfloat16",
]
MEMORY = {
    "a": {"max_seq": MAX_SEQ, "vllm": 0.35, "lora": True, "lr": 2.0e-5,
          "extra": ["policy.logprob_chunk_size=512"]},
    "b": {"max_seq": MAX_SEQ, "vllm": 0.4, "lora": True, "lr": 2.0e-5,
          "extra": [*FUSED_ADAM, "policy.logprob_chunk_size=512"]},
    "c": {"max_seq": 1536, "vllm": 0.15, "lora": False, "lr": 2.0e-6,
          "extra": [*FUSED_ADAM, "policy.automodel_cfg.checkpoint.save_consolidated=every"]},
}


def sha256_file(p: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def wait_for_executor(sock: str, deadline_s: float = 900.0) -> dict:
    """The sidecar warms its imports before it listens (executor_server.py `warmup`)."""
    t0 = time.time()
    last = None
    while time.time() - t0 < deadline_s:
        try:
            info = exec_protocol.ping(sock, timeout=10.0)
            res = exec_protocol.call_executor(sock, [{"id": "launcher-check", "cells": ["print(6 * 7)"]}], 60.0)
            if res[0]["status"] != "ok" or res[0]["output"].strip() != "42":
                raise SystemExit(f"executor self-check failed: {res[0]}")
            return info
        except exec_protocol.ExecutorUnavailable as e:
            last = e
            time.sleep(5)
    raise SystemExit(f"executor at {sock} did not answer within {deadline_s:.0f} s: {last}")


def overrides(prof: dict, a, out: Path) -> list[str]:
    mem = MEMORY[a.memory]
    model = str(a.base)
    n = prof["prompts"] * prof["generations"]
    return [
        f"policy.model_name={model}",
        f"policy.tokenizer.name={model}",
        "++policy.tokenizer.chat_template_kwargs.enable_thinking=false",
        "cluster.gpus_per_node=1",
        f"policy.max_total_sequence_length={mem['max_seq']}",
        "data.shuffle=false",                                    # file order: observed logs (consumption.py)
        "data.train.dataset_name=ResponseDataset",
        f"++data.train.data_path={DATA}",
        "++data.train.split_validation_size=0",
        "data.validation=null",
        f"data.default.prompt_file={PROMPT}",
        "data.default.system_prompt_file=null",
        "data.default.processor=math_code_rowid",                # math_code_env.math_code_rowid_processor
        f"data.default.env_name={ENV_ID}",
        f"++env.{ENV_ID}.socket={a.socket}",
        f"++env.{ENV_ID}.log_dir={out}",
        f"++env.{ENV_ID}.max_turns={MAX_TURNS}",
        f"++env.{ENV_ID}.obs_max_chars=512",
        f"++env.{ENV_ID}.timeout_s=900",
        f"grpo.max_rollout_turns={MAX_TURNS}",
        "checkpointing.enabled=true",
        "checkpointing.keep_top_k=null",
        f"checkpointing.checkpoint_dir={out}/nemo/checkpoints",
        f"checkpointing.save_period={prof['save_period']}",
        "checkpointing.save_optimizer=false",                    # cuts need weights only
        f"logger.log_dir={out}/nemo/logs",
        "logger.wandb_enabled=false",
        "logger.tensorboard_enabled=false",
        "policy.automodel_cfg.activation_checkpointing=true",
        *BACKEND,
        *(LORA_OVERRIDES if mem["lora"] else []),
        "policy.sequence_packing.enabled=false",
        "policy.dynamic_batching.enabled=false",
        f"grpo.max_num_steps={prof['steps']}", f"grpo.seed={a.seed}",
        f"grpo.num_prompts_per_step={prof['prompts']}", f"grpo.num_generations_per_prompt={prof['generations']}",
        "grpo.val_period=-1", "grpo.val_at_start=false", "grpo.val_at_end=false",
        f"policy.train_global_batch_size={n}",
        "policy.train_micro_batch_size=1",
        "policy.logprob_batch_size=1",
        f"policy.generation.max_new_tokens={MAX_NEW_TOKENS}",
        # The tool-call stop string must be in the generation config, not only per datum: with
        # `generation.stop_strings` null NeMo RL starts vLLM with skip_tokenizer_init=True
        # (models/generation/__init__.py:227-237), vLLM then has no detokenizer and silently ignores every stop
        # string (vllm v1/engine/detokenizer.py IncrementalDetokenizer.from_new_request: tokenizer None ->
        # no-op detokenizer). P1 p1-a2 (2026-11-15): 7 of 8 replies with `</code>` ran on past it. With the key set,
        # skip_tokenizer_init becomes False and vllm_worker._merge_stop_strings (vllm_worker.py:854-865) adds it.
        f'policy.generation.stop_strings=["{agentic_core.STOP}"]',
        "++policy.generation.vllm_cfg.cap_max_tokens_to_context=true",
        f"policy.generation.vllm_cfg.gpu_memory_utilization={mem['vllm']}",
        "loss_fn.reference_policy_kl_penalty=0.0",
        "++grpo.skip_reference_policy_logprobs_calculation=true",
        f"policy.optimizer.kwargs.lr={mem['lr']}",
        "policy.scheduler.0.kwargs.total_iters=10",              # 10-step warm-up from 0.1x (area 7)
        "policy.scheduler.2.milestones=[10]",
        "data_plane.enabled=false",
        *mem["extra"],
    ]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--profile", choices=tuple(PROFILES), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--base", type=Path, required=True, help="base model directory (models/qwen3-1.7b)")
    p.add_argument("--memory", choices=tuple(MEMORY), required=True)
    p.add_argument("--socket", required=True, help="executor Unix socket (the paired sidecar's spool)")
    p.add_argument("--max-steps", type=int)
    a = p.parse_args(argv)
    if a.memory == "c":
        raise SystemExit("memory rung c predates the one-shot prompt and the 1,024 / 4,096 budget (longest training "
                         "prompt 1,557 tokens > its 1,536 context); re-plan it before use (README 0.11)")
    prof = dict(PROFILES[a.profile])
    if a.max_steps:
        prof["steps"] = a.max_steps
        prof["save_period"] = min(prof["save_period"], a.max_steps)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    attempt = hooks.record_attempt(out, "start", stage="agentic_grpo", profile=a.profile)
    executor = wait_for_executor(a.socket)
    ov = overrides(prof, a, out)
    mem = MEMORY[a.memory]
    run_meta = {"seed": a.seed, "profile": a.profile, "stage": "agentic_grpo", "env_id": ENV_ID,
                "memory_rung": a.memory, "init_checkpoint": None, "lora": dict(LORA) if mem["lora"] else None,
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "examples_consumed": lambda s: s * prof["prompts"]}
    code = {f: sha256_file(HERE / f) for f in ("agentic_core.py", "math_code_env.py", "traj_log.py",
                                               "ptc_agentic_hooks.py", "gen_agentic.py", "executor/exec_protocol.py")}
    hooks.write_run_args(out, {
        "stage": "agentic_grpo", "env_id": ENV_ID, "profile": a.profile, "seed": a.seed, "dry_run": False,
        "max_steps": prof["steps"], "profile_settings": prof, "memory_rung": a.memory, "memory_settings": mem,
        "lora": run_meta["lora"], "max_rollout_turns": MAX_TURNS, "model_name_or_path": str(a.base),
        "base_model": str(a.base), "init_checkpoint": None, "train_file": str(DATA),
        "train_file_sha256": sha256_file(DATA), "prompt_file": PROMPT.name, "prompt_file_sha256": sha256_file(PROMPT),
        "max_new_tokens_per_turn": MAX_NEW_TOKENS, "max_total_sequence_length": mem["max_seq"],
        "code_sha256": code, "executor": {"socket": a.socket, "ping": executor}, "nemo_rl_overrides": ov,
        "attempt": attempt, "script_sha256": run_meta["script_sha256"],
        "seed_derivation": "NeMo RL grpo.seed (set_seed in setup); vLLM sampling seeded by NeMo RL; data order = "
                           "file order (data.shuffle=false); executor replay random.seed(0)"})
    hooks.install_cut_writer(out, a.base, run_meta)

    from nemo_rl.data.processors import register_processor
    from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
    from nemo_rl.environments.utils import register_env
    import math_code_env
    # Ray workers (the environment actor) import math_code_env, agentic_core, traj_log and exec_protocol; the env
    # actor gets the driver's whole os.environ (environments/utils.py:131-136).
    os.environ["PYTHONPATH"] = f"{HERE}:{HERE / 'executor'}:{os.environ.get('PYTHONPATH', '')}"
    register_processor("math_code_rowid", math_code_env.math_code_rowid_processor)
    fqn = "math_code_env.MathCodeEnvironment"
    ACTOR_ENVIRONMENT_REGISTRY[fqn] = PY_EXECUTABLES.SYSTEM
    register_env(ENV_ID, fqn)
    sys.path.insert(0, "/opt/nemo-rl/examples")
    import run_grpo as entry  # noqa: E402
    cfg = "/opt/nemo-rl/examples/configs/grpo_math_1B.yaml"
    sys.argv = [entry.__file__, "--config", cfg, *ov]
    try:
        entry.main()
    finally:
        hooks.record_attempt(out, "end", stage="agentic_grpo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
