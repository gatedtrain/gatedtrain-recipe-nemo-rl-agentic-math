"""Registry rows of adapter `nemo_rl_agentic`: NeMo RL agentic GRPO with a Python tool (area 10).

The training row, the evaluator rows and the adapter class, loaded by `gatedtrain.ext.registry` through the
`gatedtrain.adapters` entry point (MULTIMODAL-SPEC section 6)."""

from __future__ import annotations

from typing import Any

from gatedtrain.families.nemo_rl import (NEMO_RL_COMMIT, NEMO_RL_PYTHON, NEMO_RL_RUN_OPTIONS, NEMO_RL_UID1000_IMAGE,
                                 VLLM_WORKER_PYTHON)
from gatedtrain.ext.executor import EXECUTOR_ENV, EXECUTOR_LIMITS, EXECUTOR_TMPFS_OPTIONS
from gatedtrain.ext.rowkit import CONTAINER_PATHS, runtime_dir

TRAINER_ROWS: dict[str, dict[str, Any]] = {}
EVALUATOR_ROWS: dict[str, dict[str, Any]] = {}

# Area "agentic RL" (area 10; the maintainers' decisions "Area 10 (agentic RL) direction", "Area 10
# decisions", "Area 10 tool-use probe and choice", 2026-11-15, not published; the runtime `runtime/` (README
# section 0, run_agentic_nemo.py, ptc_agentic_hooks.py, math_code_env.py, agentic_core.py, traj_log.py,
# consumption.py, executor/, gen_agentic.py, score_agentic.py; read 2026-11-15) and its GPU probe p1-a3; gatedtrain's
# tests/README.md section 11.27, EXT-248 to EXT-262). One stage: NeMo RL's synchronous multi-turn GRPO
# (`rl.nemo_rl_agentic@0.1`) of Qwen3-1.7B with LoRA r32 in the derived NeMo RL image; the training environment is
# configuration (`envs`: env id -> its deciding evaluator; a gate binds one environment through its evaluator). Model-
# written code runs only in the paired executor sidecar (the NeMo RL image's own Python, `image_interpreter`, EXT-249),
# for the trainer and for the evaluation's generation step alike. Evaluation is two steps in the same image: greedy
# multi-turn vLLM generation (gen_agentic.py, GPU, with the sidecar) and scoring with the area's own standard-library
# answer parser (score_agentic.py, CPU, `python -I`, no sidecar).
NAME = "nemo_rl_agentic"
# The runtime directory: this repository's runtime/ (scripts, data, models, items, executor/; mounted read-only at its
# own host path, so its container path is its host path), or the override; earlier runs lie under the host's output
# root as `{output_root}/<OUT_NAME>`.
RUNTIME_ENV = "GATEDTRAIN_RECIPE_NEMO_RL_AGENTIC_MATH_RUNTIME"
OUT_NAME = "nemo-rl-agentic-math"
RUNTIME = runtime_dir(RUNTIME_ENV, __file__)
AREA10 = RUNTIME
AREA10_IMAGE = NEMO_RL_UID1000_IMAGE                         # gatedtrain-a6-nemo-rl:1 (uid 1000)
AREA10_EXECUTOR = f"{AREA10}/executor"              # executor_server.py, sandbox_runner.py, exec_protocol.py
# The executor code that runs generated programs, pinned (executor_server.py `ping` reports exactly these SHA-256; the
# trainer's run_args.json `executor.ping` and the generator's gen_meta.json `executor` record them, EXT-254, EXT-257).
# MATHEXEC_CODE_SHA256 is the current executor (M2 hardening: jobs cannot reach the executor's socket, descriptors or
# a sibling job's result); MATHEXEC_CODE_SHA256_GATE_2026_11_16 is the executor before it, which the 2026-11-16 gate
# run used. The adapter accepts either, the same one throughout a run, and flags a run of the pre-M2 executor.
MATHEXEC_CODE_SHA256 = {"exec_protocol.py": "sha256:cbd8df69d462676420e3d4ddc922be50bcc387bc8d57722dbb60b8815b9783dd",
                        "executor_server.py": "sha256:d1f3256d3f18e85b38950247cb2d1b6ae93c1a18a500e5b63d51bce8c6a9a6dc",
                        "sandbox_runner.py": "sha256:4b656e183b51c8604fe09cdd092c6e98fdfbee50a9f5b8a32d4322586cd353f0"}
MATHEXEC_CODE_SHA256_GATE_2026_11_16 = {
    "exec_protocol.py": "sha256:cbd8df69d462676420e3d4ddc922be50bcc387bc8d57722dbb60b8815b9783dd",
    "executor_server.py": "sha256:42971d92b4b215a8ab50abfbb4905ca9df1bf068cc7d613b771806d2fd846e2d",
    "sandbox_runner.py": "sha256:6e1082ec606cdb3dd56a1bb8fbb86429ddbbdc33fd632e2005f9e0272f3b1eaf"}
MATHEXEC_VERSIONS = {"current": MATHEXEC_CODE_SHA256, "gate-2026-11-16 (pre-M2)": MATHEXEC_CODE_SHA256_GATE_2026_11_16}
TRAINER_ROWS[NAME] = {
    "uses": ("rl.nemo_rl_agentic@0.1",),
    # Pinned by NeMo RL's commit (env-freeze.txt `# nemo-rl commit <40 hex>`, ptc_agentic_hooks.write_run_args), as area 7.
    "pins": (),
    "nemo_rl_commit": NEMO_RL_COMMIT,
    "outputs": ("cuts/checkpoint-<step>/ LoRA cuts with cut.json written last (hard links to Automodel's PEFT files)",
                "run_args.json", "env-freeze.txt", "env_calls.jsonl", "trajectories.jsonl", "attempts.jsonl",
                "cuts.jsonl", "nemo/logs/exp_<n>/train_data_step<N>.jsonl"),
    # EXT-250: ptc_agentic_hooks.write_cut `deployable` (every cut file but base_ref.json, training_info.json, cut.json and
    # automodel_peft_config.json): the LoRA adapter, the tokenizer files NeMo RL saved, the deployment sampler. The
    # model hash composes the base directory's manifest hash with these (MM-18).
    "components": ("adapter_model.safetensors", "adapter_config.json", "tokenizer.json", "tokenizer_config.json",
                   "vocab.json", "merges.txt", "chat_template.jinja", "special_tokens_map.json", "added_tokens.json",
                   "sampler.json"),
    # EXT-252: observed per-trajectory log (env_calls.jsonl of the environment actor joined with NeMo RL's step log); a
    # log that does not match the run refuses; the `iteration` fallback is the operator's decision, never automatic.
    "consumption": {"target": "row", "fallback": None},
    "gpu_h_estimate": 4.5,                           # estimate (README "Probe plan": area total 2.6-4.5)
    "evaluators": ("agentic.math_code_items@0.1",),
    # EXT-253: the training environments a run may name (run_args.json `env_id`) and each one's deciding evaluator. A
    # gate binds one environment (decision "Area 10 (agentic RL) direction", not published); an environment without an evaluator is
    # known to the consumption join only (NeMo RL's sliding_puzzle shape, CPU fixtures) and can never be gated.
    "envs": {"math_code": "agentic.math_code_items@0.1"},
    "framework": "nemo_rl",
    # EXT-259: the runtime directory (scripts, model, data, items, executor) and the training and evaluation outputs (an
    # evaluation of a cut mounts it read-only at /inputs/cut, scoring the generation output at /inputs/gen).
    "input_roots": (AREA10, "{output_root}/" + OUT_NAME),
    "spec_binding": False,                           # the smoke launch precedes the registration
    # README section 8 item 5 and the GPU probes (run_probe10.sh): every name, none a credential.
    "env_names": ("FLASHINFER_CUDA_ARCH_LIST", "HF_HUB_OFFLINE", "HOME", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED",
                  "RAY_memory_monitor_refresh_ms", "TORCH_CUDA_ARCH_LIST"),
    "scan_skip": (),
    "interpreter_trees": (),
    "interpreter": NEMO_RL_PYTHON,
    "image_interpreters": {"run_agentic_nemo.py": NEMO_RL_PYTHON, "gen_agentic.py": VLLM_WORKER_PYTHON,
                           "score_agentic.py": NEMO_RL_PYTHON},
    "container_paths": CONTAINER_PATHS,
    "env_paths": ("HOME",),
    "run_options": {"run_agentic_nemo.py": NEMO_RL_RUN_OPTIONS, "gen_agentic.py": NEMO_RL_RUN_OPTIONS},
    "interpreter_flags": {"score_agentic.py": "-I"},
    # EXT-249: the trainer and the evaluation's generation step each pair the executor sidecar, started before and stopped
    # after them; they share only the spool (the Unix socket). Exactly what executor/run-executor.sh serve prints: the
    # NeMo RL image's own interpreter (`image_interpreter`, pinned by the image ID; numpy and sympy for model code), the
    # executor directory read-only at its own path (never the area root, which holds items-test/ and data/ answers), the
    # B18 isolation of area 6 (same tmpfs, limits and environment names).
    "sidecar": {
        "image": AREA10_IMAGE,
        "image_interpreter": NEMO_RL_PYTHON,
        "argv": (NEMO_RL_PYTHON, "-I", f"{AREA10_EXECUTOR}/executor_server.py", "--serve", "/spool/mathexec.sock",
                 "--workers", "8"),
        "ro_mounts": (AREA10_EXECUTOR,),
        "name_prefix": "gatedtrain-mathexec-",              # run-executor.sh serve: gatedtrain-mathexec-<label>
        "spool": "/spool",
        "socket": "/spool/mathexec.sock",
        "env_names": EXECUTOR_ENV,
        "env_paths": ("HOME",),
        "tmpfs": ("/tmp",),
        "tmpfs_options": EXECUTOR_TMPFS_OPTIONS,
        "limits": EXECUTOR_LIMITS,
    },
}

# Area "agentic RL" evaluator (decisions "Area 10 decisions", not published: deciding binary `exact_match`, T3 with the
# construct-alignment and shared-annotation disclosures, N 500, K 1; tool use is a sub-metric). Scorer: score_agentic.py
# (its own `\boxed{}` integer parser, not math_verify, which gives the training reward); code_sha256 is the composite it
# writes (SHA-256 of the compact sorted JSON {"score_agentic.py": <its SHA-256>}) of the version WITHOUT the `env_id` key
# in eval_meta.json (MM-2 refuses unknown eval_meta keys, so the runtime's earlier a3bc32c8... cannot write an arm that
# collects; tests/README.md section 11.27 "Runtime changes"), as renamed for gatedtrain 0.3.0: 839de6f2... is the
# 2026-11-16 gate's a750f5be... with the published name map applied (ptc-hub 0.2.0: f3a9e563...; schema id
# `gatedtrain.item_eval`, nothing else; results/2026-11-16/gate/registration-ptc-hub.yaml and
# registration-gatedtrain.yaml `rename_proof`, RESULT.md); the version the string it writes in the
# image (Python 3.13.14, p1-a3 run_args.json `python`). An item is one MATH test problem with an integer reference;
# its `text` (= `problem`) is the exact-overlap key against the training rows' `input`.
SCORE_AGENTIC_SHA256 = "sha256:839de6f29b2b10f1dc9052563d1f5c31a1b0fcbca63c931e98045685d269a2c1"
SCORE_AGENTIC_CODE_SHA256 = "sha256:3ede3a5a8b1bdced8b7eabd832139ddd3f489446016088e293fd5ac2df64febe"
EVALUATOR_ROWS["agentic.math_code_items@0.1"] = {
    "metric": "exact_match",
    "direction": "higher_is_better",
    "score_range": (0, 1),
    "binary": True,
    "tier": "T3",
    "env_id": "math_code",                           # EXT-253: the training environment this evaluator gates
    # MM-4: evaluator faults (no generation for a registered (item, seed); the executor failed twice for the item, an
    # MM-4 fault of the evaluation, never the policy's; a reference that is not an integer or a malformed generation).
    "error_classes": ("missing_output", "sandbox_failure", "scorer_error"),
    # MM-4: the policy's faults, scored 0 (an integer answer that is not the reference; no answer; no answer and the turn
    # limit or the context ended the trajectory). Code that raised or timed out is part of the transcript.
    "score_outcomes": ("wrong_answer", "no_answer", "max_turns", "overflow"),
    "payload_text_field": "text",
    "item_payload": ("text", "problem", "answer", "subject", "level"),
    "normalizer": False,
    "env_version": False,
    # The evaluator's sampler (score_agentic.py: gen_agentic.py SAMPLER plus the generation backend and the executor's
    # code hashes; SAMPLER already names the prompt file and its hash).
    "sampler_keys": ("backend", "chat_template", "executor", "executor_code_sha256", "include_stop_str_in_output",
                     "max_model_len", "max_new_tokens_per_turn", "max_turns", "messages", "observation", "prompt_file",
                     "prompt_file_sha256", "samples_per_seed", "stop", "strategy", "temperature", "top_p"),
    "framework": "vllm",
    "samples": "500, 1",
    # 05 section 6.1, R15-1: another answer parser (a tool) than the reward's, the same construct.
    "construct_aligned": {
        "reward": "nemo_rl_hf_math_verify_boxed_final_turn",
        "statement": ("exact_match measures final-answer correctness on held-out MATH test problems with integer "
                      "answers, the construct the training reward (NeMo RL's HFVerifyWorker math_verify on the final "
                      "assistant turn, paid only when that turn contains \\boxed) also rewards, with another answer "
                      "parser (score_agentic.py: the last \\boxed{} of the final turn normalized to an integer; not "
                      "math_verify) and the same multi-turn protocol, executor and prompt as training: a pass shows "
                      "that the policy answers more held-out problems correctly under an independent parser, partly "
                      "through the boxed-answer format the reward pays for, not a general gain in mathematical "
                      "reasoning or in tool use (05 section 6.1, R15-1)")},
    # 05 section 6.1, R16-1: the references come from the training answers' annotation process.
    "shared_annotation": {
        "convention": ("MATH competition solutions' final \\boxed{} answers restricted to integers (make_data.py "
                       "integer_answer), the same extraction as the training rows' reference answers"),
        "statement": ("The references of exact_match are held-out hendrycks_math test answers extracted by the same "
                      "integer rule from the same competition corpus as the training answers; no training stage "
                      "consumed them, so the programmatic scorer stays T3 (05 section 6.1, R16-1). Part of a gain may be "
                      "learning that answer convention (one boxed integer): a pass shows more correct integer answers "
                      "on unseen MATH problems, not a general gain in mathematics. MATH is public since 2021 and the "
                      "base was probably trained on it (unverified); the paired design measures the change.")},
    "scorer": {"name": "math_integer_boxed_own_parser", "version": "math_int_boxed_v1; python 3.13.14",
               "weights_sha256": None, "code_sha256": SCORE_AGENTIC_CODE_SHA256},
}

PLUGIN = {"adapters": {NAME: "gatedtrain_recipe_nemo_rl_agentic_math.adapter:NemoRlAgenticAdapter"}, "trainer_rows": TRAINER_ROWS,
          "evaluator_rows": EVALUATOR_ROWS}
