"""NeMo RL adapter for area "agentic RL" (multi-turn GRPO of Qwen3-1.7B with LoRA r32 and a Python tool run in an
executor sidecar; area 10, the maintainers' decisions "Area 10 (agentic RL) direction", "Area 10
decisions", "Area 10 tool-use probe and choice" (not published); tests/README.md section 11.27, EXT-248 to
EXT-262).

MM-12: a pure-Python class that imports no framework, torch or pickle loader. It reads JSON, JSONL, the text
`env-freeze.txt` and safetensors headers (standard library), and hashes everything else.

Run layout, read from the area runtime's code `runtime/` (README section 0;
`run_agentic_nemo.py`, `ptc_agentic_hooks.py`, `math_code_env.py`, `agentic_core.py`, `traj_log.py`, `consumption.py`,
`executor/`, `gen_agentic.py`, `score_agentic.py`, `make_data.py`; read 2026-11-15) and its GPU probe p1-a3
(`<output_root>/nemo-rl-agentic-math/p1-a3/train/`, 3 steps at the final settings):

- one stage, `rl.nemo_rl_agentic@0.1` (run_args.json `stage` "agentic_grpo"), from the plain base directory
  `models/qwen3-1.7b` (MODELS.json repo Qwen/Qwen3-1.7B, revision 70d244cc...), LoRA rank 32, alpha 64 on every decoder
  projection (memory rungs a and b; rung c, full weights, is outside the decision and refused);
- the training environment is configuration: run_args.json `env_id` names it (`math_code` now: MATH problems with
  integer answers and a `<code>`/`<result>` Python tool; `ENVS` holds what the logs of each environment may show, and
  the registry row's `envs` maps an environment to its deciding evaluator; NeMo RL's `sliding_puzzle` is known to the
  consumption join only, with no evaluator, so it can never be gated);
- `write_cut` (ptc_agentic_hooks.py) builds `cuts/checkpoint-<N>/` through `cuts/.checkpoint-<N>.partial/` after NeMo RL
  publishes `nemo/checkpoints/step_<N>`: hard links to Automodel's PEFT files (`adapter_model.safetensors`,
  `adapter_config.json`, `automodel_peft_config.json`), copies of NeMo RL's tokenizer files, `sampler.json`
  (gen_agentic.py SAMPLER), `base_ref.json` (MODELS.json repo, revision, files, dir name; `init_checkpoint` null),
  `training_info.json`, then `cut.json` last (schema `ptc_mm.agentic_cut`, kind `lora_adapter`: `files`, `deployable` =
  every file but base_ref.json, training_info.json, automodel_peft_config.json and cut.json, `global_step`, `env_id`,
  `lora`, `memory_rung`, `examples_consumed` = 16 x step, `seed`, `profile`, `stage`, `script_sha256`,
  `nemo_rl_step_dir`);
- consumption (`traj_log.TrajectoryLogger`, environment-agnostic): `env_calls.jsonl`, one line per sample per
  environment call (= per model turn) `{env_id, call, position, traj_call, traj_pos, turn, row_id, reward, terminated,
  end_reason, tool_status, obs_chars, response_sha256, utc}`, the trajectory stamped by its turn-0 call and position;
  `trajectories.jsonl` one line per terminated trajectory; NeMo RL's `nemo/logs/exp_<n>/train_data_step<N>.jsonl`
  (grpo.py:4667-4691) one line per trajectory, every field a batch of one (logger.py:1427-1441): `content` [[prompt,
  assistant, observation, ...]], `rewards` [sum over turns], `input_lengths` [L], `token_ids` / `token_loss_mask`
  [[...]]. The join (as consumption.py, re-implemented here) gives one record per trajectory;
- `run_args.json` (`write_run_args`): `stage`, `env_id`, `profile`, `seed`, `dry_run`, `max_steps`, `profile_settings`,
  `memory_rung`, `memory_settings`, `lora`, `max_rollout_turns`, `model_name_or_path`, `base_model`, `init_checkpoint`,
  `train_file`, `train_file_sha256`, `prompt_file`, `prompt_file_sha256`, `max_new_tokens_per_turn`,
  `max_total_sequence_length`, `code_sha256` (six helper files), `executor` {socket, ping}, `nemo_rl_overrides`,
  `attempt`, `script_sha256`, `framework` {name, commit, versions}; `env-freeze.txt` package lines and
  `# nemo-rl commit <40 hex>`.

Identity (MM-17, MM-18): LoRA. The plain base directory's model hash is the manifest hash of all its files; a cut's
`adapter_hash` covers its cut.json `deployable` files, its `model_hash` is SHA-256 of the canonical JSON {base, adapter},
`base_ref` names the pinned repo and revision and the base's manifest hash.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from pathlib import Path
from typing import Any

from gatedtrain.canonical import canonical_json, entries_hash, file_sha256, manifest_entries, manifest_hash, sha256_bytes, text_lines
from gatedtrain.errors import Problem
from gatedtrain.ext.registry import EVALUATOR_ROWS, TRAINER_ROWS
from gatedtrain_recipe_nemo_rl_agentic_math.rows import (AREA10, AREA10_IMAGE, MATHEXEC_CODE_SHA256, MATHEXEC_VERSIONS,
                                                  SCORE_AGENTIC_SHA256)
from gatedtrain.ext.sdk import (compact_sha256 as _compact_sha256, float32, is_int as _int, is_num as _num, norm_path as _norm_path,
                   parse_options, plain_file as _plain_file, read_cut_marker, read_json as _read_json, read_jsonl,
                   read_run_args, safetensors_problem, sha_or_none as _sha_or_none, shards_concat_sha256)
from gatedtrain.families.nemo_rl import (CHECKPOINT_DIR_RE, CUTS_DIR, CUT_MARKER, EXP_DIR_RE, FREEZE_COMMIT_RE, FREEZE_FILE,
                                 NEMO_LOGS, NEMO_RL_COMMIT, SHA_RE, TRAIN_DATA_RE)

NAME = "nemo_rl_agentic"
ROW = TRAINER_ROWS[NAME]
TRAINER_ID = ROW["uses"][0]
EVALUATOR_ID = ROW["evaluators"][0]
STAGE = "agentic_grpo"
LIBRARY = ("NeMo RL 4f39fd8 (derived image gatedtrain-a6-nemo-rl:1) with Automodel LoRA and the area runtime's cut writer "
           "(ptc_agentic_hooks.write_cut after NeMo RL publishes step_<N>; cut.json written last)")
RUNTIME = AREA10                                                  # the runtime's container path (identity mount)
IMAGE = AREA10_IMAGE
BASE_REL = "models/qwen3-1.7b"
BASE_REPO, BASE_REVISION = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
BASE_INDEX = "model.safetensors.index.json"
BASE_REQUIRED = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", BASE_INDEX)
WEIGHTS_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.safetensors")
CUT_SCHEMA = "ptc_mm.agentic_cut"
CUT_KIND = "lora_adapter"
LORA_WEIGHTS, ADAPTER_CONFIG, PEFT_CONFIG = "adapter_model.safetensors", "adapter_config.json", "automodel_peft_config.json"
# ptc_agentic_hooks.py CUT_REQUIRED_LORA, the optional files write_cut copies when NeMo RL saved them, NOT_DEPLOYABLE.
CUT_REQUIRED = (LORA_WEIGHTS, ADAPTER_CONFIG, "tokenizer.json", "tokenizer_config.json", "sampler.json",
                "base_ref.json")
CUT_OPTIONAL = (PEFT_CONFIG, "training_info.json", "vocab.json", "merges.txt", "chat_template.jinja",
                "special_tokens_map.json", "added_tokens.json")
NOT_DEPLOYABLE = ("base_ref.json", "training_info.json", PEFT_CONFIG)          # besides cut.json
BASE_REF = "base_ref.json"
SAMPLER_FILE = "sampler.json"
LORA = {"dim": 32, "alpha": 64}                                   # run_agentic_nemo.py LORA (decision: LoRA r32)
# The LoRA targets: every decoder projection of the 28 Qwen3 layers, never the tied lm_head (run_agentic_nemo.py
# `target_modules: ['*_proj']`, `match_all_linear=false`; adapter_config.json lists the 196 resolved module names).
LORA_LAYERS = 28
LORA_TARGET_RE = re.compile(r"model\.layers\.(\d+)\.(?:self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj)")
# run_agentic_nemo.py PROFILES (the launcher's), MEMORY (rung c refused: full weights are outside the decision).
PROFILES = {"probe": {"steps": 3, "save_period": 3}, "smoke": {"steps": 5, "save_period": 5},
            "full": {"steps": 60, "save_period": 20}}
LAUNCH_PROFILES = ("smoke", "full")
MEMORY_RUNGS = ("a", "b")
PROMPTS, GENERATIONS = 16, 8
FULL_MAX_STEPS = (60, 100)            # decision "Area 10 decisions" (not published): 60-100 steps (README D5)
SOCKET = "/spool/mathexec.sock"
ENV_CALLS, TRAJECTORIES = "env_calls.jsonl", "trajectories.jsonl"
ENV_LINE_KEYS = ("call", "end_reason", "env_id", "obs_chars", "position", "response_sha256", "reward", "row_id",
                 "terminated", "tool_status", "traj_call", "traj_pos", "turn", "utc")
TRAJ_LINE_KEYS = ("call", "end_reason", "env_id", "response_sha256", "row_id", "traj_call", "traj_pos", "turn", "utc")
LOGGED = ("content", "rewards", "input_lengths", "token_ids", "token_loss_mask")
EOS_IDS = (151645, 151643)            # Qwen3 generation_config.json eos_token_id (consumption.py EOS_IDS)
# The deployment prompt and turn budget (README 0.11): gen_agentic.py SAMPLER, every cut's sampler.json.
PROMPT_FILE = "prompt_math_code_oneshot.txt"
PROMPT_SHA256 = "sha256:3b7145c1edf2526dcd13d5477d343d68e883b30c8e20d1a5c5e3f19af7b6230f"
MAX_NEW_TOKENS, MAX_SEQ = 1024, 4096
CUT_SAMPLER = {
    "chat_template": "base tokenizer chat template, add_generation_prompt=True, enable_thinking=False",
    "executor": "executor/executor_server.py sidecar, stateless replay, limits of executor_server.DEFAULT_LIMITS",
    "include_stop_str_in_output": True, "max_model_len": MAX_SEQ, "max_new_tokens_per_turn": MAX_NEW_TOKENS,
    "max_turns": 4, "messages": "[{role: user, content: <prompt_file>.format(problem)}]",
    "observation": "agentic_core.observation(), raw tokens without special tokens, obs_max_chars 512",
    "prompt_file": PROMPT_FILE, "prompt_file_sha256": PROMPT_SHA256, "samples_per_seed": 1, "stop": ["</code>"],
    "strategy": "greedy", "temperature": 0.0, "top_p": 1.0}
EVAL_ONLY_SAMPLER_KEYS = ("backend", "executor_code_sha256")      # score_agentic.py adds them to the SAMPLER
SAMPLER = {**CUT_SAMPLER, "backend": "vllm", "executor_code_sha256": dict(MATHEXEC_CODE_SHA256)}
# The turn logic, generator and wire protocol shared by training and evaluation (gen_meta.json `code_sha256`, run_args.json
# `code_sha256`), pinned: a gate compares a cut trained and evaluated with exactly this protocol. GEN_CODE_SHA256 is the
# current version; GEN_CODE_SHA256_GATE_2026_11_16 is the one the 2026-11-16 gate run recorded (agentic_core.py before
# an internal document reference was removed from its docstring; the code is the same). The adapter accepts either, the
# same one throughout a run (trainer and both arms' generation), and flags a run of the gate version. A training run of
# the gate version also binds the current agentic_core.py beside its launched script (checks, EXT-83), so the
# published runs verify against the published runtime.
GEN_CODE_SHA256 = {"agentic_core.py": "sha256:8c5028886847b5a19b70e9be07b1c728408e58c9f3647574fc06f6a43235bfc3",
                   "executor/exec_protocol.py": MATHEXEC_CODE_SHA256["exec_protocol.py"],
                   "gen_agentic.py": "sha256:8f32dcd29de7a71b42476d7918623a940c918d65fc2bb49d1b59a1b99f429669"}
GEN_CODE_SHA256_GATE_2026_11_16 = {
    "agentic_core.py": "sha256:c1076f0af2b0c6299e45a6dabccca69d915d76d0131730c497f16d00a77c5847",
    "executor/exec_protocol.py": MATHEXEC_CODE_SHA256["exec_protocol.py"],
    "gen_agentic.py": "sha256:8f32dcd29de7a71b42476d7918623a940c918d65fc2bb49d1b59a1b99f429669"}
GEN_CODE_VERSIONS = {"current": GEN_CODE_SHA256, "gate-2026-11-16": GEN_CODE_SHA256_GATE_2026_11_16}
# executor_server.py DEFAULT_LIMITS and its ping (run_args.json `executor.ping`, gen_meta.json `executor`).
EXECUTOR_PING = {"limits": {"as_mb": 2048, "capture_chars": 16384, "cpu_s": 10, "fsize_mb": 1, "nofile": 64,
                            "nproc_extra": 32, "wall_s": 10.0},
                 "python": "/opt/nemo_rl_venv/bin/python", "workers": 8}
# run_args.json `code_sha256`: the files the launched script and its Ray actor read from its own directory.
HELPERS = ("agentic_core.py", "math_code_env.py", "traj_log.py", "ptc_agentic_hooks.py", "gen_agentic.py",
           "executor/exec_protocol.py")
REQUIRED_OVERRIDES = ("data.shuffle=false",                      # file order: the observed logs assume it
                      "data_plane.enabled=false",                # the classic synchronous grpo_train
                      "policy.automodel_cfg.lora_cfg.enabled=true",
                      f"policy.automodel_cfg.lora_cfg.dim={LORA['dim']}",
                      f"policy.automodel_cfg.lora_cfg.alpha={LORA['alpha']}",
                      "policy.automodel_cfg.lora_cfg.match_all_linear=false",
                      f"policy.max_total_sequence_length={MAX_SEQ}",
                      f"policy.generation.max_new_tokens={MAX_NEW_TOKENS}",
                      "++policy.tokenizer.chat_template_kwargs.enable_thinking=false")

# EXT-253: what each training environment's logs may show. `ends`: the environment's own terminal end reasons and the
# terminal rewards each allows (non-terminal turns reward 0); `tool_statuses`: a non-terminal turn's tool status (None:
# the environment has no tool); `terminal_obs`: whether a terminal turn returns an observation; `max_turns`: the turn
# limit the environment decides (None: the launcher's `grpo.max_rollout_turns`); `stop`: the stop string the launcher
# passes; `actor`: the Ray actor the launcher registers. The loop itself ends a trajectory at the turn limit (end reason
# `max_turns`) or when an observation fills the context (`overflow`, rollouts.py:1180-1196).
ENVS: dict[str, dict[str, Any]] = {
    # agentic_core.MathCodeCore / math_code_env.MathCodeEnvironment (README 0.4, 0.11). MathCodeCore.step verifies every
    # terminal turn whose text has `\boxed`, a tool call at turn 4 included, so `max_turns` may carry 1.0 (full-1: 1 of
    # 138 such ends).
    "math_code": {"actor": "math_code_env.MathCodeEnvironment", "stop": "</code>", "max_turns": 4,
                  "ends": {"final": (0.0, 1.0), "no_answer": (0.0,), "max_turns": (0.0, 1.0)},
                  "tool_statuses": ("ok", "error", "syntax_error", "timeout", "crash"), "terminal_obs": False},
    # NeMo RL's SlidingPuzzleEnv through traj_log.LoggedSlidingPuzzleEnv (sliding_puzzle.py:265-342: reward 1.0 only at
    # the solving move; `max_moves` terminates with an <error> observation; traj_log.sliding_end_reason).
    "sliding_puzzle": {"actor": "traj_log.LoggedSlidingPuzzleEnv", "stop": "</action>", "max_turns": None,
                       "ends": {"solved": (1.0,), "max_moves": (0.0,)}, "tool_statuses": None, "terminal_obs": True},
}
LOOP_ENDS = ("max_turns", "overflow")
TOOL_ERRORS = ("error", "syntax_error", "timeout", "crash")


class PreM2ExecutorWarning(UserWarning):
    """The collected run used the executor of the 2026-11-16 gate, before the M2 hardening (rows.py)."""


class GateGenCodeWarning(UserWarning):
    """The collected run used the turn logic and generator of the 2026-11-16 gate (GEN_CODE_SHA256_GATE_2026_11_16)."""


def gen_code_version(code: Any) -> str | None:
    """The name of the pinned turn-logic and generator version (GEN_CODE_VERSIONS) whose hashes `code` are, else
    None."""
    return next((name for name, want in GEN_CODE_VERSIONS.items() if code == want), None)


def executor_version(code: Any) -> str | None:
    """The name of the pinned executor version (rows.MATHEXEC_VERSIONS) whose code hashes `code` are, else None."""
    return next((name for name, want in MATHEXEC_VERSIONS.items() if code == want), None)


def env_evaluator(env_id: Any) -> str | None:
    """The deciding evaluator of a training environment (registry row `envs`), or None (not gateable)."""
    return ROW["envs"].get(env_id) if isinstance(env_id, str) else None


def evaluator_env(evaluator_id: Any) -> str | None:
    """The training environment an evaluator gates (its registry row's `env_id`), or None."""
    row = EVALUATOR_ROWS.get(evaluator_id) if isinstance(evaluator_id, str) else None
    return row.get("env_id") if isinstance(row, dict) else None


SCRIPT_TRAIN, SCRIPT_GEN, SCRIPT_SCORE = "run_agentic_nemo.py", "gen_agentic.py", "score_agentic.py"
FLAGGED = (SCRIPT_SCORE,)                                        # `python -I score_agentic.py` (standard library only)
SCRIPT_PATHS = {s: f"{RUNTIME}/{s}" for s in (SCRIPT_TRAIN, SCRIPT_GEN, SCRIPT_SCORE)}
# argparse options of the three scripts (run_agentic_nemo.py, gen_agentic.py, score_agentic.py main): option -> takes a
# value.
TRAIN_OPTIONS = {"--profile": True, "--out": True, "--seed": True, "--base": True, "--memory": True, "--socket": True,
                 "--max-steps": True}
GEN_OPTIONS = {"--model": True, "--adapter": True, "--items": True, "--out": True, "--socket": True,
               "--eval-seeds": True, "--limit": True, "--gpu-memory-utilization": True}
SCORE_OPTIONS = {"--gen": True, "--items": True, "--out": True, "--eval-seeds": True, "--limit": True,
                 "--evaluator-id": True, "--model-hash": True, "--base-model-hash": True, "--item-set-hash": True,
                 "--self-test": False}
OPTIONS = {SCRIPT_TRAIN: TRAIN_OPTIONS, SCRIPT_GEN: GEN_OPTIONS, SCRIPT_SCORE: SCORE_OPTIONS}
ITEM_SETS = {"items-test": EVALUATOR_ID}                         # item set -> the evaluator that scores it
EVAL_CUT_CONTAINER = "/inputs/cut"
GEN_CONTAINER = "/inputs/gen"
SPOOL = "/spool"
# gen_agentic.py main: the evaluated model directory's files it hashes into gen_meta.json `model_files`.
GEN_MODEL_SUFFIXES = (".json", ".safetensors", ".txt", ".jinja")
# score_agentic.py: the arm's sub-metrics (finite numbers, MM-1) and the answer parser (copied, compared with the
# runtime's by a test).
SUB_METRICS = ("answer_found", "env_tokens", "model_tokens", "tool_calls", "tool_errors", "tool_timeouts",
               "truncated", "turns_used")
END_REASONS = ("final", "no_answer", "max_turns", "overflow")    # agentic_core.END_REASONS
_BOX = re.compile(r"\\(?:boxed|fbox)\s*\{")
_WRAP = re.compile(r"^\\(?:text|mathrm|mathbf|textbf)\{(.*)\}$", re.S)
_GROUPED = re.compile(r"-?\d{1,3}(?:,\d{3})+")
_INT = re.compile(r"-?\d+(?:\.0+)?")

CHECKPOINT_COMPLETENESS = {
    "library": LIBRARY,
    "rule": ("a cuts/checkpoint-<N> directory is a cut iff its cut.json, written last after an atomic rename from "
             "cuts/.checkpoint-<N>.partial, parses with schema ptc_mm.agentic_cut 1.0, kind lora_adapter, lora {dim 32, "
             "alpha 64}, global_step = N, nemo_rl_step_dir step_<N>, a positive integer examples_consumed, stage "
             "agentic_grpo, profile probe, smoke or full, memory rung a or b, an integer seed, a known env_id, and files "
             "listing exactly the other files of the directory (the LoRA adapter and its configuration, the tokenizer "
             "files, sampler.json, base_ref.json, the optional automodel_peft_config.json and training_info.json), each "
             "with its SHA-256; deployable lists exactly the files but base_ref.json, training_info.json and "
             "automodel_peft_config.json; the adapter weights are a whole safetensors file and adapter_config.json is a "
             "PEFT LORA config with r 32 and lora_alpha 64; NeMo RL's nemo/checkpoints/step_<N> are resume state, never "
             "cuts; the cut's last step included is N"),
    "complete_dir_pattern": f"{CUTS_DIR}/{CHECKPOINT_DIR_RE.pattern}",
    "sources": ["runtime/ptc_agentic_hooks.py write_cut, install_cut_writer (cut.json last, os.rename)",
                "nemo-rl 4f39fd8 nemo_rl/utils/checkpoint_storage.py:80-97 (publish: tmp_step_N -> step_N)",
                "Automodel checkpointing.py:410-421, :1527-1529 (PEFT checkpoint: adapter_model.safetensors, "
                "adapter_config.json)",
                "the GPU probe p1-a3's cut train/cuts/checkpoint-3 (tests/fixtures/agentic/p1-a3)"],
    "verified": True,                    # p1-a3 (3-step GPU probe, 2026-11-15) wrote a cut of this shape
}
SEED_DERIVATION = ("run_agentic_nemo.py passes the launch seed as NeMo RL grpo.seed (set_seed in NeMo RL's setup); the "
                   "data order is the file order (data.shuffle=false), observed per trajectory in env_calls.jsonl; vLLM's "
                   "sampling seed is whatever NeMo RL derives and is not claimed; the executor replays cells with "
                   "random.seed(0). The training seed is run_args.json `seed`, which must equal the launch record's "
                   "training_seed and every cut's cut.json `seed`. Evaluation is greedy (K = 1; eval seed 11 is "
                   "recorded, nothing is drawn); 05 section 14 H(run_seed, step, data_ref, sample_index) is not claimed.")


# --- the scorer's answer parser (score_agentic.py, copied; tests compare it with the runtime's) ---

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


def parse_answer(text: Any) -> int | None:
    """score_agentic.py parse_answer: the integer in the last \\boxed{} of the final turn, or None."""
    if not isinstance(text, str) or not text:
        return None
    cand = last_boxed(text)
    return None if cand is None else normalize_int(cand)


def reference_int(ref: Any) -> int | None:
    return int(ref) if isinstance(ref, str) and re.fullmatch(r"-?\d+", ref) else None


def select_items(items: list, limit: Any) -> list:
    """gen_agentic.py / score_agentic.py select_items: `--limit N` takes N items evenly spaced over the file."""
    if not limit or limit >= len(items):
        return items
    return [items[(i * len(items)) // limit] for i in range(limit)]


def normalized_text(text: str) -> str:
    """make_data.py's overlap normalization: lower case, `$` and whitespace removed."""
    return re.sub(r"\s+", "", text.lower().replace("$", ""))


def _shards(data: Path) -> list[tuple[str, list[dict]]]:
    """[(shard name, rows)] of the spec's training shards in file-name order (the concatenation order of train.jsonl).
    Raises OSError; ValueError when a shard does not parse."""
    from gatedtrain.ledger import shard_table

    out = []
    for name, _ in shard_table(data):
        rows, why = read_jsonl(Path(data) / name)
        if rows is None:
            raise ValueError(f"training shard {name}: {why}")
        out.append((name, rows))
    return out


def _runs(mask: list) -> list[tuple[int, int]]:
    """[(value, length)] run-length encoding of a 0/1 mask (consumption.py runs)."""
    out: list[list[int]] = []
    for v in mask:
        v = 1 if v else 0
        if out and out[-1][0] == v:
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return [(v, n) for v, n in out]


def _sha_text(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


def _bad_index_weights(d: Path, names: list[str]) -> str | None:
    """The base's weights: whole safetensors files named exactly by the index's weight_map."""
    weights = sorted(n for n in names if n.endswith(".safetensors"))
    if not weights:
        return "no *.safetensors weights file"
    odd = [n for n in weights if not WEIGHTS_RE.fullmatch(n)]
    if odd:
        return f"weights file names {odd} are not plain *.safetensors names"
    idx, why = _read_json(Path(d) / BASE_INDEX)
    wm = idx.get("weight_map") if isinstance(idx, dict) else None
    if not (isinstance(wm, dict) and wm and all(isinstance(v, str) for v in wm.values())):
        return f"{BASE_INDEX} has no weight_map of file names{f' ({why})' if why else ''}"
    if sorted(set(wm.values())) != weights:
        return f"{BASE_INDEX} weight_map names {sorted(set(wm.values()))}, not the weights files {weights}"
    for n in weights:
        bad = safetensors_problem(Path(d) / n)
        if bad:
            return bad
    return None


def lora_config_problem(cfg: Any, base_model: Any = None) -> str | None:
    """adapter_config.json of a cut: HF PEFT LORA, r 32, lora_alpha 64, no DoRA, and target_modules exactly the seven
    projections of each of the 28 decoder layers (196 names, never lm_head); `base_model_name_or_path` the run's base
    when given."""
    if not isinstance(cfg, dict):
        return f"{ADAPTER_CONFIG} is not a JSON object"
    got = {k: cfg.get(k) for k in ("peft_type", "r", "lora_alpha")}
    if got != {"peft_type": "LORA", "r": LORA["dim"], "lora_alpha": LORA["alpha"]} or isinstance(cfg.get("r"), bool):
        return f"{ADAPTER_CONFIG} is {got}, not peft_type LORA, r {LORA['dim']}, lora_alpha {LORA['alpha']}"
    if cfg.get("use_dora") not in (None, False):
        return f"{ADAPTER_CONFIG} use_dora is {cfg.get('use_dora')!r}"
    tm = cfg.get("target_modules")
    if not (isinstance(tm, list) and all(isinstance(t, str) for t in tm)):
        return f"{ADAPTER_CONFIG} target_modules is not a list of module names"
    layers = [m for t in tm if (m := LORA_TARGET_RE.fullmatch(t))]
    if len(layers) != len(tm) or len(set(tm)) != len(tm) or len(tm) != 7 * LORA_LAYERS \
            or {int(m.group(1)) for m in layers} != set(range(LORA_LAYERS)):
        odd = [t for t in tm if not LORA_TARGET_RE.fullmatch(t)][:3]
        return (f"{ADAPTER_CONFIG} target_modules are {len(tm)} names{f' (e.g. {odd})' if odd else ''}, not the q, k, v, o, "
                f"gate, up and down projections of each of the {LORA_LAYERS} decoder layers ('*_proj', never lm_head)")
    if base_model is not None and cfg.get("base_model_name_or_path") != base_model:
        return (f"{ADAPTER_CONFIG} base_model_name_or_path is {cfg.get('base_model_name_or_path')!r}, not the run's base "
                f"{base_model!r}")
    return None


class NemoRlAgenticAdapter:
    """The MM-12 interface for NeMo RL multi-turn GRPO runs of the area runtime (EXT-248 to EXT-262)."""

    name = NAME
    trainer_id = TRAINER_ID
    trainer_ids = (TRAINER_ID,)
    evaluator_ids = tuple(ROW["evaluators"])
    framework_name = ROW["framework"]
    checkpoint_completeness = CHECKPOINT_COMPLETENESS
    seed_derivation = SEED_DERIVATION
    binds_spec = bool(ROW["spec_binding"])
    arm_needs_items_dir = True                          # arm_problems() receives the item set and evaluated model
    wants_evaluator = True                              # EXT-253: checks() receives the gate's evaluator
    consumption_fix = ("collect the run's own env_calls.jsonl, trajectories.jsonl and NeMo RL's train_data_step<N>.jsonl "
                       "as the run wrote them (MM-23, MM-27)")
    payload_source = "a training row's `input` (the problem NeMo RL formats into the prompt; exact overlap, MM-11)"

    def __init__(self) -> None:
        self._base_cache: dict[str, list[dict]] = {}
        self._executors: list[tuple[str, str]] = []     # (where, executor version) seen in this collect
        self._gen_codes: list[tuple[str, str]] = []     # (where, turn-logic and generator version) seen in this collect

    def _gen_code_problem(self, code: Any, where: str) -> str | None:
        """None when `code` is a pinned turn-logic and generator version and the same one as every earlier record of
        this run (the trainer's run_args.json, each arm's generation) named; else why not."""
        version = gen_code_version(code)
        if version is None:
            return (f"{where} code_sha256 {code!r} of the turn logic and generator is none of the pinned versions "
                    f"{sorted(GEN_CODE_VERSIONS)} ({GEN_CODE_SHA256} current)")
        others = [f"{w} ({v})" for w, v in self._gen_codes if v != version]
        self._gen_codes.append((where, version))
        if others:
            return (f"{where} names the {version} turn logic and generator, but {', '.join(others)} named another: "
                    "one run uses one version")
        return None

    def _executor_problem(self, code: Any, where: str) -> str | None:
        """None when `code` is a pinned executor version and the same one as every earlier record of this run (the
        trainer's ping, the registered sampler, each arm's generation) named; else why not."""
        version = executor_version(code)
        if version is None:
            return f"{where} executor code {code!r} is none of the pinned executor versions {sorted(MATHEXEC_VERSIONS)}"
        others = [f"{w} ({v})" for w, v in self._executors if v != version]
        self._executors.append((where, version))
        if others:
            return (f"{where} names the {version} executor, but {', '.join(others)} named another: one run uses one "
                    "executor")
        return None

    def metric(self, evaluator_id: str) -> dict[str, Any]:
        if evaluator_id not in self.evaluator_ids:
            raise KeyError(evaluator_id)
        row = EVALUATOR_ROWS[evaluator_id]
        return {"metric": row["metric"], "direction": row["direction"], "score_range": tuple(row["score_range"]),
                "error_classes": tuple(row["error_classes"]), "score_outcomes": tuple(row["score_outcomes"]),
                "scorer": dict(row["scorer"]), "binary": row["binary"], "tier": row["tier"]}

    # --- the base directory and cuts (MM-17, MM-18, MM-20; EXT-250, EXT-251) ---

    @staticmethod
    def is_cut(d: Path) -> bool:
        """Whether `d` is (meant to be) a LoRA cut: it holds cut.json. The plain base directory never does."""
        return (Path(d) / CUT_MARKER).exists() or (Path(d) / CUT_MARKER).is_symlink()

    def needs_base(self, d: Path) -> bool:
        """ext-arm-fill: a cut's model hash composes the base (MM-18), so `--base` is required."""
        return self.is_cut(d)

    @staticmethod
    def base_dir_problem(d: Path) -> str | None:
        """None when `d` is a Qwen3 directory: config, generation config, tokenizer files, and whole weights named by
        the index."""
        for rel in BASE_REQUIRED:
            if not _plain_file(Path(d) / rel):
                return f"{rel} is missing (not a Qwen3 model directory)"
        try:
            names = sorted(p.name for p in Path(d).iterdir() if _plain_file(p))
        except OSError as exc:
            return f"{Path(d).name} cannot be listed ({exc.__class__.__name__})"
        return _bad_index_weights(Path(d), names)

    def _cut_files_problem(self, d: Path) -> str | None:
        """Review 9 M1 (EXT-154): a file that cannot be read makes the directory incomplete, never an exception."""
        try:
            return self._cut_files_problem_unguarded(d)
        except OSError as exc:
            return (f"{Path(exc.filename).name if exc.filename else Path(d).name} cannot be read "
                    f"({exc.__class__.__name__}: {exc.strerror or exc})")

    def _cut_files_problem_unguarded(self, d: Path) -> str | None:
        doc, why = read_cut_marker(d)
        if doc is None:
            return why
        if (doc.get("schema"), doc.get("schema_version")) != (CUT_SCHEMA, "1.0"):
            return f"{CUT_MARKER} schema is {doc.get('schema')!r} {doc.get('schema_version')!r}, not {CUT_SCHEMA} 1.0"
        if doc.get("kind") != CUT_KIND:
            return (f"{CUT_MARKER} kind is {doc.get('kind')!r}, not {CUT_KIND} (full weights, memory rung c, are outside "
                    "the decision: LoRA r32)")
        if doc.get("lora") != LORA:
            return f"{CUT_MARKER} lora is {doc.get('lora')!r}, not {LORA}"
        files = doc["files"]
        named = set(CUT_REQUIRED) | set(CUT_OPTIONAL)
        unknown = sorted(n for n in files if n not in named)
        if unknown:
            return f"{CUT_MARKER} lists {unknown}, which write_cut never writes (a cut holds {sorted(named)})"
        missing = sorted(set(CUT_REQUIRED) - set(files))
        if missing:
            return f"{CUT_MARKER} does not list {missing}"
        present = sorted(p.name for p in Path(d).iterdir())
        if present != sorted([*files, CUT_MARKER]):
            extra = sorted(set(present) - set(files) - {CUT_MARKER})
            return f"the directory holds {extra or present}, which {CUT_MARKER} does not list"
        for name in sorted(files):
            if not _plain_file(Path(d) / name):
                return f"{name} is not a plain file"
            if file_sha256(Path(d) / name) != files[name]:
                return f"{name} is not the SHA-256 {CUT_MARKER} records"
        deploy = sorted(n for n in files if n not in NOT_DEPLOYABLE)
        dep = doc.get("deployable")
        if not (isinstance(dep, list) and all(isinstance(x, str) for x in dep) and sorted(dep) == deploy):
            return f"{CUT_MARKER} deployable is {dep!r}, not every file but {list(NOT_DEPLOYABLE)}: {deploy}"
        gs = doc.get("global_step")
        if not (_int(gs) and gs >= 1):
            return f"{CUT_MARKER} global_step is {gs!r}, not a positive integer"
        if doc.get("nemo_rl_step_dir") != f"step_{gs}":
            return (f"{CUT_MARKER} nemo_rl_step_dir is {doc.get('nemo_rl_step_dir')!r}, not step_{gs} (the NeMo RL "
                    "checkpoint the cut was built from)")
        if not (_int(doc.get("examples_consumed")) and doc["examples_consumed"] >= 1):
            return f"{CUT_MARKER} examples_consumed is {doc.get('examples_consumed')!r}, not a positive integer"
        if doc.get("stage") != STAGE:
            return f"{CUT_MARKER} stage is {doc.get('stage')!r}, not {STAGE}"
        if doc.get("profile") not in PROFILES:
            return f"{CUT_MARKER} profile is {doc.get('profile')!r}, not one of {list(PROFILES)}"
        if doc.get("memory_rung") not in MEMORY_RUNGS:
            return f"{CUT_MARKER} memory_rung is {doc.get('memory_rung')!r}, not one of {list(MEMORY_RUNGS)} (LoRA rungs)"
        if doc.get("env_id") not in ENVS:
            return f"{CUT_MARKER} env_id is {doc.get('env_id')!r}, not one of the adapter's environments {list(ENVS)}"
        if not _int(doc.get("seed")):
            return f"{CUT_MARKER} seed is {doc.get('seed')!r}, not an integer"
        cfg, why = _read_json(Path(d) / ADAPTER_CONFIG)
        bad = lora_config_problem(cfg) if why is None else why
        if bad:
            return bad
        return safetensors_problem(Path(d) / LORA_WEIGHTS)

    def model_dir_problem(self, d: Path) -> str | None:
        """None when `d` is a complete cut (it holds cut.json) or a plain base model directory."""
        d = Path(d)
        if d.is_symlink() or not d.is_dir():
            return f"{d} is not a plain directory"
        if self.is_cut(d):
            return self._cut_files_problem(d)
        return self.base_dir_problem(d)

    def cut_problem(self, d: Path, step: int) -> str | None:
        d = Path(d)
        if d.is_symlink() or not d.is_dir():
            return f"{d} is not a plain directory"
        bad = self._cut_files_problem(d)
        if bad:
            return bad
        doc, _ = read_cut_marker(d)
        if doc.get("global_step") != step:
            return f"{CUT_MARKER} global_step is {doc.get('global_step')!r}, not {step} (the directory's step)"
        return None

    def checkpoint_dirs(self, stage_out: Path) -> list[tuple[int, Path]]:
        """Every `cuts/checkpoint-<N>` directory, complete or not, in step order; `.checkpoint-<N>.partial` and NeMo RL's
        `nemo/checkpoints/step_<N>` are never one."""
        cuts = Path(stage_out) / CUTS_DIR
        if cuts.is_symlink() or not cuts.is_dir():
            return []
        out = []
        for p in cuts.iterdir():
            m = CHECKPOINT_DIR_RE.match(p.name)
            if m and p.is_dir() and not p.is_symlink():
                out.append((int(m.group(1)), p))
        return sorted(out)

    def find_cuts(self, stage_out: Path) -> tuple[list[tuple[int, Path]], list[Path]]:
        """(complete [(global_step, path)] in step order, incomplete [path]) (EXT-250)."""
        complete, incomplete = [], []
        for step, d in self.checkpoint_dirs(stage_out):
            if self.cut_problem(d, step) is None:
                complete.append((step, d))
            else:
                incomplete.append(d)
        return complete, incomplete

    def _base_entries(self, base: Path) -> list[dict]:
        """The base directory's manifest entries, walked once per adapter instance (4.1 GB on the real base)."""
        key = str(Path(base).resolve())
        if key not in self._base_cache:
            self._base_cache[key] = manifest_entries(Path(base))
        return self._base_cache[key]

    def identity(self, cut: Path, base: Any = None, *, entries: list[dict] | None = None) -> dict[str, Any]:
        """EXT-251 (MM-17, MM-18). The base alone: `model_hash` = `cut_hash` = its manifest hash. A cut: `adapter_hash`
        over its cut.json `deployable` files, `model_hash` = sha256(canonical_json({base, adapter})), `cut_hash` over the
        whole cut, `base_ref` = {repo, revision (base_ref.json), manifest_hash of the base}. Raises ValueError for a cut
        without `base`."""
        is_cut = self.is_cut(cut)
        if entries is None and not is_cut:
            entries = self._base_entries(Path(cut))
        if entries is None:
            entries = manifest_entries(Path(cut))
        if not is_cut:
            self._base_cache.setdefault(str(Path(cut).resolve()), entries)
            h = entries_hash(entries)
            return {"model_hash": h, "cut_hash": h, "components": [e["path"] for e in entries], "base_ref": None}
        if base is None:
            raise ValueError(f"{cut} is a LoRA cut: its model hash composes the base directory's hash (MM-18), so the "
                             "base directory is needed")
        doc, _ = read_cut_marker(cut)
        deploy = set(doc.get("deployable") or []) if doc and isinstance(doc.get("deployable"), list) else set()
        comps = [e for e in entries if e["path"] in deploy]
        adapter = entries_hash(comps)
        base_hash = entries_hash(self._base_entries(Path(base)))
        ref, _ = _read_json(Path(cut) / BASE_REF)
        ref = ref if isinstance(ref, dict) else {}
        return {"model_hash": sha256_bytes(canonical_json({"base": base_hash, "adapter": adapter})),
                "cut_hash": entries_hash(entries), "adapter_hash": adapter, "components": [e["path"] for e in comps],
                "base_ref": {"repo": ref.get("repo"), "revision": ref.get("revision"), "manifest_hash": base_hash}}

    # --- consumption (MM-23, MM-27; EXT-252, EXT-253) ---

    def _plan(self, stage_out: Path, data: Path, cuts: list) -> tuple[list[dict], list[str]]:
        """([one dict per trajectory of the steps up to the last cut], the problems that refuse the logs): the
        environment's per-turn log joined with NeMo RL's step log, trajectory by trajectory (consumption.py's rules,
        re-implemented, plus the call order NeMo RL's synchronous loop guarantees and the environment's own rules)."""
        args, why = read_run_args(stage_out)
        if args is None:
            return [], [why]
        if not cuts:
            return [], ["no complete cut"]
        if args.get("stage") != STAGE:
            return [], [f"run_args.json stage is {args.get('stage')!r}, not {STAGE}"]
        env_id = args.get("env_id")
        spec = ENVS.get(env_id) if isinstance(env_id, str) else None
        if spec is None:
            return [], [f"run_args.json env_id {env_id!r} is not one of the adapter's environments {sorted(ENVS)} "
                        "(registry row `envs`, EXT-253)"]
        prof = args.get("profile_settings") if isinstance(args.get("profile_settings"), dict) else {}
        P, G = prof.get("prompts"), prof.get("generations")
        if not (_int(P) and P > 0 and _int(G) and G > 0):
            return [], [f"run_args.json profile_settings prompts {P!r} and generations {G!r} are not positive integers"]
        max_steps, T = args.get("max_steps"), args.get("max_rollout_turns")
        if not (_int(max_steps) and max_steps >= 1):
            return [], [f"run_args.json max_steps {max_steps!r} is not a positive integer"]
        if not (_int(T) and T >= 1) or (spec["max_turns"] is not None and T != spec["max_turns"]):
            return [], [f"run_args.json max_rollout_turns {T!r} is not "
                        + (f"{spec['max_turns']}, the turn limit of environment {env_id}" if spec["max_turns"]
                           else "a positive integer")]
        try:
            shards = _shards(data)
        except (OSError, ValueError) as exc:
            return [], [f"the spec's training shards cannot be read ({exc})"]
        rows = [(name, r) for name, rs in shards for r in rs]
        ids = [r.get("row_id") for _, r in rows]
        if not rows or not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids):
            return [], ["the training rows have no unique non-empty string row_id each"]
        shard_of = {r["row_id"]: name for name, r in rows}
        spe = len(rows) // P                              # drop_last (NeMo RL's dataloader)
        if spe < 1:
            return [], [f"{len(rows)} training rows do not fill one step of {P} prompts"]
        marks = []
        for _, d in cuts:
            doc, why = read_cut_marker(d)
            if doc is None:
                return [], [f"{Path(d).name}: {why}"]
            marks.append((doc.get("global_step"), doc.get("examples_consumed"), Path(d).name))
        B = P * G
        out: list[str] = []

        def bad(msg: str) -> bool:
            out.append(msg)
            return len(out) >= 8

        env, why = read_jsonl(Path(stage_out) / ENV_CALLS)
        if env is None:
            return [], [why]
        if not env:
            return [], [f"{ENV_CALLS} is empty"]
        # 1. Line shape and the environment's own rules.
        ends_ok, statuses = spec["ends"], spec["tool_statuses"]
        for n, x in enumerate(env, start=1):
            where = f"{ENV_CALLS} line {n}"
            if sorted(x) != sorted(ENV_LINE_KEYS):
                if bad(f"{where}: keys {sorted(x)} are not traj_log.TrajectoryLogger's {sorted(ENV_LINE_KEYS)}"):
                    return [], out
                continue
            ints = [k for k in ("call", "position", "traj_call", "traj_pos", "turn", "obs_chars") if not _int(x[k])]
            if ints or not isinstance(x["terminated"], bool) or not _num(x["reward"]) \
                    or not (isinstance(x["response_sha256"], str) and SHA_RE.fullmatch(x["response_sha256"])):
                if bad(f"{where}: {ints or 'terminated, reward or response_sha256'} malformed"):
                    return [], out
                continue
            if x["env_id"] != env_id:
                bad(f"{where}: env_id {x['env_id']!r} is not run_args.json env_id {env_id!r} (one environment per run)")
            elif x["terminated"]:
                end = x["end_reason"]
                if end not in ends_ok:
                    bad(f"{where}: end_reason {end!r} is not one of environment {env_id}'s {sorted(ends_ok)}")
                elif float(x["reward"]) not in ends_ok[end]:
                    bad(f"{where}: reward {x['reward']!r} at end_reason {end} is not one of {list(ends_ok[end])} "
                        f"(environment {env_id})")
                if x["tool_status"] is not None:
                    bad(f"{where}: a terminal turn with tool_status {x['tool_status']!r} (only a continuing turn ran a "
                        "tool call)")
                if (x["obs_chars"] > 0) != spec["terminal_obs"]:
                    bad(f"{where}: a terminal turn with obs_chars {x['obs_chars']} (environment {env_id} returns "
                        f"{'an' if spec['terminal_obs'] else 'no'} observation when it ends a trajectory)")
            else:
                if x["end_reason"] is not None or x["reward"] != 0:
                    bad(f"{where}: a continuing turn with end_reason {x['end_reason']!r} and reward {x['reward']!r} (the "
                        "reward is paid at the terminal turn only)")
                if (statuses is None and x["tool_status"] is not None) or (
                        statuses is not None and x["tool_status"] not in statuses):
                    bad(f"{where}: tool_status {x['tool_status']!r} is not one of environment {env_id}'s "
                        f"{list(statuses) if statuses else [None]}")
                if x["obs_chars"] < 1:
                    bad(f"{where}: a continuing turn without an observation")
            if len(out) >= 8:
                return [], out
        if out:
            return [], out
        # 2. Calls in order: numbered from 1, positions from 0 within a call; a call either starts the trajectories of a
        #    step (every line in its first turn, stamped with this call) or continues the still active trajectories of
        #    the latest step in their batch order (rollouts.py:1084-1243; traj_log stamps).
        calls: list[list[dict]] = []
        for n, x in enumerate(env, start=1):
            if x["call"] == len(calls) + 1 and x["position"] == 0:
                calls.append([x])
            elif calls and x["call"] == len(calls) and x["position"] == len(calls[-1]):
                calls[-1].append(x)
            elif bad(f"{ENV_CALLS} line {n}: call {x['call']} position {x['position']} does not follow call "
                     f"{len(calls)} position {len(calls[-1]) - 1 if calls else None} (calls numbered from 1, positions "
                     "from 0, in file order)"):
                return [], out
        if out:
            return [], out
        firsts: list[int] = []
        last_lines: dict[tuple[int, int], dict] = {}
        trajs: dict[tuple[int, int], list[dict]] = {}
        for c, lines in enumerate(calls, start=1):
            starts = [x["traj_call"] == c for x in lines]
            if all(starts):
                if any(x["turn"] != 1 or x["traj_pos"] != x["position"] for x in lines):
                    bad(f"call {c} starts trajectories but not every line is a first turn at its own position")
                elif len(lines) != B:
                    bad(f"call {c} starts {len(lines)} trajectories, not prompts x generations = {B}")
                firsts.append(c)
                prev = {}
            elif any(starts):
                bad(f"call {c} mixes new trajectories with continuing ones (NeMo RL starts a step with one call over the "
                    "whole batch)")
            else:
                if not firsts or any(x["traj_call"] != firsts[-1] for x in lines):
                    bad(f"call {c} continues trajectories of another step than the latest ({firsts[-1] if firsts else None})")
                else:
                    tp = [x["traj_pos"] for x in lines]
                    if tp != sorted(set(tp)):
                        bad(f"call {c}: traj_pos {tp[:6]}... are not increasing (NeMo RL keeps the active samples in "
                            "batch order)")
                    for x in lines:
                        p = prev.get(x["traj_pos"])
                        if p is None or p["terminated"] or x["turn"] != p["turn"] + 1:
                            bad(f"call {c} position {x['position']}: trajectory ({x['traj_call']}, {x['traj_pos']}) does "
                                "not continue a non-terminated turn of the previous call")
                            break
            if len(out) >= 8:
                return [], out
            prev = {x["traj_pos"]: x for x in lines}
            for x in lines:
                trajs.setdefault((x["traj_call"], x["traj_pos"]), []).append(x)
                last_lines[(x["traj_call"], x["traj_pos"])] = x
        if out:
            return [], out
        if len(firsts) != max_steps:
            return [], [f"{ENV_CALLS} holds {len(firsts)} steps (calls that start trajectories), not run_args.json "
                        f"max_steps {max_steps}"]
        # 3. trajectories.jsonl: one line per terminated trajectory, its last env line's fields.
        tl, why = read_jsonl(Path(stage_out) / TRAJECTORIES)
        if tl is None:
            return [], [why]
        want = [{k: x[k] for k in TRAJ_LINE_KEYS} for x in env if x["terminated"]]
        if tl != want:
            return [], [f"{TRAJECTORIES} ({len(tl)} lines) is not one line per terminated trajectory in {ENV_CALLS} "
                        f"({len(want)}), with its last line's fields"]
        # 4. NeMo RL's step logs, joined trajectory by trajectory.
        logs = Path(stage_out).joinpath(*NEMO_LOGS)
        try:
            exps = sorted(p for p in logs.iterdir() if p.is_dir() and EXP_DIR_RE.fullmatch(p.name)) if logs.is_dir() else []
        except OSError as exc:
            return [], [f"{'/'.join(NEMO_LOGS)} cannot be listed ({exc.__class__.__name__})"]
        if len(exps) != 1:
            return [], [f"{'/'.join(NEMO_LOGS)} holds {[p.name for p in exps]}, not one exp_<n> directory (a rerun "
                        "starts from an empty output directory)"]
        exp = exps[0]
        try:
            names = {int(m.group(1)): p for p in exp.iterdir() if (m := TRAIN_DATA_RE.fullmatch(p.name))}
        except OSError as exc:
            return [], [f"{exp.name} cannot be listed ({exc.__class__.__name__})"]
        beyond = sorted(k for k in names if not 1 <= k <= max_steps)
        if beyond:
            return [], [f"{exp.name} holds train_data_step{beyond[0]}.jsonl, beyond run_args.json max_steps {max_steps}"]
        last = max((m[0] for m in marks if _int(m[0])), default=0)
        plan: list[dict] = []
        for k, call in enumerate(firsts, start=1):
            fname = f"train_data_step{k}.jsonl"
            if k not in names:
                if bad(f"{exp.name}/{fname} is missing (NeMo RL logs every step, grpo.py:4667-4691)"):
                    return [], out
                continue
            logged, why = read_jsonl(names[k])
            if logged is None:
                if bad(f"{exp.name}/{why}"):
                    return [], out
                continue
            if len(logged) != B:
                if bad(f"{exp.name}/{fname} logs {len(logged)} trajectories, not {B}"):
                    return [], out
                continue
            start = ((k - 1) % spe) * P
            want_rows = [rows[start + g][1]["row_id"] for g in range(P)]
            got_rows = [trajs[(call, pos)][0]["row_id"] for pos in range(B)]
            if got_rows != [rid for rid in want_rows for _ in range(G)]:
                if bad(f"step {k} trajectories have rows {list(dict.fromkeys(got_rows))[:3]}..., not rows {start} .. "
                       f"{start + P - 1} of the training file in file order, each {G} times in a row (data.shuffle=false, "
                       "drop_last, num_generations_per_prompt)"):
                    return [], out
                continue
            for pos in range(B):
                lines = trajs[(call, pos)]
                rec = logged[pos]
                where = f"step {k} trajectory {pos}"
                one = self._join_one(rec, pos, lines, T, where, f"{exp.name}/{fname}")
                if isinstance(one, str):
                    if bad(one):
                        return [], out
                    continue
                if k <= last:
                    plan.append({**one, "step": k, "traj_pos": pos, "member": pos % G, "row_id": lines[0]["row_id"],
                                 "shard": shard_of[lines[0]["row_id"]], "env_id": env_id})
        for gs, consumed, name in marks:                  # MM-27
            if not (_int(gs) and gs <= max_steps):
                bad(f"{name}: global_step {gs!r} is beyond run_args.json max_steps {max_steps}")
            elif consumed != P * gs or not _int(consumed):
                bad(f"{name} cut.json examples_consumed is {consumed!r}, not {P} prompts x {gs}")
        return ([], out) if out else (plan, [])

    @staticmethod
    def _join_one(rec: dict, pos: int, lines: list[dict], max_turns: int, where: str, fname: str) -> dict | str:
        """One trajectory: its env lines (turns 1..T) with its line of train_data_step<k>.jsonl (consumption.build)."""
        if rec.get("idx") != pos or not _int(rec.get("idx")):
            return f"{fname} line {pos + 1}: idx {rec.get('idx')!r} is not {pos}"
        val = {}
        for key in LOGGED:
            v = rec.get(key)
            if not (isinstance(v, list) and len(v) == 1):
                return f"{fname} line {pos + 1}: {key} is not a batch of one (logger.py:1427-1441)"
            val[key] = v[0]
        content, reward, L, ids, mask = (val[k] for k in LOGGED)
        if not (isinstance(content, list) and len(content) >= 3 and len(content) % 2 == 1
                and all(isinstance(c, str) for c in content)):
            return f"{fname} line {pos + 1}: content is not [prompt, (assistant, observation)+]"
        if not _num(reward):
            return f"{fname} line {pos + 1}: rewards {reward!r} is not one finite number"
        if not (_int(L) and isinstance(mask, list) and isinstance(ids, list) and 0 < L <= len(mask)
                and len(ids) == len(mask) and all(v in (0, 1) and not isinstance(v, bool) for v in mask)
                and all(_int(t) for t in ids[:L])):
            return f"{fname} line {pos + 1}: input_lengths / token_ids / token_loss_mask shapes"
        T = len(lines)
        if [x["turn"] for x in lines] != list(range(1, T + 1)):
            return f"{where}: turns {[x['turn'] for x in lines]}"
        if len({x["row_id"] for x in lines}) != 1:
            return f"{where}: row_id changes between turns"
        lastx = lines[-1]
        if lastx["terminated"]:
            end = lastx["end_reason"]
        else:
            end = "max_turns" if T == max_turns else "overflow"
        if len(content) != 2 * T + 1:
            return f"{where}: {fname} content holds {(len(content) - 1) // 2} turns, the environment logged {T}"
        total = sum(float(x["reward"]) for x in lines)
        if float32(total) != float32(reward):
            return (f"{where}: {ENV_CALLS} reward {total!r} is not {fname} rewards {reward!r} (float32): the environment's "
                    "log and NeMo RL's step log do not join")
        if _sha_text("".join(content[1::2])) != lastx["response_sha256"]:
            return f"{where}: the SHA-256 of the logged assistant turns is not the environment's response_sha256"
        obs = [len(c) for c in content[2::2]]
        if obs != [x["obs_chars"] for x in lines]:
            return f"{where}: the logged observations' lengths {obs} are not the environment's obs_chars"
        r = _runs(mask[:L])
        prompt = r[0][1] if r and r[0][0] == 0 else 0
        body = r[1:] if r and r[0][0] == 0 else r
        per_turn, i, at = [], 0, prompt
        while i < len(body):
            v, n = body[i]
            if v != 1:
                return f"{where}: token_loss_mask runs {r[:6]}... do not alternate model and environment tokens"
            envt = body[i + 1][1] if i + 1 < len(body) else 0
            per_turn.append({"model_tokens": n, "env_tokens": envt, "eos": ids[at + n - 1] in EOS_IDS,
                             "tool_status": lines[len(per_turn)]["tool_status"] if len(per_turn) < T else None})
            at += n + envt
            i += 2
        if len(per_turn) != T:
            return f"{where}: {len(per_turn)} runs of model tokens for {T} turns"
        model = sum(t["model_tokens"] for t in per_turn)
        envt = sum(t["env_tokens"] for t in per_turn)
        if prompt + model + envt != L or prompt < 1:
            return f"{where}: prompt, model and environment tokens do not add up to input_lengths {L}"
        return {"turns": T, "per_turn": per_turn, "prompt_tokens": prompt, "model_tokens": model, "env_tokens": envt,
                "reward": total, "end_reason": end, "response_sha256": lastx["response_sha256"]}

    def consumption_problems(self, stage_out: Path, data: Path, cuts: list) -> list[str]:
        """EXT-252: why the observed logs cannot give the per-trajectory consumption up to the last cut."""
        return self._plan(stage_out, data, cuts)[1]

    def consumption(self, stage_out: Path, data: Path, cuts: list, *, run_id: str = "",
                    attempt_id: str = "0") -> dict[str, Any]:
        """Precision `row`, derivation `observed`: one record per trajectory, `rewards` = [its total reward],
        `sample_id` = H(run, attempt, step, row_id, member) (member: the trajectory's place among its row's
        generations in the step), `data_ref.row` = sha256(row_id), shard = the shard holding the row. The ledger record
        has no token fields: the totals of prompt, model and environment tokens go into the reason (EXT-252)."""
        from gatedtrain.ledger import WRITER_ID, check_precision

        plan, bad = self._plan(stage_out, data, cuts)
        if bad:                                           # collect refuses these before writing
            raise ValueError("; ".join(bad))
        check_precision("row", "observed")
        snapshot = manifest_hash(data)
        rows = []
        for t in plan:
            rid = t["row_id"]
            key = (run_id, attempt_id, t["step"], rid, t["member"])
            rows.append({
                "run_id": run_id, "attempt_id": attempt_id, "step": t["step"],
                "sample_id": "sha256:" + hashlib.sha256("\x00".join(str(v) for v in key).encode("utf-8")).hexdigest(),
                "data_ref": {"snapshot": snapshot, "shard": t["shard"], "row": sha256_bytes(rid.encode("utf-8")),
                             "mixture": None, "component": None},
                "parent_ref": None, "outcome": "consumed", "reason": None, "rule_version": None, "gen_version": None,
                "reward_epoch": None, "rewards": [float(t["reward"])], "sampler_config_hash": None, "seed_inputs": None,
                "chunk": None, "precision": "row", "derivation": "observed", "writer_id": WRITER_ID,
            })
        last = max(step for step, _ in cuts)
        ends: dict[str, int] = {}
        for t in plan:
            ends[t["end_reason"]] = ends.get(t["end_reason"], 0) + 1
        tools = sum(1 for t in plan for p in t["per_turn"] if p["tool_status"] is not None)
        tool_eos = sum(1 for t in plan for p in t["per_turn"] if p["tool_status"] is not None and p["eos"])
        env_id = plan[0]["env_id"] if plan else None
        return {"precision": "row", "rows": rows, "derivation": "observed",
                "reason": (f"{ENV_CALLS} (environment {env_id}: one line per trajectory and turn, written by the "
                           f"environment actor) joined trajectory by trajectory with NeMo RL's train_data_step<N>.jsonl "
                           f"(float32 reward, assistant-text SHA-256, token masks; MM-23, MM-27): {len(rows)} records "
                           f"of steps 1 .. {last}, one per trajectory, data_ref.row = sha256(row_id) in the shard "
                           f"holding the row, derivation observed, no trajectory filtered; "
                           f"{sum(t['turns'] for t in plan)} model turns, {tools} executed tool calls ({tool_eos} of "
                           f"them ending at EOS); tokens prompt {sum(t['prompt_tokens'] for t in plan)}, model "
                           f"{sum(t['model_tokens'] for t in plan)}, environment {sum(t['env_tokens'] for t in plan)} "
                           f"(per trajectory in the run's logs: the consumption record has no token fields); end "
                           f"reasons {dict(sorted(ends.items()))}")}

    def extra_read_outputs(self, stage_out: Path) -> list[Path]:
        """EXT-64: NeMo RL's step logs, trajectories.jsonl, attempts.jsonl and cuts.jsonl, which collect reads besides
        READ_OUTPUTS; each must postdate the launch."""
        import os

        logs = Path(stage_out).joinpath(*NEMO_LOGS)
        out = [Path(stage_out) / n for n in (TRAJECTORIES, "attempts.jsonl", "cuts.jsonl")
               if os.path.lexists(Path(stage_out) / n)]
        try:
            for exp in sorted(logs.iterdir()) if logs.is_dir() else []:
                if exp.is_dir() and EXP_DIR_RE.fullmatch(exp.name):
                    out += sorted(p for p in exp.iterdir() if TRAIN_DATA_RE.fullmatch(p.name))
        except OSError:
            pass                                          # the consumption check reports an unlistable directory
        return out

    # --- item sets and exact overlap (MM-11; EXT-256) ---

    @staticmethod
    def _training_rows(data: Path) -> list[tuple[str, int, dict]]:
        return [(name, n, r) for name, rs in _shards(data) for n, r in enumerate(rs, start=1)]

    def training_payloads(self, stage_out: Path, base: Path, cuts: list, *,
                          data: Path | None = None) -> tuple[set[str] | None, str | None]:
        """EXT-256 (MM-11): the exact-overlap keys of the spec's training rows, their `input` (the problem NeMo RL formats
        into the prompt); an item overlaps when its `text` is one of them. (None, why) when the shards cannot be read."""
        if data is None:
            return None, "the spec's training data is not readable"
        try:
            return {r["input"] for _, _, r in self._training_rows(data) if isinstance(r.get("input"), str)}, None
        except (OSError, ValueError) as exc:
            return None, str(exc)

    def item_overlap_problems(self, item_rows: list[dict], data: Path) -> list[str]:
        """EXT-256: the keys are well formed on both sides (a training row's `input` is its user message, an item's
        `text` is its `problem`, the references are integers) and no item equals a training problem after make_data.py's
        normalization (lower case, `$` and whitespace removed), whatever `allow_exact_overlap` says. Problems (each
        refuses); OSError propagates."""
        out: list[str] = []
        try:
            rows = self._training_rows(data)
        except ValueError as exc:
            return [str(exc)]
        norm = set()
        for name, n, r in rows:
            msgs = r.get("messages")
            user = msgs[0].get("content") if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict) else None
            if not (isinstance(r.get("input"), str) and r["input"] and r["input"] == user):
                out.append(f"training row {name} line {n} ({r.get('row_id')!r}): `input` is not its user message (the "
                           "problem NeMo RL formats into the prompt, the exact-overlap key)")
            elif reference_int(r.get("answer")) is None:
                out.append(f"training row {name} line {n} ({r.get('row_id')!r}): answer {r.get('answer')!r} is not an "
                           "integer (make_data.py integer rule)")
            else:
                norm.add(normalized_text(r["input"]))
        for r in item_rows:
            if not (r.get("text") == r.get("problem") and isinstance(r.get("text"), str) and r["text"]):
                out.append(f"item {r.get('item')!r}: `text` is not its `problem` (the exact-overlap key)")
            elif reference_int(r.get("answer")) is None:
                out.append(f"item {r.get('item')!r}: answer {r.get('answer')!r} is not an integer (the scorer would "
                           "report scorer_error)")
            elif normalized_text(r["text"]) in norm:
                out.append(f"item {r.get('item')!r} equals a training problem after normalization (lower case, `$` and "
                           "whitespace removed; make_data.py requires 0)")
        return out[:8] + ([f"{len(out) - 8} more payload problems"] if len(out) > 8 else [])

    # --- arms (EXT-257) ---

    def arm_problems(self, arm: Path, records: list[dict], item_rows: list[dict], items_dir: Path | None = None,
                     model_dir: Path | None = None) -> list[str]:
        """EXT-257 (score_agentic.py sha256 839de6f2...). Per record of a registered item: its output
        (`outputs/<item>__r<k>.json`) names the record's item, repeat and seed and the item's problem and reference; a
        scored record has exactly the score_agentic.py sub-metrics, and the adapter re-derives them, the parsed answer,
        the outcome and the score from the output's transcript and response (copied parser); an error record has the
        shape its class implies. Beside the arm (the scoring output root): `env-runtime.json` names the registered
        items.jsonl, the pinned scorer and the SHA-256 of `gen_meta.json`, which is there too (`_gen_meta_problems`)."""
        arm = Path(arm)
        out: list[str] = []
        meta, _ = _read_json(arm / "eval_meta.json")
        meta = meta if isinstance(meta, dict) else {}
        by_item = {r.get("item"): r for r in item_rows}
        for rec in records:
            it = by_item.get(rec.get("item"))
            if it is None:
                continue                                  # an unknown item: the gate reports it
            ref = rec.get("output_ref")
            if not (isinstance(ref, dict) and isinstance(ref.get("path"), str)):
                out.append(f"item {rec.get('item')!r} has no output_ref (score_agentic.py writes one per (item, seed))")
                continue
            out += self._record_problems(arm, rec, ref["path"], it)
            if len(out) >= 8:
                return out[:8]
        root = arm.parent
        rt, why = _read_json(root / "env-runtime.json")
        if not isinstance(rt, dict):
            out.append(f"the scoring run's env-runtime.json beside the arm: {why or 'not an object'} (score_agentic.py "
                       "writes it next to arm-exact_match/)")
            return out
        want_items = _sha_or_none(Path(items_dir) / "items.jsonl") if items_dir is not None else None
        if items_dir is not None and rt.get("items_sha256") != want_items:
            out.append(f"env-runtime.json items_sha256 {rt.get('items_sha256')!r} is not the registered item set's "
                       f"items.jsonl {want_items}: the scoring run read another item set")
        code = rt.get("code_sha256") if isinstance(rt.get("code_sha256"), dict) else {}
        scorer = meta.get("scorer") if isinstance(meta.get("scorer"), dict) else {}
        if code != {"score_agentic.py": SCORE_AGENTIC_SHA256}:
            out.append(f"env-runtime.json code_sha256 {code!r} is not the pinned score_agentic.py {SCORE_AGENTIC_SHA256}")
        elif rt.get("scorer_code_sha256") != _compact_sha256(code):
            out.append("env-runtime.json scorer_code_sha256 is not the hash of its code_sha256")
        if rt.get("scorer_code_sha256") != scorer.get("code_sha256"):
            out.append(f"env-runtime.json scorer_code_sha256 {rt.get('scorer_code_sha256')!r} is not the arm's scorer "
                       f"code_sha256 {scorer.get('code_sha256')!r}")
        out += self._gen_meta_problems(root, rt, meta, item_rows, want_items if items_dir is not None else None,
                                       model_dir)
        return out

    @staticmethod
    def _record_problems(arm: Path, rec: dict, path: str, it: dict) -> list[str]:
        where = f"output {path}"
        doc, why = _read_json(arm / path)
        if not isinstance(doc, dict):
            return [f"{where}: {why or 'not a JSON object'}"]
        if (doc.get("item"), doc.get("repeat"), doc.get("seed")) != (rec.get("item"), rec.get("repeat"), rec.get("seed")):
            return [f"{where} names item {doc.get('item')!r} repeat {doc.get('repeat')!r} seed {doc.get('seed')!r}, not "
                    f"the record's {rec.get('item')!r} {rec.get('repeat')!r} {rec.get('seed')!r}"]
        if doc.get("problem") != it.get("problem") or doc.get("reference") != it.get("answer"):
            return [f"{where}: problem or reference is not the registered item's"]
        ref = reference_int(it.get("answer"))
        err = rec.get("error")
        if err is not None:
            cls = err.get("class") if isinstance(err, dict) else None
            if doc.get("error") != err:
                return [f"{where}: error {doc.get('error')!r} is not the arm line's {err!r}"]
            if cls in ("missing_output", "scorer_error") and (doc.get("response") is not None
                                                              or doc.get("transcript") is not None):
                return [f"{where}: a {cls} record whose output has a generation"]
            if cls == "sandbox_failure" and not isinstance(doc.get("transcript"), list):
                return [f"{where}: a sandbox_failure record without the generation's transcript"]
            return []
        if ref is None:
            return [f"{where}: scored, but the reference {it.get('answer')!r} is not an integer (a scorer_error)"]
        sub = rec.get("sub_metrics")
        if not (isinstance(sub, dict) and sorted(sub) == sorted(SUB_METRICS)):
            return [f"{where}: sub_metrics {sorted(sub) if isinstance(sub, dict) else sub!r} are not exactly "
                    f"{list(SUB_METRICS)} (score_agentic.py)"]
        if not all(_int(sub[k]) and sub[k] >= 0 for k in SUB_METRICS) or sub["answer_found"] not in (0, 1) \
                or sub["truncated"] not in (0, 1):
            return [f"{where}: sub_metrics are not counts (answer_found and truncated 0 or 1)"]
        if doc.get("score") != rec.get("score") or doc.get("sub_metrics") != sub:
            return [f"{where}: score or sub_metrics differ from the arm's line"]
        turns, end = doc.get("transcript"), doc.get("end_reason")
        if not (isinstance(turns, list) and 1 <= len(turns) <= CUT_SAMPLER["max_turns"]
                and all(isinstance(t, dict) and isinstance(t.get("text"), str) and _int(t.get("n_tokens"))
                        and t["n_tokens"] >= 1 for t in turns)):
            return [f"{where}: transcript is not 1 to {CUT_SAMPLER['max_turns']} turns of generated text"]
        if end not in END_REASONS:
            return [f"{where}: end_reason {end!r} is not one of {list(END_REASONS)} (agentic_core.END_REASONS)"]
        statuses = [t.get("tool_status") for t in turns]
        if any(s is None for s in statuses[:-1]) or (end in ("final", "no_answer", "max_turns")
                                                     and statuses[-1] is not None):
            return [f"{where}: tool statuses {statuses} do not fit end_reason {end} (a turn continues only after an "
                    "executed tool call; a final, unanswered or turn-limit turn runs none)"]
        if doc.get("response") != turns[-1]["text"]:
            return [f"{where}: response is not the final turn's text (the scorer reads the final turn only)"]
        answer = parse_answer(doc.get("response"))
        if doc.get("parsed_answer") != answer:
            return [f"{where}: parsed_answer {doc.get('parsed_answer')!r} is not the answer parser's {answer!r} of its "
                    "response"]
        obs_ok = all(_int(t.get("obs_tokens", 0)) and t.get("obs_tokens", 0) >= 0 for t in turns)
        want = {"turns_used": len(turns), "tool_calls": sum(s is not None for s in statuses),
                "tool_errors": sum(s in TOOL_ERRORS for s in statuses),
                "tool_timeouts": sum(s == "timeout" for s in statuses), "answer_found": int(answer is not None),
                "truncated": int(end == "overflow" or any(t.get("finish_reason") == "length" for t in turns)),
                "model_tokens": sum(t["n_tokens"] for t in turns),
                "env_tokens": sum(t.get("obs_tokens", 0) for t in turns) if obs_ok else None}
        diff = sorted(k for k in SUB_METRICS if sub[k] != want[k])
        if diff:
            return [f"{where}: sub_metrics {diff} are not the transcript's ({ {k: want[k] for k in diff} })"]
        outcome = ("correct" if answer == ref else "wrong_answer") if answer is not None else \
            (end if end in LOOP_ENDS else "no_answer")
        if doc.get("outcome") != outcome or rec.get("score") != float(outcome == "correct"):
            return [f"{where}: outcome {doc.get('outcome')!r} score {rec.get('score')!r} with parsed answer {answer!r}, "
                    f"reference {ref} and end_reason {end}: the outcome is {outcome} (exact_match is 1 exactly for the "
                    "reference integer; no answer scores 0, MM-4)"]
        return []

    def _gen_meta_problems(self, root: Path, rt: dict, meta: dict, item_rows: list[dict], items_sha: str | None,
                           model_dir: Path | None) -> list[str]:
        """EXT-257: gen_meta.json, copied by score_agentic.py into the scoring output root, against the scoring record,
        the arm, the pins (prompt, sampler, executor, turn logic) and the evaluated model (the base, or base plus a LoRA
        cut mounted at /inputs/cut)."""
        from gatedtrain.ext import arm as _arm

        out: list[str] = []
        gm, why = _read_json(root / "gen_meta.json")
        if not isinstance(gm, dict):
            return [f"gen_meta.json beside the arm: {why or 'not an object'} (score_agentic.py copies the generation run's)"]
        if rt.get("gen_meta_sha256") != _sha_or_none(root / "gen_meta.json"):
            out.append(f"gen_meta.json is not the file the scoring run read (env-runtime.json gen_meta_sha256 "
                       f"{rt.get('gen_meta_sha256')!r})")
        if items_sha is not None and gm.get("items_sha256") != items_sha:
            out.append(f"gen_meta.json items_sha256 {gm.get('items_sha256')!r} is not the registered item set's "
                       f"{items_sha}: the generation read another item set")
        want_env = evaluator_env(meta.get("evaluator_id"))
        if gm.get("env_id") != want_env:
            out.append(f"gen_meta.json env_id {gm.get('env_id')!r} is not {want_env!r}, the environment of evaluator "
                       f"{meta.get('evaluator_id')!r} (EXT-253)")
        if (gm.get("prompt_file"), gm.get("prompt_file_sha256")) != (PROMPT_FILE, PROMPT_SHA256):
            out.append(f"gen_meta.json prompt file {gm.get('prompt_file')!r} {gm.get('prompt_file_sha256')!r} is not the "
                       f"deployment prompt {PROMPT_FILE} {PROMPT_SHA256} the training used")
        if gm.get("backend") != "vllm":
            out.append(f"gen_meta.json backend {gm.get('backend')!r} is not vllm (gen_agentic.py)")
        if not _arm.same_json(gm.get("sampler"), CUT_SAMPLER):
            out.append("gen_meta.json sampler is not gen_agentic.py's SAMPLER (prompt, turn budget, stop string)")
        ex = gm.get("executor") if isinstance(gm.get("executor"), dict) else {}
        why_ex = self._executor_problem(ex.get("code_sha256"), f"{root.name}/gen_meta.json")
        if why_ex is not None or ex.get("ok") is not True or any(ex.get(k) != v for k, v in EXECUTOR_PING.items()):
            out.append("gen_meta.json executor is not the pinned executor sidecar (code, limits, workers, interpreter; "
                       f"executor_server.py ping){': ' + why_ex if why_ex else ''}")
        why_gen = self._gen_code_problem(gm.get("code_sha256"), f"{root.name}/gen_meta.json")
        if why_gen is not None:
            out.append(f"gen_meta.json: {why_gen}")
        sampler = gm.get("sampler")
        want = ({**sampler, "backend": gm.get("backend"), "prompt_file_sha256": gm.get("prompt_file_sha256"),
                 "executor_code_sha256": ex.get("code_sha256")} if isinstance(sampler, dict) else None)
        if not _arm.same_json(meta.get("sampler"), want):
            out.append("eval_meta.json sampler is not gen_meta.json's sampler with its backend, prompt file hash and "
                       "executor code")
        if not _arm.same_json(gm.get("eval_seeds"), meta.get("eval_seeds")):
            out.append(f"gen_meta.json eval_seeds {gm.get('eval_seeds')!r} are not the arm's {meta.get('eval_seeds')!r}")
        versions = gm.get("versions") if isinstance(gm.get("versions"), dict) else {}
        if meta.get("framework") != {"name": "vllm", "version": versions.get("vllm")}:
            out.append(f"eval_meta.json framework {meta.get('framework')!r} is not gen_meta.json's vLLM "
                       f"{versions.get('vllm')!r}")
        lim = gm.get("limit")
        if lim is not None and not (_int(lim) and lim > 0):
            out.append(f"gen_meta.json limit {lim!r} is not a positive integer or null")
        elif gm.get("item_ids") != [r.get("item") for r in select_items(item_rows, lim)]:
            out.append("gen_meta.json item_ids are not the registered items " + ("" if lim is None else f"--limit {lim} ")
                       + "selects (gen_agentic.select_items)")
        if gm.get("model_dir") != f"{RUNTIME}/{BASE_REL}":
            out.append(f"gen_meta.json model_dir {gm.get('model_dir')!r} is not the runtime's base {RUNTIME}/{BASE_REL} "
                       "(a LoRA cut is evaluated on the base, --adapter)")
        if model_dir is not None:
            out += self._evaluated_model_problems(gm, Path(model_dir))
        return out

    def _evaluated_model_problems(self, gm: dict, model_dir: Path) -> list[str]:
        out = []
        try:
            if self.is_cut(model_dir):
                doc, why = read_cut_marker(model_dir)
                ref, _ = _read_json(model_dir / BASE_REF)
                files = ref.get("files") if isinstance(ref, dict) and isinstance(ref.get("files"), dict) else {}
                want_files = {n: d.get("sha256") for n, d in sorted(files.items())
                              if isinstance(d, dict) and n.endswith(GEN_MODEL_SUFFIXES)}
                deploy = set((doc or {}).get("deployable") or [])
                want_adapter = {"dir": EVAL_CUT_CONTAINER, "global_step": (doc or {}).get("global_step"),
                                "cut_json_sha256": _sha_or_none(model_dir / CUT_MARKER),
                                "files": {n: h for n, h in sorted(((doc or {}).get("files") or {}).items())
                                          if n in deploy}}
            else:
                want_files = {e["path"]: "sha256:" + e["sha256"] for e in self._base_entries(model_dir)
                              if "/" not in e["path"] and e["path"].endswith(GEN_MODEL_SUFFIXES)}
                want_adapter = None
        except OSError as exc:
            return [f"the evaluated model {model_dir} cannot be hashed ({exc.__class__.__name__})"]
        if gm.get("model_files") != want_files:
            out.append(f"gen_meta.json model_files are not the base files the evaluated model {model_dir} runs on (the arm "
                       "generated with another base)")
        if gm.get("adapter") != want_adapter:
            out.append(f"gen_meta.json adapter {gm.get('adapter')!r} is not the evaluated model's "
                       f"{'LoRA cut' if want_adapter else 'none (the base alone)'} (the arm generated with another model)")
        if gm.get("cut") is not None:
            out.append("gen_meta.json cut is set: a full-weight cut as --model is outside the decision (LoRA r32)")
        return out

    def report_notes(self, records_by_arm: dict[str, list[dict]]) -> list[str]:
        """Decision "Area 10 decisions" (not published): exact match decides; tool use and the other score_agentic.py sub-metrics are
        reported per arm (scored records). A run of the pre-M2 executor raises PreM2ExecutorWarning, one of the gate's
        turn logic and generator GateGenCodeWarning (both shown by the collect command on stderr) instead of adding a
        note, so the 2026-11-16 gate re-collects to the same report."""
        old = sorted({w for w, v in self._executors if v != "current"})
        if old:
            warnings.warn(PreM2ExecutorWarning(
                f"this run used the executor of the 2026-11-16 gate, before the M2 hardening ({', '.join(old)}): a job "
                "could reach the executor's socket and other jobs' results (RESULT.md, executor M2)"), stacklevel=2)
        old_gen = sorted({w for w, v in self._gen_codes if v != "current"})
        if old_gen:
            warnings.warn(GateGenCodeWarning(
                f"this run used the turn logic and generator of the 2026-11-16 gate ({', '.join(old_gen)}): "
                "agentic_core.py before an internal reference was removed from its docstring, the same code "
                "(RESULT.md, rename to ptc-hub 0.2.0)"), stacklevel=2)
        cells = []
        for arm in ("baseline", "candidate"):
            subs = [r["sub_metrics"] for r in records_by_arm.get(arm, []) if isinstance(r.get("sub_metrics"), dict)
                    and r.get("score") is not None]
            if not subs:
                cells.append(f"{arm} n/a")
                continue
            n = len(subs)
            share = lambda k: sum(1 for s in subs if s.get(k, 0) > 0) / n
            mean = lambda k: sum(s.get(k, 0) for s in subs) / n
            cells.append(f"{arm} used-tool share {share('tool_calls'):.3f}, mean tool calls {mean('tool_calls'):.2f}, "
                         f"mean tool errors {mean('tool_errors'):.2f}, answer-found share {share('answer_found'):.3f}, "
                         f"truncated share {share('truncated'):.3f}, mean turns {mean('turns_used'):.2f}, mean model "
                         f"tokens {mean('model_tokens'):.1f}, mean environment tokens {mean('env_tokens'):.1f}")
        return [f"Sub-metrics (not gated; scored records): {'; '.join(cells)}."]

    # --- pins and checks (MM-21, MM-29, MM-33; EXT-253, EXT-254, EXT-255) ---

    def framework_identity(self, stage_out: Path, launch: dict) -> dict[str, Any]:
        """env-freeze.txt names NeMo RL's commit (`# nemo-rl commit <40 hex>`, ptc_agentic_hooks.write_run_args), which is
        the pinned commit and the launch record's framework {name nemo_rl, version = commit = the pin} (MM-33). Raises
        ValueError with the reason (OSError for an unreadable freeze)."""
        freeze = Path(stage_out) / FREEZE_FILE
        if not _plain_file(freeze):
            raise ValueError(f"{freeze} is missing")
        commits = [m.group(1) for line in text_lines(freeze.read_text(encoding="utf-8"))
                   if (m := FREEZE_COMMIT_RE.fullmatch(line.strip()))]
        if commits != [NEMO_RL_COMMIT]:
            raise ValueError(f"{freeze} names NeMo RL commits {commits}, not exactly the pinned {NEMO_RL_COMMIT} (the "
                             "image's /opt/nemo-rl)")
        fw = launch.get("framework") if isinstance(launch.get("framework"), dict) else {}
        if (fw.get("name"), fw.get("version"), fw.get("commit")) != (self.framework_name, NEMO_RL_COMMIT, NEMO_RL_COMMIT):
            raise ValueError(f"{freeze} names NeMo RL {NEMO_RL_COMMIT}, but the launch record claims framework "
                             f"{fw.get('name')!r} version {fw.get('version')!r} commit {fw.get('commit')!r} (NeMo RL is "
                             "pinned by commit: version and commit are the commit)")
        return {"name": self.framework_name, "version": NEMO_RL_COMMIT, "freeze_sha256": file_sha256(freeze)}

    def script_problems(self, stage_out: Path, launch: dict) -> list[Problem]:
        """EXT-83: run_args.json `script_sha256` (run_agentic_nemo.py hashes itself) is the launch record's."""
        args, why = read_run_args(stage_out)
        got = args.get("script_sha256") if isinstance(args, dict) else None
        if got != launch.get("script_sha256"):
            return [Problem("EXT-LAUNCH", "--launch-record.script_sha256", f"run_args.json script_sha256 {got!r} is not "
                            f"the launch record's {launch.get('script_sha256')!r}{f' ({why})' if why else ''} (EXT-83)",
                            "collect the outputs of the run launched with this record's training script")]
        return []

    BASE_FIX = ("train from the runtime's base directory (run_agentic_nemo.py --base <area10-agentic>/models/qwen3-1.7b) "
                "and gate against that directory")
    DATA_FIX = ("train on the runtime's data/train.jsonl (the concatenation of the spec's shards data/shards/) inside "
                "the read-only runtime mount")
    ENV_FIX = ("gate a run of the environment the deciding evaluator scores (a gate binds one environment; another "
               "environment is a new registration with its own evaluator)")

    def mount_binding_problems(self, mounts: Any, *, base: Path, data: Path, base_field: str, data_field: str,
                               flag: str = "--launch-record") -> list[Problem]:
        """`ext-launch --spec` (optional for this adapter): the spec's base directory and its training shards lie inside
        read-only mounts (EXT-254)."""
        from gatedtrain.ext.launch import _ro_table, _within

        table = _ro_table(mounts)
        out = []
        if not any(_within(str(Path(base).resolve()), host) for _, host in table):
            out.append(Problem("EXT-BASE", base_field, f"{flag} mounts no read-only directory that is or holds the "
                               f"spec's base directory {base}", self.BASE_FIX))
        if not any(_within(str(Path(data).resolve()), host) for _, host in table):
            out.append(Problem("EXT-DATA", data_field, f"{flag} mounts no read-only directory that is or holds the "
                               f"spec's training data {data}", self.DATA_FIX))
        return out

    def _base_files_problem(self, ref: dict, base_dir: Path, where: str) -> str | None:
        """None when base_ref.json names the pinned base and its `files` are the base directory's files (MODELS.json
        shape {name: {bytes, sha256}})."""
        if (ref.get("repo"), ref.get("revision")) != (BASE_REPO, BASE_REVISION):
            return (f"{where} names base {ref.get('repo')!r} revision {ref.get('revision')!r}, not {BASE_REPO} @ "
                    f"{BASE_REVISION} (a branch name is refused, MM-18)")
        try:
            entries = self._base_entries(base_dir)
        except (OSError, ValueError):
            return f"the base directory {base_dir} cannot be hashed"
        want = {e["path"]: {"bytes": e["size"], "sha256": "sha256:" + e["sha256"]} for e in entries}
        if ref.get("files") != want:
            return f"{where} files are not the SHA-256 and sizes of the base directory {base_dir} (MM-21)"
        if ref.get("dir_name") != Path(BASE_REL).name:
            return f"{where} dir_name is {ref.get('dir_name')!r}, not {Path(BASE_REL).name}"
        if ref.get("init_checkpoint") is not None:
            return f"{where} init_checkpoint is {ref.get('init_checkpoint')!r}: the run starts from the base"
        return None

    def checks(self, *, last_cut: Path, base: Path, launch: dict, base_field: str, stage_out: Path | None = None,
               data: Path | None = None, data_field: str = "data", evaluator: str | None = None,
               **_: Any) -> list[Problem]:
        """EXT-253 to EXT-255: the environment the gate's evaluator scores, base and data binding, the launch the record
        names, seed, profile, last cut, LoRA settings, script, helper files and pinned turn logic, the executor the run
        talked to, the prompt and turn budget, memory rung and NeMo RL settings."""
        from gatedtrain.ext.launch import container_file

        stage_out = Path(stage_out) if stage_out is not None else Path(last_cut).parent.parent
        args, why = read_run_args(stage_out)
        if args is None:
            return [Problem("EXT-BASE", base_field, f"{why}, so the run cannot be bound to the spec's base (EXT-254)",
                            self.BASE_FIX)]
        out: list[Problem] = []
        mounts = launch.get("mounts")
        if args.get("stage") != STAGE:
            return [Problem("EXT-STAGE", "--stage-out", f"run_args.json stage is {args.get('stage')!r}, not {STAGE} "
                            f"({TRAINER_ID})", "collect a run of run_agentic_nemo.py")]
        cuts = [d for _, d in self.find_cuts(stage_out)[0]] or [Path(last_cut)]
        cut_docs = []
        for d in cuts:
            doc, _ = read_cut_marker(d)
            cut_docs.append((Path(d).name, doc or {}))

        # EXT-253: the environment. A gate binds one environment through its evaluator.
        env_id = args.get("env_id")
        if env_id not in ENVS:
            out.append(Problem("EXT-ENV", "--stage-out", f"run_args.json env_id {env_id!r} is not one of the adapter's "
                               f"environments {sorted(ENVS)}", self.ENV_FIX))
        elif evaluator is not None and evaluator_env(evaluator) != env_id:
            out.append(Problem("EXT-ENV", "stages[1].evaluator", f"the run trained in environment {env_id!r}, but the "
                               f"gate's evaluator {evaluator} scores environment {evaluator_env(evaluator)!r}"
                               + (f" (environment {env_id} has evaluator {env_evaluator(env_id)})" if env_evaluator(env_id)
                                  else f" (environment {env_id} has no registered evaluator: it cannot be gated)"),
                               self.ENV_FIX))
        spec = ENVS.get(env_id) or {}
        for name, doc in cut_docs:
            if doc.get("env_id") != env_id:
                out.append(Problem("EXT-ENV", "--stage-out", f"{name}/cut.json env_id {doc.get('env_id')!r} is not "
                                   f"run_args.json env_id {env_id!r}", "collect the cuts of this run"))

        # MM-21: the base the run loaded, through a read-only mount, and every cut's base_ref.json and adapter config.
        base_model = args.get("base_model")
        hit = container_file(f"{base_model.rstrip('/')}/config.json", mounts) if isinstance(base_model, str) else None
        base_dir = Path(hit[1]).parent if hit else None
        if base_dir is None or base_dir.resolve() != Path(base).resolve():
            out.append(Problem("EXT-BASE", base_field, f"run_args.json base_model {base_model!r} is not, through a "
                               f"read-only mount, the spec's base {base} (MM-21)", self.BASE_FIX))
        if args.get("model_name_or_path") != base_model:
            out.append(Problem("EXT-BASE", base_field, f"run_args.json model_name_or_path "
                               f"{args.get('model_name_or_path')!r} is not its base_model {base_model!r}", self.BASE_FIX))
        if args.get("init_checkpoint") is not None:
            out.append(Problem("EXT-BASE", base_field, f"run_args.json init_checkpoint is "
                               f"{args.get('init_checkpoint')!r}: the run starts from the base", self.BASE_FIX))
        for name, _doc in cut_docs:
            ref, why = _read_json(Path(stage_out) / CUTS_DIR / name / BASE_REF)
            if not isinstance(ref, dict):
                out.append(Problem("EXT-BASE", base_field, f"{name}/{BASE_REF}: {why or 'not an object'}", self.BASE_FIX))
            elif base_dir is not None:
                bad = self._base_files_problem(ref, base_dir, f"{name}/{BASE_REF}")
                if bad:
                    out.append(Problem("EXT-BASE", base_field, bad, self.BASE_FIX))
            cfg, why = _read_json(Path(stage_out) / CUTS_DIR / name / ADAPTER_CONFIG)
            bad = lora_config_problem(cfg, base_model) if why is None else why
            if bad:
                out.append(Problem("EXT-BASE", base_field, f"{name}: {bad}", self.BASE_FIX))

        # EXT-65 analogue: the file the run read is the concatenation of the spec's shards, in the mount.
        if data is not None:
            out += self._data_problems(args, Path(data), mounts, data_field)

        out += self._launch_binding_problems(args, launch)

        # Seed, profile, dry run, steps, memory rung, LoRA, script, helpers, prompt, budget, executor, overrides.
        claimed, field = launch.get("training_seed"), "--launch-record.training_seed"
        seed = args.get("seed")
        if not (_int(seed) and seed == claimed):
            out.append(Problem("EXT-LAUNCH", field, f"the launch record claims training_seed {claimed!r}, but "
                               f"run_args.json `seed` is {seed!r} (MM-29)", "launch with the seed the record names"))
        if args.get("profile") != launch.get("profile"):
            out.append(Problem("EXT-PROFILE", "--launch-record.profile", f"run_args.json `profile` is "
                               f"{args.get('profile')!r}, the launch record's {launch.get('profile')!r}",
                               "collect the run launched with this record's profile"))
        if args.get("dry_run") is not False:
            out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json dry_run is {args.get('dry_run')!r}, not false",
                               "collect a real training run"))
        if not (_int(args.get("attempt")) and args["attempt"] >= 1):
            out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json attempt is {args.get('attempt')!r}, not a "
                               "positive integer", "collect the run as run_agentic_nemo.py wrote it"))
        if args.get("memory_rung") not in MEMORY_RUNGS:
            out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json memory_rung is {args.get('memory_rung')!r}, "
                               f"not one of the LoRA rungs {list(MEMORY_RUNGS)} (rung c, full weights, is outside the "
                               "decision)", "collect a run of memory rung a or b"))
        if args.get("lora") != LORA:
            out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json lora is {args.get('lora')!r}, not {LORA} "
                               "(decision: LoRA r32)", "train with run_agentic_nemo.py's LoRA settings"))
        prof = args.get("profile_settings") if isinstance(args.get("profile_settings"), dict) else {}
        for key, want in (("prompts", PROMPTS), ("generations", GENERATIONS)):
            if not (_int(prof.get(key)) and prof[key] == want):
                out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json profile_settings.{key} is "
                                   f"{prof.get(key)!r}, not {want} (run_agentic_nemo.py PROFILES: 16 prompts x 8 "
                                   "generations per step)", "collect a run of run_agentic_nemo.py's profiles"))
        m = CHECKPOINT_DIR_RE.match(Path(last_cut).name)
        last = int(m.group(1)) if m else None
        if not (_int(args.get("max_steps")) and args["max_steps"] == last and prof.get("steps") == last):
            out.append(Problem("EXT-STAGE", "--stage-out", f"the last cut is step {last}, but run_args.json max_steps is "
                               f"{args.get('max_steps')!r} (profile_settings.steps {prof.get('steps')!r}): the run did "
                               "not save its last step", "collect a run that finished and saved its last step"))
        if args.get("script_sha256") != launch.get("script_sha256"):
            out.append(Problem("EXT-LAUNCH", "--launch-record.script_sha256", f"run_args.json script_sha256 "
                               f"{args.get('script_sha256')!r} is not the launch record's "
                               f"{launch.get('script_sha256')!r} (EXT-83)",
                               "collect the run launched with this record's training script"))
        script = container_file(launch.get("script_path"), mounts) if isinstance(launch.get("script_path"), str) else None
        code = args.get("code_sha256") if isinstance(args.get("code_sha256"), dict) else {}
        if sorted(code) != sorted(HELPERS):
            out.append(Problem("EXT-LAUNCH", "--launch-record.script_path", f"run_args.json code_sha256 names "
                               f"{sorted(code)}, not {sorted(HELPERS)}", "collect the run as run_agentic_nemo.py wrote it"))
        pinned = {k: code.get(k) for k in GEN_CODE_SHA256}
        run_version = gen_code_version(pinned)
        for fname in [*HELPERS, PROMPT_FILE]:
            host = Path(script[1]).parent / fname if script else None
            now = _sha_or_none(host) if host is not None else None
            got = args.get("prompt_file_sha256") if fname == PROMPT_FILE else code.get(fname)
            # A run of a named earlier gen-code version (GEN_CODE_VERSIONS: the same code, documentation changed only)
            # binds the current version's file beside the script: the published runtime carries only that one.
            if now is not None and got != now and run_version is not None and fname in GEN_CODE_SHA256 \
                    and now == GEN_CODE_SHA256[fname]:
                continue
            if now is None or got != now:
                out.append(Problem("EXT-LAUNCH", "--launch-record.script_path", f"run_args.json SHA-256 of {fname} "
                                   f"{got!r} is not that of {fname} beside the launched script ({now!r}): the run read "
                                   "other code or prompt than the record binds (EXT-83)",
                                   "collect the run launched from this runtime directory, unchanged since"))
        why_gen = self._gen_code_problem(pinned, "run_args.json")
        if why_gen is not None:
            out.append(Problem("EXT-LAUNCH", "--launch-record.script_path", f"{why_gen}: training and evaluation would "
                               "decide turns differently", "train with the pinned agentic_core.py, gen_agentic.py "
                               "and executor/exec_protocol.py"))
        settings = {"prompt_file": PROMPT_FILE, "prompt_file_sha256": PROMPT_SHA256,
                    "max_new_tokens_per_turn": MAX_NEW_TOKENS, "max_total_sequence_length": MAX_SEQ}
        for key, want in settings.items():
            if args.get(key) != want:
                out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json {key} is {args.get(key)!r}, not {want!r} "
                                   "(the deployment prompt and turn budget of gen_agentic.py SAMPLER, README 0.11)",
                                   "train with the evaluator's prompt and turn budget"))
        ex = args.get("executor") if isinstance(args.get("executor"), dict) else {}
        ping = ex.get("ping") if isinstance(ex.get("ping"), dict) else {}
        why_ex = self._executor_problem(ping.get("code_sha256"), "run_args.json")
        if ex.get("socket") != SOCKET or ping.get("ok") is not True or why_ex is not None \
                or any(ping.get(k) != v for k, v in EXECUTOR_PING.items()):
            out.append(Problem("EXT-STAGE", "--stage-out", "run_args.json executor is not the pinned executor sidecar at "
                               f"{SOCKET} (code {MATHEXEC_CODE_SHA256}, limits, workers, interpreter; executor_server.py "
                               f"ping){': ' + why_ex if why_ex else ''}",
                               "train paired with the pinned executor sidecar (ext-launch prints it)"))
        fw = args.get("framework") if isinstance(args.get("framework"), dict) else {}
        if fw.get("commit") != NEMO_RL_COMMIT:
            out.append(Problem("EXT-FREEZE", "--stage-out", f"run_args.json framework.commit is {fw.get('commit')!r}, not "
                               f"the pinned NeMo RL {NEMO_RL_COMMIT}", "train in the pinned NeMo RL image"))
        ov = args.get("nemo_rl_overrides")
        turns = args.get("max_rollout_turns")
        if spec and spec["max_turns"] is not None and turns != spec["max_turns"]:
            out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json max_rollout_turns is {turns!r}, not "
                               f"{spec['max_turns']} (environment {env_id})", "train with run_agentic_nemo.py's settings"))
        env_ov = ((f"data.default.env_name={env_id}", f"grpo.max_rollout_turns={turns}",
                   f'policy.generation.stop_strings=["{spec["stop"]}"]') if spec else ())
        for want in (*REQUIRED_OVERRIDES, *env_ov):
            if not (isinstance(ov, list) and want in ov):
                out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json nemo_rl_overrides do not hold {want!r}: the "
                                   "observed logs assume file order, the synchronous grpo_train, LoRA r32, the turn "
                                   "budget, the environment and its stop string (README 0.10: without "
                                   "generation.stop_strings vLLM ignores the stop string)",
                                   "train with run_agentic_nemo.py's overrides"))
        for name, doc in cut_docs:
            for key, want, rule, fld in (("seed", claimed, "EXT-LAUNCH", field),
                                         ("profile", launch.get("profile"), "EXT-PROFILE", "--launch-record.profile"),
                                         ("stage", STAGE, "EXT-STAGE", "--stage-out"),
                                         ("memory_rung", args.get("memory_rung"), "EXT-STAGE", "--stage-out"),
                                         ("script_sha256", launch.get("script_sha256"), "EXT-LAUNCH",
                                          "--launch-record.script_sha256")):
                if doc.get(key) != want or (key == "seed" and not _int(doc.get(key))):
                    out.append(Problem(rule, fld, f"{name}/cut.json {key} is {doc.get(key)!r}, not {want!r}",
                                       "collect the cuts of the run launched with this record"))
        return out

    def _launch_binding_problems(self, args: dict, launch: dict) -> list[Problem]:
        """The record's launch is the run the outputs show: its argv runs run_agentic_nemo.py with `--seed` =
        run_args.json seed, `--profile` = its profile, `--memory` = its memory_rung, `--socket` = its executor socket and,
        when given, `--max-steps` = its max_steps."""
        fix = "collect the run with the launch record ext-launch wrote for it"
        fld = "--launch-record.argv"
        argv = launch.get("argv") if isinstance(launch.get("argv"), list) else []
        script = argv[1].rsplit("/", 1)[-1] if len(argv) > 1 and isinstance(argv[1], str) else None
        if script != SCRIPT_TRAIN:
            return [Problem("EXT-LAUNCH", fld, f"the record's argv[1] {argv[1] if len(argv) > 1 else None!r} is not "
                            f"{SCRIPT_TRAIN}: the outputs are a training run", fix)]
        given = {o: v for o, v, _ in parse_options([a for a in argv[2:] if isinstance(a, str)], TRAIN_OPTIONS)}
        out = []
        ex = args.get("executor") if isinstance(args.get("executor"), dict) else {}
        for opt, got in (("--seed", args.get("seed")), ("--profile", args.get("profile")),
                         ("--memory", args.get("memory_rung")), ("--socket", ex.get("socket"))):
            if given.get(opt) != str(got):
                out.append(Problem("EXT-LAUNCH", fld, f"the record's {opt} {given.get(opt)!r} is not the run's "
                                   f"{got!r} (run_args.json)", fix))
        if "--max-steps" in given:
            if given["--max-steps"] != str(args.get("max_steps")):
                out.append(Problem("EXT-LAUNCH", fld, f"the record's --max-steps {given['--max-steps']!r} is not "
                                   f"run_args.json max_steps {args.get('max_steps')!r}", fix))
        elif args.get("profile") in PROFILES and args.get("max_steps") != PROFILES[args["profile"]]["steps"]:
            out.append(Problem("EXT-LAUNCH", fld, f"the record has no --max-steps, so profile {args['profile']} trains "
                               f"{PROFILES[args['profile']]['steps']} steps, but run_args.json max_steps is "
                               f"{args.get('max_steps')!r}", fix))
        return out

    def _data_problems(self, args: dict, data: Path, mounts: Any, field: str) -> list[Problem]:
        from gatedtrain.ext.launch import container_file

        out = []
        if data.name != "shards":
            return [Problem("EXT-DATA", field, f"the spec's training data {data} is not <runtime>/data/shards",
                            self.DATA_FIX)]
        concat = data.parent / "train.jsonl"
        path = args.get("train_file")
        hit = container_file(path, mounts) if isinstance(path, str) else None
        if hit is None or Path(hit[1]).resolve() != concat.resolve():
            out.append(Problem("EXT-DATA", field, f"run_args.json train_file {path!r} is not, through a read-only mount, "
                               f"{concat} (EXT-254)", self.DATA_FIX))
            return out
        now = _sha_or_none(concat)
        try:
            shards = shards_concat_sha256(data)
        except (OSError, ValueError) as exc:
            return out + [Problem("EXT-DATA", field, f"cannot hash the training shards ({exc.__class__.__name__})",
                                  "keep the training dataset readable")]
        if args.get("train_file_sha256") != now:
            out.append(Problem("EXT-DATA", field, f"run_args.json train_file_sha256 {args.get('train_file_sha256')!r} is "
                               f"not the SHA-256 of {concat} ({now})", self.DATA_FIX))
        if now != shards:
            out.append(Problem("EXT-DATA", field, f"{concat} is not the concatenation of the spec's shards {data} in "
                               "file-name order: the run read other rows than the spec names", self.DATA_FIX))
        return out

    def sampler_problems(self, cuts: list[Path], sampler: Any) -> list[Problem]:
        """EXT-255: every cut's deployment sampler (sampler.json, a component) is the registered sampler without the
        evaluator's keys (backend, executor_code_sha256), and the registered sampler names the pinned prompt and
        executor."""
        from gatedtrain.ext import arm

        out = []
        if not isinstance(sampler, dict):
            return [Problem("EXT-SAMPLER", "registration.sampler", "the registered sampler is not a mapping",
                            "register score_agentic.py's sampler")]
        why_ex = self._executor_problem(sampler.get("executor_code_sha256"), "registration.sampler")
        if why_ex is not None:
            out.append(Problem("EXT-SAMPLER", "registration.sampler.executor_code_sha256", why_ex,
                               "register the evaluation of the area's prompt and executor"))
        for key, want in (("prompt_file_sha256", PROMPT_SHA256), ("backend", "vllm")):
            if not arm.same_json(sampler.get(key), want):
                out.append(Problem("EXT-SAMPLER", f"registration.sampler.{key}", f"the registered {key} "
                                   f"{sampler.get(key)!r} is not {want!r} (the deployment prompt, the pinned executor "
                                   "and vLLM)", "register the evaluation of the area's prompt and executor"))
        want = {k: v for k, v in sampler.items() if k not in EVAL_ONLY_SAMPLER_KEYS}
        for d in cuts:
            doc, _ = _read_json(Path(d) / SAMPLER_FILE)
            if not arm.same_json(doc, want):
                out.append(Problem("EXT-SAMPLER", "registration.sampler", f"{Path(d).name}/{SAMPLER_FILE} is not the "
                                   "registered sampler without backend and executor_code_sha256: the cut deploys another "
                                   "prompt or turn budget than the gate evaluates", "register gen_agentic.py's SAMPLER the "
                                   "cuts carry"))
        return out

    # --- launch side (EXT-258, EXT-259) ---

    @staticmethod
    def _script(argv: Any) -> tuple[str | None, int]:
        """(the script's file name, its argv index): argv[2] after score_agentic.py's `-I`, else argv[1]."""
        if not (isinstance(argv, list) and len(argv) > 1 and all(isinstance(a, str) for a in argv)):
            return None, 1
        i = 2 if (len(argv) > 2 and argv[1] == "-I" and argv[2].rsplit("/", 1)[-1] in FLAGGED) else 1
        return argv[i].rsplit("/", 1)[-1], i

    def argv_problems(self, argv: Any, *, profile: str | None = None) -> list[str]:
        """ext-launch, --recheck and ext-collect. The script is one of the runtime's at its container path:
        run_agentic_nemo.py (`--profile` the launch's, `--out /out`, an integer `--seed`, `--base` the runtime's base,
        `--memory` a or b, `--socket` the spool socket, `--max-steps` a positive integer, 60 to 100 under full),
        gen_agentic.py (`--model` the runtime's base, `--adapter /inputs/cut` or none, `--items items-test`, `--out /out`,
        `--socket` the spool socket, `--eval-seeds` integers, `--gpu-memory-utilization` in (0, 1]) or `-I
        score_agentic.py` (`--gen /inputs/gen`, `--items items-test`, `--out /out`, `--evaluator-id` the item set's
        evaluator, hashes as sha256, no `--self-test`); only score_agentic.py takes `-I`; `--limit` is refused under
        `--profile full`. Options are resolved as argparse resolves them (a unique prefix counts)."""
        if not (isinstance(argv, list) and len(argv) > 1 and all(isinstance(a, str) for a in argv)):
            return []                                     # bind_argv reports the shape
        script, si = self._script(argv)
        if script not in OPTIONS or argv[si] != SCRIPT_PATHS[script]:
            return [f"argv[{si}] {argv[si]!r} is not one of the runtime's scripts {sorted(SCRIPT_PATHS.values())} (a "
                    "copy elsewhere imports other helpers and data)"]
        if (script in FLAGGED) != (si == 2):
            return [f"{script} runs as `python -I {SCRIPT_PATHS[script]}`" if script in FLAGGED else
                    f"{script} follows the interpreter directly"]
        given: dict[str, str | None] = {}
        for opt, value, _ in parse_options(argv[si + 1:], OPTIONS[script]):
            given[opt] = value                            # argparse keeps the last value
        out = []

        def need(opt: str, allowed: tuple, what: str) -> None:
            if given.get(opt) not in allowed:
                out.append(f"{script} {opt} {given.get(opt)!r} is not {what}")

        items = {f"{RUNTIME}/{s}": eid for s, eid in ITEM_SETS.items()}
        base = f"{RUNTIME}/{BASE_REL}"
        if script == SCRIPT_TRAIN:
            need("--profile", (profile,) if profile in LAUNCH_PROFILES else LAUNCH_PROFILES,
                 f"the launch profile {profile!r} (smoke or full; the probe profile is never collected)")
            need("--out", ("/out",), "/out")
            if not re.fullmatch(r"[0-9]+", given.get("--seed") or ""):
                out.append(f"{script} --seed {given.get('--seed')!r} is not an integer")
            need("--base", (base,), f"the runtime's base {base}")
            need("--memory", MEMORY_RUNGS, f"a LoRA memory rung {list(MEMORY_RUNGS)} (rung c is outside the decision)")
            need("--socket", (SOCKET,), f"the executor sidecar's socket {SOCKET}")
            if "--max-steps" in given:
                ms = given["--max-steps"] or ""
                if not re.fullmatch(r"[1-9][0-9]*", ms):
                    out.append(f"{script} --max-steps {given['--max-steps']!r} is not a positive integer")
                elif (profile == "full" or given.get("--profile") == "full") and not (
                        FULL_MAX_STEPS[0] <= int(ms) <= FULL_MAX_STEPS[1]):
                    out.append(f"{script} --max-steps {ms} under --profile full is not in {FULL_MAX_STEPS[0]} .. "
                               f"{FULL_MAX_STEPS[1]} (decision: 60-100 steps)")
        elif script == SCRIPT_GEN:
            need("--model", (base,), f"the runtime's base {base} (a LoRA cut is passed as --adapter {EVAL_CUT_CONTAINER})")
            if "--adapter" in given:
                need("--adapter", (EVAL_CUT_CONTAINER,), EVAL_CUT_CONTAINER)
            need("--items", tuple(items), f"the runtime's item set {', '.join(items)}")
            need("--out", ("/out",), "/out")
            need("--socket", (SOCKET,), f"the executor sidecar's socket {SOCKET}")
            if "--gpu-memory-utilization" in given:
                try:
                    ok = 0.0 < float(given["--gpu-memory-utilization"] or "") <= 1.0
                except ValueError:
                    ok = False
                if not ok:
                    out.append(f"{script} --gpu-memory-utilization {given['--gpu-memory-utilization']!r} is not in (0, 1]")
        else:
            need("--gen", (GEN_CONTAINER,), GEN_CONTAINER)
            need("--items", tuple(items), f"the runtime's item set {', '.join(items)}")
            need("--out", ("/out",), "/out")
            if "--self-test" in given:
                out.append(f"{script} --self-test: the scoring step scores a generation")
            eid = given.get("--evaluator-id", EVALUATOR_ID)
            if given.get("--items") in items and eid != items[given["--items"]]:
                out.append(f"{script} --evaluator-id {eid!r} is not {items[given['--items']]!r}, the evaluator of "
                           f"{given['--items']}")
            for opt in ("--model-hash", "--base-model-hash", "--item-set-hash"):
                if opt in given and not SHA_RE.fullmatch(given[opt] or ""):
                    out.append(f"{script} {opt} {given[opt]!r} is not sha256:<64 hex>")
        if script in (SCRIPT_GEN, SCRIPT_SCORE):
            if "--eval-seeds" in given and not re.fullmatch(r"[0-9]+(,[0-9]+)*", given["--eval-seeds"] or ""):
                out.append(f"{script} --eval-seeds {given['--eval-seeds']!r} is not a comma-separated list of integers")
            if "--limit" in given:
                if not re.fullmatch(r"[1-9][0-9]*", given["--limit"] or ""):
                    out.append(f"{script} --limit {given['--limit']!r} is not a positive integer")
                elif profile == "full":
                    out.append(f"{script} --limit {given['--limit']!r} under --profile full: a gate arm evaluates the whole "
                               "registered item set; --limit is for the smoke arms")
        return out

    def launch_problems(self, doc: dict, *, kind: str) -> list[tuple[str, str]]:
        """[(field, message)] of this adapter's launch rules (request or record): all three scripts run in the pinned
        NeMo RL image (`container`, uid 1000) with network none; training and generation on the GPU, each paired with
        the executor sidecar (EXT-249: model-written code runs only there), scoring on CPU with a tmpfs /tmp and no
        sidecar (it runs no generated code); each script has exactly its mount set (the runtime read-only at its own
        path, the output at /out; training and generation the spool at /spool; generation of a cut adds it at
        /inputs/cut, scoring the generation output at /inputs/gen); the training seed is the record's; the framework is
        NeMo RL pinned by commit."""
        out: list[tuple[str, str]] = []
        argv = doc.get("argv")
        script, si = self._script(argv)
        rt = doc.get("runtime") if isinstance(doc.get("runtime"), dict) else {}
        fw = doc.get("framework") if isinstance(doc.get("framework"), dict) else {}
        if (fw.get("version"), fw.get("commit")) != (NEMO_RL_COMMIT, NEMO_RL_COMMIT):
            out.append(("framework", f"framework version {fw.get('version')!r} and commit {fw.get('commit')!r} are not "
                        f"the pinned NeMo RL commit {NEMO_RL_COMMIT} (pinned by commit)"))
        if script not in OPTIONS:
            return out                                    # bind_argv and argv_problems report the script
        if rt.get("kind") != "container" or rt.get("image") != IMAGE:
            out.append(("runtime", f"{script} runs in the area's NeMo RL image: runtime {{kind: container, image: {IMAGE}}} "
                        f"(gatedtrain-a6-nemo-rl:1, uid 1000 may write NeMo RL's vLLM patch locks), not {rt.get('kind')!r} "
                        f"{rt.get('image')!r}"))
        if doc.get("network") != "none":
            out.append(("network", f"{script} runs with network none (the image holds every input), not "
                        f"{doc.get('network')!r}"))
        given = {o: v for o, v, _ in parse_options(argv[si + 1:], OPTIONS[script])}
        has = "sidecar" in doc
        if script == SCRIPT_SCORE:
            if doc.get("gpus") is not False:
                out.append(("gpus", f"{script} runs on CPU (gpus false)"))
            if doc.get("tmpfs") != ["/tmp"]:
                out.append(("tmpfs", f"tmpfs {doc.get('tmpfs')!r} is not ['/tmp']"))
            if has:
                out.append(("sidecar", f"{script} pairs no sidecar (it runs no generated code)"))
        else:
            if doc.get("gpus") is not True:
                out.append(("gpus", f"{script} needs the GPU (gpus true)"))
            if not has:
                out.append(("sidecar", f"{script} pairs the executor sidecar (decision \"Area 10 decisions\": generated "
                            "code runs only in a separate container): the request names it under `sidecar`, exactly as "
                            "executor/run-executor.sh serve prints it"))
        if script == SCRIPT_TRAIN:
            seed = doc.get("training_seed")
            if _int(seed) and given.get("--seed") != str(seed):
                out.append(("argv", f"{SCRIPT_TRAIN} --seed {given.get('--seed')!r} is not the training_seed {seed} "
                            "(MM-29)"))
        want = [(RUNTIME, "ro"), ("/out", "rw")]
        if script in (SCRIPT_TRAIN, SCRIPT_GEN):
            want.append((SPOOL, "rw"))
        if script == SCRIPT_GEN and "--adapter" in given:
            want.append((EVAL_CUT_CONTAINER, "ro"))
        elif script == SCRIPT_SCORE:
            want.append((GEN_CONTAINER, "ro"))
        mounts = doc.get("mounts") if isinstance(doc.get("mounts"), list) else []
        got = sorted((_norm_path(m.get("container")), m.get("mode")) for m in mounts if isinstance(m, dict))
        if got != sorted(want) or len(got) != len(mounts):
            out.append(("mounts", f"{script} mounts {got}, not exactly {sorted(want)}"))
        return out
