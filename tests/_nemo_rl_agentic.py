"""Area "agentic RL" test helpers (tests/README.md section 11.27, EXT-248 to EXT-262).

Written from the area runtime (`runtime/`: `run_agentic_nemo.py` `PROFILES`,
`MEMORY`, `overrides`, `main`; `ptc_agentic_hooks.py` `write_cut`, `write_run_args`, `record_attempt`; `traj_log.py`
`TrajectoryLogger.record`; `math_code_env.py` `env_step`; `agentic_core.py` `MathCodeCore`; `gen_agentic.py` `SAMPLER`,
`run_trajectories`, `main`; `score_agentic.py` `score_doc`, `main`; `make_data.py` rows and items;
`executor/executor_server.py` ping; read 2026-11-15) and from its real outputs, not from gatedtrain:
- `fixtures/agentic/p1-a3/`: a cut of the 3-step GPU probe `p1-a3/train/` (`cut_probe`, below;
  provenance in its SOURCE.json): per step two prompt groups and two members of each (tool calls ok / error /
  syntax_error, final with reward 0 and 1, no_answer, the env's max_turns), env lines and NeMo RL's step-log lines kept
  byte for byte in their fields but renumbered (calls, positions, traj_call, traj_pos, idx), step-log token arrays cut to
  `input_lengths` (the right padding dropped) and the unused float arrays dropped; run_args.json with profile_settings
  prompts 2 and generations 2; the six training rows in `data/shards/`; `cuts/checkpoint-3/` with its small files copied,
  the 70 MB adapter weights and the tokenizer, vocabulary and merges files replaced by stand-ins and cut.json re-hashed
  (examples_consumed 2 x 3);
- `fixtures/agentic/sliding-puzzle/`: NeMo RL's real SlidingPuzzleEnv driven through the area's
  `traj_log.LoggedSlidingPuzzleEnv` and the area's rollout double (`sliding-puzzle/make_sliding.py.txt`, recorded in SOURCE.json):
  env_calls.jsonl, trajectories.jsonl, train_data_step{1,2}.jsonl and the four puzzle rows;
- `fixtures/agentic/runtime/`: byte copies of the pinned `agentic_core.py`, `gen_agentic.py`,
  `executor/exec_protocol.py` and `prompt_math_code_oneshot.txt` (their SHA-256 are the adapter's pins; never executed);
- `fixtures/agentic/real-arm/`: a scoring run of the proposed `score_agentic.py` (a750f5be..., the runtime's file
  without `env_id` in eval_meta.json; the scorer the 2026-11-16 gate ran, before the rename to ptc-hub 0.2.0, so its
  eval_meta.json keeps the schema id `ptc_skel.item_eval`: `renamed_copy`) on 20 generation records written by
  `gen_doc` for the 20 real items `--limit 20` selects (`real-arm/make_real_arm.py.txt`), and those items.
The fixtures copied from real outputs name the runtime directory by the neutral path `NEUTRAL_RUNTIME` (below)
instead of the host path they were recorded with.

The runtime directory is mounted read-only at its own host path, so the rows name it as a container path. The tests
pin it to a neutral layout under one temporary root `PATHS_ROOT` (below, in the git-ignored `build/`):
`GATEDTRAIN_RECIPE_NEMO_RL_AGENTIC_MATH_RUNTIME` names `<PATHS_ROOT>/recipe/runtime`, set before gatedtrain's registry loads the rows.
The real-file tests read the real runtime (`REAL_RUNTIME`: `runtime/` of this checkout, or the variable's value when set
before the tests start) and the GPU probe (`GATEDTRAIN_RECIPE_NEMO_RL_AGENTIC_MATH_OUTPUTS`, the directory holding `p1-a3/`);
each skips what is absent.

`AgenticScenario(root)` builds the rest from that code: a mirrored host (`GATEDTRAIN_EXT_MIRROR_ROOT` = `<root>/host`
stands for the container's `/`, EXT-112) under `<root>/host/<PATHS_ROOT>/`:
- `recipe/runtime/` the runtime stand-in: script stand-ins (never executed) and the pinned copies, `executor/`,
  `models/qwen3-1.7b/` (the 12 file names of the real base, two weights files named by the index) and
  `models/MODELS.json`, `data/shards/shard-0000{0..3}.jsonl` (48 rows, 12 per shard) and `data/train.jsonl`,
  `items-test/` (40 items);
- `out/nemo-rl-agentic-math/train-1/{train,spool}` a full run (seed 10100, memory rung a, 60 steps of 16 prompts x 8
  generations, 3 steps per epoch, cuts 20, 40, 60; every kind of trajectory: tool calls, errors, the env's max_turns,
  overflow, no answer) and two evaluations `eval-base-1`, `eval-cand-1` (`gen/`, `score/`, `spool/`);
- `<root>/`: spec, registration, trainer.yaml, launch record (with its sidecar) and smoke record.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import yaml

# The neutral layout (module docstring): set before gatedtrain's registry imports the recipe's rows.
RUNTIME_ENV = "GATEDTRAIN_RECIPE_NEMO_RL_AGENTIC_MATH_RUNTIME"
OUTPUTS_ENV = "GATEDTRAIN_RECIPE_NEMO_RL_AGENTIC_MATH_OUTPUTS"
REPO = Path(__file__).resolve().parents[1]
REAL_RUNTIME = Path(os.environ.get(RUNTIME_ENV) or REPO / "runtime")   # the real runtime (real-file tests)
# The real GPU outputs (the probe p1-a3/) for the real-file tests; absent unless OUTPUTS_ENV names their directory.
REAL_OUT = Path(os.environ.get(OUTPUTS_ENV) or "/nonexistent/nemo-rl-agentic-math-outputs")
assert "gatedtrain.ext.registry" not in sys.modules, "import _nemo_rl_agentic before gatedtrain.ext.registry (the rows read the runtime)"
# Not under /tmp: the containers' tmpfs is /tmp, and a mount never nests in it (EXT-88). build/ is git-ignored.
(REPO / "build").mkdir(exist_ok=True)
PATHS_ROOT = tempfile.mkdtemp(prefix="test-paths-", dir=REPO / "build")
atexit.register(shutil.rmtree, PATHS_ROOT, True)
AREA10 = f"{PATHS_ROOT}/recipe/runtime"                      # container path = host path (identity mount)
Path(AREA10).mkdir(parents=True)
os.environ[RUNTIME_ENV] = AREA10

EXT = Path(__file__).resolve().parent / "fixtures"     # the recipe's fixtures (gatedtrain: tests/fixtures/ext)
from gatedtrain.testing.ext import ENV_HOST_WORK, ENV_INPUT_ROOTS, ENV_MIRROR_ROOT, ENV_OUTPUT_ROOT, gpu_step_script, launch_label, safetensors_bytes, sha_of, write_canonical
from gatedtrain.testing.support import canonical_bytes, file_sha256, manifest_entries, manifest_hash, run_cli

AGENTIC = EXT / "agentic"
PROBE = AGENTIC / "p1-a3"
SLIDING = AGENTIC / "sliding-puzzle"
RUNTIME_COPIES = AGENTIC / "runtime"
REAL_ARM = AGENTIC / "real-arm"
# The runtime path the real arm was recorded with (gen_meta.json model_dir is <runtime>/models/qwen3-1.7b).
RECORDED_AREA10 = json.loads((REAL_ARM / "gen_meta.json").read_text(encoding="utf-8"))["model_dir"].rsplit("/models/", 1)[0]


def renamed_copy(path: Path) -> Path:
    """Give the record at `path`, a copy (never a fixture itself), the schema id gatedtrain 0.3.0 reads: the
    `ptc_skel.<name>` that ptc-skel 0.1.0 wrote becomes `gatedtrain.<name>` (text replacement of the `schema` value only;
    the file keeps its layout and every other byte). Returns `path`."""
    text = path.read_text(encoding="utf-8")
    new = re.sub(r'("schema": ")ptc_skel\.', r"\1gatedtrain.", text, count=1)
    assert new != text, f"{path}: no ptc_skel schema id"
    path.write_text(new, encoding="utf-8")
    return path


def localized_real_arm(dst: Path) -> Path:
    """A copy of `real-arm/` at `dst` whose gen_meta.json names the test runtime (`AREA10`) instead of the recorded one,
    env-runtime.json `gen_meta_sha256` following (text replacements: the files keep their layout), and whose arm
    metadata carries the renamed schema id (`renamed_copy`). Returns `dst`."""
    shutil.copytree(REAL_ARM, dst)
    renamed_copy(dst / "arm-exact_match" / "eval_meta.json")
    gm = dst / "gen_meta.json"
    old_sha = file_sha256(gm)
    gm.write_text(gm.read_text(encoding="utf-8").replace(f'"{RECORDED_AREA10}/', f'"{AREA10}/'), encoding="utf-8")
    rt = dst / "env-runtime.json"
    rt.write_text(rt.read_text(encoding="utf-8").replace(old_sha, file_sha256(gm)), encoding="utf-8")
    return dst
ADAPTER = "nemo_rl_agentic"
TRAINER_ID = "rl.nemo_rl_agentic@0.1"
EVALUATOR_ID = "agentic.math_code_items@0.1"
EXECUTOR = f"{AREA10}/executor"
OUT_ROOT = f"{PATHS_ROOT}/out"
DATA_ROOT = PATHS_ROOT
OUT_NAME = "nemo-rl-agentic-math"
REAL = REAL_RUNTIME
REAL_PROBE = REAL_OUT / "p1-a3" / "train"
NEMO_RL_COMMIT = "4f39fd86289a2ffd5aaf5d202702f7fd7185c924"
TIERB_IMAGE = "sha256:8cd5b714ea91ec7ea9a976c75b0ae6f6932140cf7a8aea803100de6331a2113a"
IMAGE = "sha256:7a4078e70b28cc6de622fa3cf25c7596c52337fa3f1779488b402c5fa1c74e69"   # gatedtrain-a6-nemo-rl:1 (uid 1000)
AREA6_EXECUTOR_IMAGE = "sha256:9b0cfb6e125b8986ad562ef18cff5268460b833d046374b292ca164c4558b89a"
TRAIN_PY = "/opt/nemo_rl_venv/bin/python"                   # nemo-rl docker/Dockerfile:227, :621
GEN_PY = "/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python"
BASE_REL = "models/qwen3-1.7b"
BASE_REPO, BASE_REVISION = "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
BASE_WEIGHTS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
INDEX = "model.safetensors.index.json"
BASE_FILES = (".gitattributes", "LICENSE", "README.md", "config.json", "generation_config.json", "merges.txt",
              *BASE_WEIGHTS, INDEX, "tokenizer.json", "tokenizer_config.json", "vocab.json")
# The real base's pins (MODELS.json, equal to the Hub's LFS etags kept in hf-meta/; README section 0.1).
REAL_BASE_LFS = {"model-00001-of-00002.safetensors": "sha256:169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5",
                 "model-00002-of-00002.safetensors": "sha256:912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9",
                 "tokenizer.json": "sha256:aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"}
REAL_BASE_MODEL_HASH = "sha256:11e805260ced8dbff699088815939c03c32220e8a546d9ac713567ef21f0c327"   # manifest, 2026-11-15
REAL_ITEM_SET_HASH = "sha256:0d88fddd57ada8d7968d8855e0bb397ed7ce6ae829c88f539e11f64fa566c675"
REAL_DATA_HASH = "sha256:3d0dd927e6a779ca856b0be36373a0e6ff55ce0f16cf04a75a90c0b1517f1754"
PROMPT_FILE = "prompt_math_code_oneshot.txt"
PROMPT_SHA256 = "sha256:3b7145c1edf2526dcd13d5477d343d68e883b30c8e20d1a5c5e3f19af7b6230f"
STOP = "</code>"
# gen_agentic.py SAMPLER (every cut's sampler.json, ptc_agentic_hooks.write_cut; byte content of p1-a3's sampler.json)
# and the evaluator's sampler (score_agentic.py: SAMPLER plus the backend and the executor's code hashes).
CUT_SAMPLER = {"chat_template": "base tokenizer chat template, add_generation_prompt=True, enable_thinking=False",
               "executor": ("executor/executor_server.py sidecar, stateless replay, limits of "
                            "executor_server.DEFAULT_LIMITS"),
               "include_stop_str_in_output": True, "max_model_len": 4096, "max_new_tokens_per_turn": 1024,
               "max_turns": 4, "messages": "[{role: user, content: <prompt_file>.format(problem)}]",
               "observation": "agentic_core.observation(), raw tokens without special tokens, obs_max_chars 512",
               "prompt_file": PROMPT_FILE, "prompt_file_sha256": PROMPT_SHA256, "samples_per_seed": 1,
               "stop": [STOP], "strategy": "greedy", "temperature": 0.0, "top_p": 1.0}
# The executor after the M2 hardening, and the one the 2026-11-16 gate run (and every real fixture) used.
EXECUTOR_CODE = {"exec_protocol.py": "sha256:cbd8df69d462676420e3d4ddc922be50bcc387bc8d57722dbb60b8815b9783dd",
                 "executor_server.py": "sha256:d1f3256d3f18e85b38950247cb2d1b6ae93c1a18a500e5b63d51bce8c6a9a6dc",
                 "sandbox_runner.py": "sha256:4b656e183b51c8604fe09cdd092c6e98fdfbee50a9f5b8a32d4322586cd353f0"}
EXECUTOR_CODE_GATE = {"exec_protocol.py": "sha256:cbd8df69d462676420e3d4ddc922be50bcc387bc8d57722dbb60b8815b9783dd",
                      "executor_server.py": "sha256:42971d92b4b215a8ab50abfbb4905ca9df1bf068cc7d613b771806d2fd846e2d",
                      "sandbox_runner.py": "sha256:6e1082ec606cdb3dd56a1bb8fbb86429ddbbdc33fd632e2005f9e0272f3b1eaf"}
SAMPLER = {**CUT_SAMPLER, "backend": "vllm", "executor_code_sha256": dict(EXECUTOR_CODE)}
SAMPLER_GATE = {**SAMPLER, "executor_code_sha256": dict(EXECUTOR_CODE_GATE)}
PING = {"code_sha256": dict(EXECUTOR_CODE),
        "limits": {"as_mb": 2048, "capture_chars": 16384, "cpu_s": 10, "fsize_mb": 1, "nofile": 64, "nproc_extra": 32,
                   "wall_s": 10.0},
        "ok": True, "python": TRAIN_PY, "python_version": "3.13.14", "workers": 8}
PING_GATE = {**PING, "code_sha256": dict(EXECUTOR_CODE_GATE)}
GEN_CODE = {"agentic_core.py": "sha256:8c5028886847b5a19b70e9be07b1c728408e58c9f3647574fc06f6a43235bfc3",
            "executor/exec_protocol.py": EXECUTOR_CODE["exec_protocol.py"],
            "gen_agentic.py": "sha256:8f32dcd29de7a71b42476d7918623a940c918d65fc2bb49d1b59a1b99f429669"}
# The turn logic and generator the 2026-11-16 gate run (and the real fixtures) recorded: agentic_core.py before an
# internal reference was removed from its docstring.
GEN_CODE_GATE = {**GEN_CODE,
                 "agentic_core.py": "sha256:c1076f0af2b0c6299e45a6dabccca69d915d76d0131730c497f16d00a77c5847"}
SCORE_AGENTIC_SHA256 = "sha256:839de6f29b2b10f1dc9052563d1f5c31a1b0fcbca63c931e98045685d269a2c1"   # pinned now
# The scorer the 2026-11-16 gate and the real arm fixture ran: SCORE_AGENTIC_SHA256 is it with the name map applied.
SCORE_AGENTIC_GATE_SHA256 = "sha256:a750f5be01871a61e2ba08d5569d58634eaf4aeb35faeb636df790f5254c2ab2"
SCORER_CODE_GATE = "sha256:f8e356f400bcdf7af9fecd0c1cd43926b416a1d8def8563cd77f621e59c952bb"
SCORE_AGENTIC_RUNTIME_SHA256 = "sha256:a3bc32c8dae5f9f00d07990ce0ef5f5029129e41684f003008829023ddde3c44"   # 2026-11-15
SCORER_CODE = "sha256:3ede3a5a8b1bdced8b7eabd832139ddd3f489446016088e293fd5ac2df64febe"
SCORER = {"name": "math_integer_boxed_own_parser", "version": "math_int_boxed_v1; python 3.13.14",
          "weights_sha256": None, "code_sha256": SCORER_CODE}
SUB_METRICS = ("answer_found", "env_tokens", "model_tokens", "tool_calls", "tool_errors", "tool_timeouts", "truncated",
               "turns_used")
SEEDS = [11]
SEED = 10100
PROMPTS, GENERATIONS = 16, 8
MAX_TURNS = 4
LORA = {"dim": 32, "alpha": 64}
EOS = 151645
ENV_NAMES = ["FLASHINFER_CUDA_ARCH_LIST", "HF_HUB_OFFLINE", "HOME", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED",
             "RAY_memory_monitor_refresh_ms", "TORCH_CUDA_ARCH_LIST"]
TRAIN_ENV = {"FLASHINFER_CUDA_ARCH_LIST": "8.9", "HF_HUB_OFFLINE": "1", "HOME": "/tmp/home",
             "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "RAY_memory_monitor_refresh_ms": "0",
             "TORCH_CUDA_ARCH_LIST": "8.9"}
GEN_ENV = {"FLASHINFER_CUDA_ARCH_LIST": "8.9", "HF_HUB_OFFLINE": "1", "HOME": "/tmp/home", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONUNBUFFERED": "1", "TORCH_CUDA_ARCH_LIST": "8.9"}
SCORE_ENV = {"HF_HUB_OFFLINE": "1", "HOME": "/tmp/home", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
SIDECAR_ENV = {"HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
SIDECAR_ARGV = [TRAIN_PY, "-I", f"{EXECUTOR}/executor_server.py", "--serve", "/spool/mathexec.sock", "--workers", "8"]
SIDECAR_LIMITS = {"cpus": 8, "memory": "8g", "memory_swap": "8g", "pids": 256, "nofile": "1024:1024", "core": 0}
SCRIPTS = ("run_agentic_nemo.py", "ptc_agentic_hooks.py", "math_code_env.py", "traj_log.py", "score_agentic.py",
           "consumption.py", "make_data.py")
PINNED_COPIES = {"agentic_core.py": "agentic_core.py", "gen_agentic.py": "gen_agentic.py",
                 "executor/exec_protocol.py": "exec_protocol.py", PROMPT_FILE: PROMPT_FILE}
EXECUTOR_STANDINS = ("executor_server.py", "sandbox_runner.py", "run-executor.sh")
HELPERS = ("agentic_core.py", "math_code_env.py", "traj_log.py", "ptc_agentic_hooks.py", "gen_agentic.py",
           "executor/exec_protocol.py")
LORA_TARGETS = [f"model.layers.{i}.{m}" for i in range(28)
                for m in ("mlp.down_proj", "mlp.gate_proj", "mlp.up_proj", "self_attn.k_proj", "self_attn.o_proj",
                          "self_attn.q_proj", "self_attn.v_proj")]
SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra",
            "precalculus")


def sha_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, doc: Any) -> Path:
    """The runtime's `json.dumps(..., indent=2, sort_keys=True) + "\\n"`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, docs: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(d, sort_keys=True) + "\n" for d in docs), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def float32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def compact_sha(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                                 ensure_ascii=False).encode("utf-8")).hexdigest()


def host_config_sha256(host_work: Path, output_root: Path, input_roots: list, *, mirror_root: Path) -> str:
    """EXT-85, EXT-117, EXT-192: the effective layout's hash (no interpreter trees; the image's interpreters)."""
    return sha_of({"host_work": str(host_work), "output_root": str(output_root),
                   "input_roots": [str(r) for r in input_roots], "mirror_root": str(mirror_root),
                   "interpreter": TRAIN_PY, "interpreter_trees": [],
                   "image_interpreters": {"gen_agentic.py": GEN_PY, "run_agentic_nemo.py": TRAIN_PY,
                                          "score_agentic.py": TRAIN_PY}})


# ---------------------------------------------------------------------------
# The runtime directory (make_data.py, fetch_models.py)
# ---------------------------------------------------------------------------

def problem_text(split: str, k: int) -> str:
    return f"[{split} {k}] What is ${k} + {k % 7}$ times $3$?"


def answer_of(k: int) -> str:
    return str((k + k % 7) * 3)


def train_row(k: int) -> dict:
    """make_data.py row: `input` = the problem = the user message; the integer reference answer."""
    p = problem_text("train", k)
    return {"answer": answer_of(k), "input": p, "level": f"Level {1 + k % 5}",
            "messages": [{"content": p, "role": "user"}, {"content": answer_of(k), "role": "assistant"}],
            "row_id": f"math-train-{k:05d}",
            "source": {"file": f"hendrycks_math/{SUBJECTS[k % 7]}/train-00000-of-00001.parquet", "index": k},
            "subject": SUBJECTS[k % 7]}


def write_data(d: Path, rows: list[dict], n_shards: int = 4) -> Path:
    """`<d>/shards/shard-0000<i>.jsonl` (compact JSON lines, as make_data.py) and `<d>/train.jsonl`, their
    concatenation; returns the shard directory (the spec's training data)."""
    per = len(rows) // n_shards
    text = ""
    (d / "shards").mkdir(parents=True, exist_ok=True)
    for i in range(n_shards):
        chunk = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows[i * per:(i + 1) * per])
        (d / "shards" / f"shard-{i:05d}.jsonl").write_text(chunk, encoding="utf-8")
        text += chunk
    (d / "train.jsonl").write_text(text, encoding="utf-8")
    return d / "shards"


def item_row(k: int) -> dict:
    p = problem_text("test", k)
    return {"answer": answer_of(k), "item": f"math-test-{k:04d}", "level": f"Level {1 + k % 5}", "problem": p,
            "source": {"file": f"hendrycks_math/{SUBJECTS[k % 7]}/test-00000-of-00001.parquet", "index": k},
            "subject": SUBJECTS[k % 7], "text": p}


def write_items(d: Path, n: int) -> list[dict]:
    items = [item_row(k) for k in range(n)]
    write_jsonl(d / "items.jsonl", items)
    write_json(d / "item_set_meta.json", {"env_id": "math_code", "n_items": n, "schema": "ptc_mm.agentic_items",
                                          "schema_version": "1.0", "split": "test"})
    return items


def build_base(d: Path) -> Path:
    for name in BASE_FILES:
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if name in BASE_WEIGHTS:
            p.write_bytes(safetensors_bytes(f"qwen3-1.7b {name}"))
        elif name == INDEX:
            write_json(p, {"metadata": {"total_size": 16}, "weight_map": {"a": BASE_WEIGHTS[0], "b": BASE_WEIGHTS[1]}})
        elif name.endswith(".json"):
            write_json(p, {"_fixture": name, "architectures": ["Qwen3ForCausalLM"]})
        else:
            p.write_text(f"fixture stand-in {name}\n", encoding="utf-8")
    return d


def models_json(base: Path) -> dict:
    """fetch_models.py MODELS.json: files {name: {bytes, sha256}}."""
    return {"models": {base.name: {"bytes": sum((base / n).stat().st_size for n in BASE_FILES),
                                   "files": {n: {"bytes": (base / n).stat().st_size, "sha256": file_sha256(base / n)}
                                             for n in BASE_FILES},
                                   "license": "apache-2.0", "repo": BASE_REPO, "revision": BASE_REVISION}},
            "schema": "ptc_mm.agentic_models", "schema_version": "1.0"}


def build_runtime(rt: Path) -> None:
    """Script stand-ins (never executed), the pinned byte copies and the executor directory."""
    rt.mkdir(parents=True, exist_ok=True)
    for name in SCRIPTS:
        (rt / name).write_text(f"# fixture stand-in for {AREA10}/{name}; never executed\n", encoding="utf-8")
    (rt / "executor").mkdir(exist_ok=True)
    for name in EXECUTOR_STANDINS:
        (rt / "executor" / name).write_text(f"# fixture stand-in for {EXECUTOR}/{name}; never executed\n",
                                            encoding="utf-8")
    for rel, src in PINNED_COPIES.items():
        shutil.copyfile(RUNTIME_COPIES / src, rt / rel)


# ---------------------------------------------------------------------------
# Cuts (ptc_agentic_hooks.write_cut) and identity
# ---------------------------------------------------------------------------

def adapter_config() -> dict:
    """Automodel's HF PEFT adapter_config.json (p1-a3's keys; target_modules the 196 resolved projections)."""
    return {"base_model_name_or_path": f"{AREA10}/{BASE_REL}", "bias": "none", "lora_alpha": LORA["alpha"],
            "peft_type": "LORA", "r": LORA["dim"], "target_modules": list(LORA_TARGETS), "task_type": "CAUSAL_LM",
            "use_dora": False}


def build_cut(run: Path, step: int, *, profile: str, seed: int, script_sha256: str, base: Path, memory: str = "a",
              env_id: str = "math_code", tag: str = "") -> Path:
    """`<run>/nemo/checkpoints/step_<N>/` (NeMo RL's resume state, never a cut) and `<run>/cuts/checkpoint-<N>/` as
    write_cut builds it: the PEFT files hard-linked, the tokenizer files copied, sampler.json, base_ref.json,
    training_info.json, `cut.json` last; `cuts.jsonl` appended."""
    step_dir = run / "nemo" / "checkpoints" / f"step_{step}"
    wdir = step_dir / "policy" / "weights" / "model"
    wdir.mkdir(parents=True)
    (wdir / "adapter_model.safetensors").write_bytes(safetensors_bytes(f"lora step {step} {tag}"))
    write_json(wdir / "adapter_config.json", adapter_config())
    write_json(wdir / "automodel_peft_config.json", {"dim": LORA["dim"], "alpha": LORA["alpha"],
                                                      "target_modules": ["*_proj"], "match_all_linear": False})
    tok = step_dir / "policy" / "tokenizer"
    tok.mkdir(parents=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"):
        (tok / name).write_text(f"tokenizer stand-in {name}\n", encoding="utf-8")
    (step_dir / "training_info.json").write_text(json.dumps({"consumed_samples": step * PROMPTS, "current_step": step}),
                                                 encoding="utf-8")
    d = run / "cuts" / f"checkpoint-{step}"
    d.mkdir(parents=True)
    for name in ("adapter_model.safetensors", "adapter_config.json", "automodel_peft_config.json"):
        os.link(wdir / name, d / name)
    origin = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"):
        shutil.copyfile(tok / name, d / name)
        origin[name] = f"policy/tokenizer/{name}"
    write_json(d / "sampler.json", CUT_SAMPLER)
    entry = models_json(base)["models"][base.name]
    write_json(d / "base_ref.json", {"dir_name": base.name, "files": entry["files"], "init_checkpoint": None,
                                     "repo": BASE_REPO, "revision": BASE_REVISION})
    shutil.copyfile(step_dir / "training_info.json", d / "training_info.json")
    files = {f.name: file_sha256(f) for f in sorted(d.iterdir())}
    not_dep = ("base_ref.json", "training_info.json", "automodel_peft_config.json")
    write_json(d / "cut.json", {"schema": "ptc_mm.agentic_cut", "schema_version": "1.0", "global_step": step,
                                "files": files, "kind": "lora_adapter", "lora": dict(LORA),
                                "deployable": sorted(n for n in files if n not in not_dep), "file_origin": origin,
                                "seed": seed, "profile": profile, "stage": "agentic_grpo", "env_id": env_id,
                                "memory_rung": memory, "script_sha256": script_sha256,
                                "examples_consumed": step * PROMPTS, "nemo_rl_step_dir": f"step_{step}"})
    with open(run / "cuts.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"cut": d.name, "step": step, "utc": "2026-11-15T10:00:00.000000Z"}, sort_keys=True) + "\n")
    return d


def adapter_hash(cut: Path) -> str:
    deploy = json.loads((cut / "cut.json").read_text(encoding="utf-8"))["deployable"]
    entries = [e for e in manifest_entries(cut) if e["path"] in deploy]
    return "sha256:" + hashlib.sha256(canonical_bytes(entries)).hexdigest()


def composed_hash(cut: Path, base: Path) -> str:
    """MM-18: SHA-256 of the canonical JSON {base: the base's manifest hash, adapter: the deployable files' hash}."""
    return sha_of({"base": manifest_hash(base), "adapter": adapter_hash(cut)})


# ---------------------------------------------------------------------------
# The training run (run_agentic_nemo.py + ptc_agentic_hooks.py + math_code_env.py + traj_log.py)
# ---------------------------------------------------------------------------

def overrides(*, steps: int, save_period: int, seed: int, memory: str, env_id: str = "math_code",
              max_turns: int = MAX_TURNS, stop: str = STOP) -> list[str]:
    """run_agentic_nemo.py `overrides` (abridged to the ones the adapter needs, plus some context)."""
    model = f"{AREA10}/{BASE_REL}"
    return [f"policy.model_name={model}", f"policy.tokenizer.name={model}",
            "++policy.tokenizer.chat_template_kwargs.enable_thinking=false", "policy.max_total_sequence_length=4096",
            "data.shuffle=false", f"++data.train.data_path={AREA10}/data/train.jsonl",
            f"data.default.prompt_file={AREA10}/{PROMPT_FILE}", "data.default.processor=math_code_rowid",
            f"data.default.env_name={env_id}", f"++env.{env_id}.socket=/spool/mathexec.sock", f"++env.{env_id}.log_dir=/out",
            f"grpo.max_rollout_turns={max_turns}", f"checkpointing.save_period={save_period}",
            "logger.log_dir=/out/nemo/logs", "policy.automodel_cfg.lora_cfg.enabled=true",
            f"policy.automodel_cfg.lora_cfg.dim={LORA['dim']}", f"policy.automodel_cfg.lora_cfg.alpha={LORA['alpha']}",
            "policy.automodel_cfg.lora_cfg.match_all_linear=false",
            "policy.automodel_cfg.lora_cfg.target_modules=['*_proj']", f"grpo.max_num_steps={steps}",
            f"grpo.seed={seed}", "grpo.num_prompts_per_step=16", "grpo.num_generations_per_prompt=8",
            "policy.generation.max_new_tokens=1024", f'policy.generation.stop_strings=["{stop}"]',
            "++policy.generation.vllm_cfg.cap_max_tokens_to_context=true", "data_plane.enabled=false"]


def run_args(*, profile: str, seed: int, steps: int, save_period: int, runtime: Path, script_sha: str,
             memory: str = "a", env_id: str = "math_code", ping: dict = PING) -> dict:
    """run_agentic_nemo.py `main` -> ptc_agentic_hooks.write_run_args."""
    prof = {"steps": steps, "prompts": PROMPTS, "generations": GENERATIONS, "save_period": save_period}
    mem = {"a": {"max_seq": 4096, "vllm": 0.35, "lora": True, "lr": 2.0e-5, "extra": ["policy.logprob_chunk_size=512"]},
           "b": {"max_seq": 4096, "vllm": 0.4, "lora": True, "lr": 2.0e-5, "extra": []}}[memory]
    return {"attempt": 1, "base_model": f"{AREA10}/{BASE_REL}",
            "code_sha256": {f: file_sha256(runtime / f) for f in HELPERS}, "created_at": "2026-11-15T09:14:58.000000Z",
            "dry_run": False, "env_id": env_id, "executor": {"ping": json.loads(json.dumps(ping)),
                                                             "socket": "/spool/mathexec.sock"},
            "framework": {"commit": NEMO_RL_COMMIT, "name": "nemo_rl",
                          "versions": {"nemo-rl": "0.8.0+4f39fd862", "torch": "2.13.0+cu130", "transformers": "5.16.1"}},
            "init_checkpoint": None, "lora": dict(LORA), "max_new_tokens_per_turn": 1024, "max_rollout_turns": MAX_TURNS,
            "max_steps": steps, "max_total_sequence_length": 4096, "memory_rung": memory, "memory_settings": mem,
            "model_name_or_path": f"{AREA10}/{BASE_REL}",
            "nemo_rl_overrides": overrides(steps=steps, save_period=save_period, seed=seed, memory=memory,
                                           env_id=env_id),
            "profile": profile, "profile_settings": prof, "prompt_file": PROMPT_FILE,
            "prompt_file_sha256": file_sha256(runtime / PROMPT_FILE), "python": "3.13.14",
            "script_sha256": script_sha, "seed": seed,
            "seed_derivation": "NeMo RL grpo.seed (set_seed in setup); vLLM sampling seeded by NeMo RL; data order = "
                               "file order (data.shuffle=false); executor replay random.seed(0)",
            "stage": "agentic_grpo", "train_file": f"{AREA10}/data/train.jsonl",
            "train_file_sha256": file_sha256(runtime / "data" / "train.jsonl")}


KINDS = ("tool_right", "right", "wrong", "no_answer", "err_ok_right", "env_max_turns", "overflow", "right",
         "syntax_wrong", "right")


def turn_script(kind: str, answer: str) -> tuple[list[tuple[str, str, str | None]], str | None, bool]:
    """[(assistant text, observation, tool status)] of a scripted math_code trajectory, its env end reason (None: the
    loop ended it) and whether the final answer is right."""
    ok = lambda k: (f"<code>print({k})</code>", f"<result>\n{k}\n</result>\n", "ok")
    err = ("<code>1/0</code>", "<result>\nZeroDivisionError (line 1): division by zero\n</result>\n", "error")
    syn = ("<code>print(</code>", "<result>\nSyntaxError (line 1): '(' was never closed\n</result>\n", "syntax_error")
    right = (f"so \\boxed{{{answer}}}", "", None)
    wrong = ("\\boxed{-1}", "", None)
    if kind == "tool_right":
        return [ok(1), right], "final", True
    if kind == "right":
        return [right], "final", True
    if kind == "wrong":
        return [wrong], "final", False
    if kind == "no_answer":
        return [("I think it is about five", "", None)], "no_answer", False
    if kind == "err_ok_right":
        return [err, ok(2), right], "final", True
    if kind == "env_max_turns":
        return [ok(1), ok(2), ok(3), ("<code>print(4)</code>", "", None)], "max_turns", False
    if kind == "overflow":
        return [ok(9)], None, False
    if kind == "syntax_wrong":
        return [syn, wrong], "final", False
    raise KeyError(kind)


def kind_of(step: int, pos: int) -> str:
    return KINDS[(step * 7 + pos) % len(KINDS)]


def step_logs(rows: list[dict], step: int, first_call: int) -> tuple[list[dict], list[dict], list[dict]]:
    """One GRPO step as NeMo RL's synchronous loop and the logging env produce it: (env_calls.jsonl lines,
    trajectories.jsonl lines, train_data_step<step>.jsonl lines). The step's 16 rows (file order, drop_last, each 8
    times in a row) at batch positions 0..127; one environment call per turn over the still active trajectories."""
    spe = len(rows) // PROMPTS
    start = ((step - 1) % spe) * PROMPTS
    batch = [rows[start + g] for g in range(PROMPTS) for _ in range(GENERATIONS)]
    scripts = [turn_script(kind_of(step, p), r["answer"]) for p, r in enumerate(batch)]
    env, ends = [], []
    active = list(range(len(batch)))
    call = first_call
    turn = 0
    utc = "2026-11-15T11:00:00.000000Z"
    while active:
        turn += 1
        nxt = []
        for i, p in enumerate(active):
            turns, end, is_right = scripts[p]
            text, obs, status = turns[turn - 1]
            last = turn == len(turns)
            term = last and end is not None
            resp = sha_text("".join(t[0] for t in turns[:turn]))
            line = {"call": call, "end_reason": end if term else None, "env_id": "math_code", "obs_chars": len(obs),
                    "position": i, "response_sha256": resp,
                    "reward": (1.0 if is_right else 0.0) if term else 0.0, "row_id": batch[p]["row_id"],
                    "terminated": term, "tool_status": None if term else status, "traj_call": first_call,
                    "traj_pos": p, "turn": turn, "utc": utc}
            env.append(line)
            if term:
                ends.append({k: line[k] for k in ("call", "end_reason", "env_id", "response_sha256", "row_id",
                                                  "traj_call", "traj_pos", "turn", "utc")})
            elif not last:
                nxt.append(p)
        active = nxt
        call += 1
    logged = []
    for p, r in enumerate(batch):
        turns, end, is_right = scripts[p]
        content = [f"<|im_start|>user\n{r['input']}<|im_end|>"]
        ids, mask = [11, 12, 13, 14, 15], [0] * 5
        for k, (text, obs, status) in enumerate(turns):
            content += [text, obs]
            terminal = k == len(turns) - 1 and end is not None
            ids += [21, 22, EOS if terminal and status is None and not text.endswith(STOP) else 7]
            mask += [1, 1, 1]
            n_obs = 0 if not obs else (1 if end is None and k == len(turns) - 1 else 2)   # overflow: the cut obs
            ids += [31] * n_obs
            mask += [0] * n_obs
        reward = 1.0 if (end is not None and is_right) else 0.0
        logged.append({"content": [content], "rewards": [reward], "input_lengths": [len(mask)],
                       "token_ids": [ids + [0, 0]], "token_loss_mask": [mask + [0, 0]], "sample_loss_mask": [1.0],
                       "idx": p})
    return env, ends, logged


def build_run(out: Path, runtime: Path, *, profile: str = "full", steps: int = 60, save_period: int = 20,
              seed: int = SEED, memory: str = "a", ping: dict = PING) -> Path:
    out.mkdir(parents=True)
    script_sha = file_sha256(runtime / "run_agentic_nemo.py")
    (out / "env-freeze.txt").write_text("nemo-rl==0.8.0+4f39fd862\ntorch==2.13.0+cu130\ntransformers==5.16.1\n"
                                        f"# nemo-rl commit {NEMO_RL_COMMIT}\n", encoding="utf-8")
    write_jsonl(out / "attempts.jsonl", [{"attempt": 1, "event": "start", "pid": 1, "profile": profile,
                                          "stage": "agentic_grpo", "utc": "2026-11-15T09:00:00.000000Z"},
                                         {"attempt": 1, "event": "end", "pid": 1, "stage": "agentic_grpo",
                                          "utc": "2026-11-15T10:00:00.000000Z"}])
    write_json(out / "run_args.json", run_args(profile=profile, seed=seed, steps=steps, save_period=save_period,
                                               runtime=runtime, script_sha=script_sha, memory=memory, ping=ping))
    rows = read_jsonl(runtime / "data" / "train.jsonl")
    env, ends = [], []
    logs = out / "nemo" / "logs" / "exp_001"
    logs.mkdir(parents=True)
    call = 1
    for n in range(1, steps + 1):
        e, t, lines = step_logs(rows, n, call)
        call = e[-1]["call"] + 1
        env += e
        ends += t
        (logs / f"train_data_step{n}.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")
    write_jsonl(out / "env_calls.jsonl", env)
    write_jsonl(out / "trajectories.jsonl", ends)
    base = runtime / BASE_REL
    for k in range(save_period, steps + 1, save_period):
        build_cut(out, k, profile=profile, seed=seed, script_sha256=script_sha, base=base, memory=memory)
    return out


# ---------------------------------------------------------------------------
# Evaluations (gen_agentic.py gen/ and gen_meta.json, then score_agentic.py)
# ---------------------------------------------------------------------------

def baseline_kind(i: int) -> str:
    return ("right", "wrong", "no_answer", "tool_right", "wrong", "env_max_turns", "overflow", "err_ok_right",
            "syntax_wrong", "right")[i % 10]


def candidate_kind(i: int) -> str:
    return "no_answer" if i % 10 == 2 else ("wrong" if i % 7 == 1 else ("tool_right" if i % 3 else "right"))


def gen_doc(item: dict, kind: str, seed: int = 11, repeat: int = 0) -> dict:
    """gen_agentic.py's `gen/<item>__s<seed>.json` for a scripted trajectory (run_trajectories semantics: a tool turn
    stops at `</code>`, its observation tokens follow; a final turn ends at EOS; overflow cuts the observation)."""
    turns_s, end, _ = turn_script(kind, item["answer"])
    turns = []
    for k, (text, obs, status) in enumerate(turns_s):
        last = k == len(turns_s) - 1
        code = None
        if text.rstrip().endswith(STOP) and "<code>" in text:
            code = text.rstrip()[: -len(STOP)].rsplit("<code>", 1)[1].strip()
        if end is None and last:                                          # overflow: the observation was cut
            obs_tokens = 1
        else:
            obs_tokens = 2 if obs else 0
        turns.append({"text": text, "n_tokens": 3 if status is None else 5,
                      "finish_reason": "stop", "code": code, "tool_status": status if (not last or end is None) else None,
                      "observation": obs, "obs_tokens": obs_tokens})
    prompt_tokens = 300 + len(item["problem"]) // 4
    model = sum(t["n_tokens"] for t in turns)
    envt = sum(t["obs_tokens"] for t in turns)
    return {"item": item["item"], "seed": seed, "repeat": repeat, "prompt_sha256": sha_text(item["problem"]),
            "prompt_tokens": prompt_tokens, "turns": turns, "end_reason": end or "overflow",
            "final_text": turns[-1]["text"], "model_tokens": model, "env_tokens": envt,
            "total_tokens": prompt_tokens + model + envt, "error": None}


def gen_meta(*, items: list[dict], items_sha: str, base_files: dict, cut: Path | None, limit: int | None = None,
             ping: dict = PING) -> dict:
    """gen_agentic.py gen_meta.json. `base_files`: {name: sha256} of the base directory's .json/.safetensors/.txt/.jinja
    files; `cut`: the LoRA cut mounted at /inputs/cut (None: the base alone)."""
    adapter = None
    if cut is not None:
        doc = json.loads((cut / "cut.json").read_text(encoding="utf-8"))
        adapter = {"dir": "/inputs/cut", "global_step": doc["global_step"], "cut_json_sha256": file_sha256(cut / "cut.json"),
                   "files": {f: h for f, h in doc["files"].items() if f in doc["deployable"]}}
    return {"backend": "vllm", "env_id": "math_code", "model_dir": f"{AREA10}/{BASE_REL}", "model_files": base_files,
            "adapter": adapter, "cut": None, "sampler": CUT_SAMPLER, "eval_seeds": list(SEEDS), "items": len(items),
            "item_ids": [it["item"] for it in items], "items_sha256": items_sha, "limit": limit,
            "prompt_file": PROMPT_FILE, "prompt_file_sha256": PROMPT_SHA256, "code_sha256": dict(GEN_CODE),
            "executor": json.loads(json.dumps(ping)),
            "versions": {"vllm": "0.29.0", "transformers": "5.16.1", "dtype": "bfloat16", "python": "3.13.14"},
            "wall_seconds": 120.0, "seconds_per_trajectory": 0.3}


def base_gen_files(base: Path) -> dict:
    return {p.name: file_sha256(p) for p in sorted(base.iterdir())
            if p.is_file() and p.suffix in (".json", ".safetensors", ".txt", ".jinja")}


def write_gen(out: Path, items: list[dict], kind_fn: Callable[[int], str], meta: dict) -> Path:
    for i, it in enumerate(items):
        write_json(out / "gen" / f"{it['item']}__s11.json", gen_doc(it, kind_fn(i)))
    write_json(out / "gen_meta.json", meta)
    return out


# score_agentic.py's parser (copied for the fixture builder; the adapter has its own copy, checked against the runtime)
_BOX = re.compile(r"\\(?:boxed|fbox)\s*\{")


def _parse(text: str | None) -> int | None:
    if not text:
        return None
    found = None
    for m in _BOX.finditer(text):
        depth, j = 1, m.end()
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        if depth == 0:
            found = text[m.end():j - 1]
    if found is None:
        return None
    t = re.sub(r"\s+", "", found)
    return int(t) if re.fullmatch(r"-?\d+", t) else None


def score_doc(doc: dict, ref: int) -> tuple[float, str, dict, int | None]:
    """score_agentic.py score_doc."""
    turns = doc["turns"]
    answer = _parse(doc.get("final_text"))
    statuses = [t.get("tool_status") for t in turns if t.get("tool_status") is not None]
    truncated = int(doc.get("end_reason") == "overflow" or any(t.get("finish_reason") == "length" for t in turns))
    sub = {"turns_used": len(turns), "tool_calls": len(statuses),
           "tool_errors": sum(s in ("error", "syntax_error", "timeout", "crash") for s in statuses),
           "tool_timeouts": sum(s == "timeout" for s in statuses), "answer_found": int(answer is not None),
           "truncated": truncated, "model_tokens": int(doc["model_tokens"]), "env_tokens": int(doc["env_tokens"])}
    if answer is not None:
        outcome = "correct" if answer == ref else "wrong_answer"
    elif doc.get("end_reason") in ("max_turns", "overflow"):
        outcome = doc["end_reason"]
    else:
        outcome = "no_answer"
    return float(outcome == "correct"), outcome, sub, answer


def build_score(out: Path, gen: Path, items: list[dict], items_dir: Path, *, model: str | None, base_model: str | None,
                item_set: str | None, python: str = "3.13.14") -> Path:
    """score_agentic.py main (sha256 839de6f2..., without `env_id` in eval_meta.json) in the image: outputs/,
    scores.jsonl, arm-exact_match/ (hard-linked outputs, MM-1 items.jsonl, MM-2 eval_meta.json), env-runtime.json,
    summary.json, gen_meta.json copied."""
    (out / "outputs").mkdir(parents=True)
    gm = json.loads((gen / "gen_meta.json").read_text(encoding="utf-8"))
    rows = []
    for it in items:
        ref = int(it["answer"]) if re.fullmatch(r"-?\d+", it["answer"]) else None
        gp = gen / "gen" / f"{it['item']}__s11.json"
        row = {"item": it["item"], "repeat": 0, "seed": 11, "error": None, "score": None, "sub_metrics": None,
               "outcome": None}
        doc, answer = None, None
        if not gp.exists():
            row["error"] = {"class": "missing_output", "message": "no generation for this (item, seed)"}
        elif ref is None:
            row["error"] = {"class": "scorer_error", "message": f"reference {it['answer']!r} is not an integer"}
        else:
            doc = json.loads(gp.read_text(encoding="utf-8"))
            if doc.get("error"):
                row["error"] = {"class": doc["error"]["class"], "message": doc["error"].get("message")}
            else:
                row["score"], row["outcome"], row["sub_metrics"], answer = score_doc(doc, ref)
        rel = f"outputs/{it['item']}__r0.json"
        write_json(out / rel, {"item": it["item"], "repeat": 0, "seed": 11, "problem": it["problem"],
                               "reference": it["answer"], "response": doc.get("final_text") if doc else None,
                               "transcript": doc.get("turns") if doc else None,
                               "end_reason": doc.get("end_reason") if doc else None, "parsed_answer": answer,
                               "score": row["score"], "outcome": row["outcome"], "sub_metrics": row["sub_metrics"],
                               "error": row["error"]})
        row["output_ref"] = {"path": rel, "sha256": file_sha256(out / rel)}
        rows.append(row)
    write_jsonl(out / "scores.jsonl", rows)
    code = {"score_agentic.py": SCORE_AGENTIC_SHA256}
    sampler = {**gm["sampler"], "backend": gm["backend"], "prompt_file_sha256": gm["prompt_file_sha256"],
               "executor_code_sha256": gm["executor"]["code_sha256"]}
    arm = out / "arm-exact_match"
    arm.mkdir()
    lines = []
    for r in rows:
        dst = arm / r["output_ref"]["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(out / r["output_ref"]["path"], dst)
        lines.append({"item": r["item"], "repeat": r["repeat"], "seed": r["seed"], "score": r["score"],
                      "error": r["error"], "sub_metrics": r["sub_metrics"], "output_ref": r["output_ref"]})
    write_jsonl(arm / "items.jsonl", lines)
    write_json(arm / "eval_meta.json", {
        "schema": "gatedtrain.item_eval", "schema_version": "1.0", "evaluator_id": EVALUATOR_ID, "adapter": ADAPTER,
        "metric": "exact_match", "direction": "higher_is_better", "score_range": [0, 1],
        "checkpoint": {"model_hash": model, "base_model_hash": base_model}, "normalizer_sha256": None,
        "env_version_sha256": None, "item_set_hash": item_set, "eval_seeds": list(SEEDS), "sampler": sampler,
        "sampler_hash": sha_of(sampler) if model else compact_sha(sampler),
        "scorer": {"name": "math_integer_boxed_own_parser", "version": f"math_int_boxed_v1; python {python}",
                   "weights_sha256": None, "code_sha256": compact_sha(code)},
        "framework": {"name": "vllm", "version": gm["versions"]["vllm"]}, "created_at": "2026-11-15T12:00:00Z"})
    write_json(out / "env-runtime.json", {
        "python": python, "executable": TRAIN_PY, "code_sha256": code, "scorer_code_sha256": compact_sha(code),
        "items_sha256": file_sha256(items_dir / "items.jsonl"), "gen_meta_sha256": file_sha256(gen / "gen_meta.json"),
        "uid": 1000, "gid": 1000})
    write_json(out / "summary.json", {"items": len(items), "errors": sum(1 for r in rows if r["error"])})
    shutil.copyfile(gen / "gen_meta.json", out / "gen_meta.json")
    return out


# ---------------------------------------------------------------------------
# Spec, registration, records
# ---------------------------------------------------------------------------

REGISTRATION = """registration_version: 1
registration_id: ext-agentic-fixture-superiority-v1
signed_by: prototype-unsigned
workflow: ext-agentic-fixture
gate: quality_gate
clause:
  type: superiority
  metric: exact_match
  direction: higher_is_better
  margin: 0.0
  alpha: 0.025
  power: 0.8
  planning_effect: 0.3
  sigma_plan: fixed
  unit: item
  samples_per_item: 1
  eval_seeds: [11]
  interval: cluster_bootstrap
  blocking: true
margin_justification: Fixture value (superiority, delta_sup = 0).
sigma_pilot:
  sigma: 0.5
  source: Fixture value (n_required = ceil(((1.95996 + 0.84162) x 0.5 / 0.3)^2) = 22 <= 40 items).
baseline_pilot_mean: 0.4
construct_alignment: >-
  exact_match measures final-answer correctness, the construct the math_verify reward also rewards, with another answer
  parser; a pass shows more correct answers under an independent parser, not a general gain in mathematics.
shared_annotation: >-
  The references are MATH boxed integer answers like the training answers; no training stage consumed them, so the
  scorer stays T3. Part of a gain may be learning that answer convention.
eval_distribution:
  item_set_hash: "{item_set_hash}"
sampler: {sampler}
evaluators:
  - id: {evaluator}
    manifest_hash: "{evaluator_hash}"
multiplicity:
  plan: single_gate
  family: null
analysis_code_version: gatedtrain 0.3.0
bootstrap:
  replicates: 10000
  seed: 20261115
pairing:
  max_unpaired: 0
min_paired: 10
contamination:
  allow_exact_overlap: false
"""

SPEC = """workflow: ext-agentic-fixture
version: 1
budget: {{gpu_hours: 24}}
artifacts:
  base:         {{type: model_checkpoint, uri: "file:{base}"}}
  train:        {{type: training_dataset, uri: "file:{data}"}}
  items:        {{type: item_set, uri: "file:{items}", disjoint_from: [train]}}
  trainer_cfg:  {{type: trainer_config, uri: "file:trainer.yaml"}}
  registration: {{type: registration, uri: "file:registration.yaml@sha256:{registration_sha256}"}}
stages:
  - id: agentic_grpo
    uses: rl.nemo_rl_agentic@0.1
    algorithm: grpo
    model_family: llm.qwen3@0.1
    inputs: {{model: base, data: train, config: trainer_cfg}}
    outputs: {{model: ckpt}}
  - id: quality_gate
    uses: gate.superiority@1.0
    evaluator: {evaluator}
    sampler: deployment
    inputs: {{model: ckpt, baseline: base, items: items, registration: registration}}
    test: {{metric: exact_match, alpha: 0.025, power: 0.80, mde: 0.3, cluster_by: item, seeds: 1}}
    on_fail: stop
    owner: skeleton-demo
    escalation: [skeleton-demo-lead]
"""


def smoke_record(*, base_model: str, item_set: str, **overrides) -> dict[str, Any]:
    """A PASS smoke.json (MM-38) on the deciding binary metric: 20 baseline scores, mean inside (0, 1)."""
    rec = {"schema": "gatedtrain.smoke_record", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
           "adapter": ADAPTER, "area": 10, "base_model_hash": base_model, "item_set_hash": item_set,
           "peak_gpu_mem_gb": 13.0, "host_ram_peak_gb": 20.0, "swap_peak_gb": 1.0, "seconds_per_step": 120.0,
           "projected_gpu_h": 2.5, "gpu_log_total_h": 30.0,
           "baseline_scores": [1.0 if i % 3 else 0.0 for i in range(20)], "error_count": 0,
           "verdict": "PASS", "reasons": []}
    rec.update(overrides)
    return rec


class AgenticScenario:
    """Area 10 on disk (module docstring)."""

    def __init__(self, root: Path, *, n_rows: int = 48, n_items: int = 40, steps: int = 60, save_period: int = 20,
                 ping: dict = PING):
        """`ping`: the executor ping the trainer and both generations record (PING_GATE: the pre-M2 executor)."""
        self.root = root
        self.ping = ping
        self.mirror = root / "host"
        self.data_root = self.mirror / DATA_ROOT.lstrip("/")
        self.runtime = self.mirror / AREA10.lstrip("/")
        self.output_root = self.mirror / OUT_ROOT.lstrip("/")
        self.host_work = root / "host-work"
        rt = self.runtime
        build_runtime(rt)
        self.base = build_base(rt / BASE_REL)
        write_json(rt / "models" / "MODELS.json", models_json(self.base))
        self.rows = [train_row(k) for k in range(n_rows)]
        self.data = write_data(rt / "data", self.rows)
        self.items_dir = rt / "items-test"
        self.items = write_items(self.items_dir, n_items)
        gpu_step_script(self.host_work)
        (self.host_work / "gpu-log.md").write_bytes((EXT / "gpu-log.fixture.md").read_bytes())
        self.runs = self.output_root / OUT_NAME
        self.out = build_run(self.runs / "train-1" / "train", rt, steps=steps, save_period=save_period, ping=ping)
        self.spool = self.runs / "train-1" / "spool"
        self.spool.mkdir()
        self.cut = self.out / "cuts" / f"checkpoint-{steps}"
        self.base_model = manifest_hash(self.base)
        self.cand_model = composed_hash(self.cut, self.base)
        self.item_set_hash = manifest_hash(self.items_dir)
        files = base_gen_files(self.base)
        isha = file_sha256(self.items_dir / "items.jsonl")
        self.arms = {}
        for name, kind_fn, cut, model, bmh in (("base", baseline_kind, None, self.base_model, None),
                                               ("cand", candidate_kind, self.cut, self.cand_model, self.base_model)):
            ev = self.runs / f"eval-{name}-1"
            gen = write_gen(ev / "gen", self.items, kind_fn, gen_meta(items=self.items, items_sha=isha,
                                                                     base_files=files, cut=cut, ping=ping))
            (ev / "spool").mkdir(parents=True)
            self.arms[name] = build_score(ev / "score", gen, self.items, self.items_dir, model=model, base_model=bmh,
                                          item_set=self.item_set_hash)
        (root / "trainer.yaml").write_text("backend: nemo_rl_agentic\nprofile: full\n", encoding="utf-8")
        self.render()
        rec = self.launch_record()
        write_canonical(root / "launch.json", rec)
        self.label = rec["label"]
        write_canonical(root / "smoke.json", smoke_record(base_model=self.base_model, item_set=self.item_set_hash))
        self.guard_log()

    # --- paths and environment ---

    def env(self) -> dict[str, str]:
        return {ENV_HOST_WORK: str(self.host_work), ENV_OUTPUT_ROOT: str(self.output_root),
                ENV_INPUT_ROOTS: str(self.data_root), ENV_MIRROR_ROOT: str(self.mirror)}

    def config_sha(self) -> str:
        return host_config_sha256(self.host_work, self.output_root, [self.data_root], mirror_root=self.mirror)

    def guard_log(self) -> None:
        lines = [f"2026-11-11T09:00:00Z START {self.label} attempt=1", f"2026-11-11T10:00:00Z END {self.label} attempt=1 rc=0"]
        (self.host_work / "gpu-guard.log").write_text("".join(l + "\n" for l in lines), encoding="utf-8")

    def render(self, reg_mutate: Callable[[dict], None] | None = None, sampler: dict | None = None) -> None:
        sampler = sampler or {**SAMPLER, "executor_code_sha256": dict(self.ping["code_sha256"])}
        text = REGISTRATION.format(item_set_hash=self.item_set_hash, sampler=json.dumps(sampler),
                                   evaluator=EVALUATOR_ID,
                                   evaluator_hash=sha_of({"evaluator": EVALUATOR_ID, "metric": "exact_match",
                                                          "direction": "higher_is_better", "scorer": SCORER}))
        if reg_mutate is not None:
            reg = yaml.safe_load(text)
            reg_mutate(reg)
            text = yaml.safe_dump(reg, sort_keys=False)
        (self.root / "registration.yaml").write_text(text, encoding="utf-8")
        spec = SPEC.format(base=self.base, data=self.data, items=self.items_dir, evaluator=EVALUATOR_ID,
                           registration_sha256=file_sha256(self.root / "registration.yaml")[len("sha256:"):])
        (self.root / "spec.yaml").write_text(spec, encoding="utf-8")

    # --- launch ---

    def mounts(self, out: Path, spool: Path | None = None) -> list[dict]:
        return [{"host": str(self.runtime), "container": AREA10, "mode": "ro"},
                {"host": str(out), "container": "/out", "mode": "rw"},
                {"host": str(spool or out.parent / "spool"), "container": "/spool", "mode": "rw"}]

    def argv(self, profile: str = "full", memory: str = "a") -> list[str]:
        return [TRAIN_PY, f"{AREA10}/run_agentic_nemo.py", "--profile", profile, "--out", "/out", "--seed", str(SEED),
                "--base", f"{AREA10}/{BASE_REL}", "--memory", memory, "--socket", "/spool/mathexec.sock"]

    def sidecar_request(self, spool: Path, **overrides) -> dict:
        side = {"runtime": {"kind": "container", "image": IMAGE}, "argv": list(SIDECAR_ARGV),
                "mounts": [{"host": str(self.runtime / "executor"), "container": EXECUTOR, "mode": "ro"},
                           {"host": str(spool), "container": "/spool", "mode": "rw"}],
                "env": dict(SIDECAR_ENV), "network": "none", "user": "1000:1000", "tmpfs": ["/tmp"], "read_only": True,
                "cap_drop": ["ALL"], "security_opt": ["no-new-privileges"], "gpus": False,
                "limits": dict(SIDECAR_LIMITS)}
        side.update(overrides)
        return side

    def sidecar_record(self, spool: Path) -> dict:
        side = self.sidecar_request(spool)
        env = side.pop("env")
        side["env_names"] = sorted(env)
        side["runtime"] = {"kind": "container", "image": IMAGE, "venv_lock_sha256": None}
        side["interpreter_sha256"] = None
        side["script_sha256"] = file_sha256(self.runtime / "executor" / "executor_server.py")
        return side

    def launch_record(self, **overrides) -> dict[str, Any]:
        """ext-launch's record of the full training run (with its sidecar)."""
        rec = {"schema": "gatedtrain.launch_record", "schema_version": "2.0", "tags": ["prototype", "unpromotable"],
               "adapter": ADAPTER, "area": 10, "profile": "full", "label": "mm-a10-agentic-full",
               "host_work": str(self.host_work),
               "runtime": {"kind": "container", "image": IMAGE, "venv_lock_sha256": None},
               "framework": {"name": "nemo_rl", "version": NEMO_RL_COMMIT, "commit": NEMO_RL_COMMIT},
               "training_seed": SEED, "gpus": True, "argv": self.argv(), "mounts": self.mounts(self.out, self.spool),
               "tmpfs": ["/tmp"], "network": "none", "network_reason": None, "user": "1000:1000",
               "env_names": list(ENV_NAMES), "gpu_est_h": 2.5, "gpu_used_h": 30.0, "output_dir": str(self.out),
               "output_root": str(self.output_root),
               "gpu_step_sha256": file_sha256(self.host_work / "bin" / "gpu-step.sh"), "allow_network": False,
               "interpreter_path": TRAIN_PY, "interpreter_sha256": None,
               "script_path": f"{AREA10}/run_agentic_nemo.py",
               "script_sha256": file_sha256(self.runtime / "run_agentic_nemo.py"),
               "host_config_sha256": self.config_sha(), "sidecar": self.sidecar_record(self.spool)}
        rec.update(overrides)
        rec["label"] = launch_label(rec)
        return rec

    def rewrite_launch(self, **overrides) -> Path:
        rec = self.launch_record(**overrides)
        p = write_canonical(self.root / "launch.json", rec)
        self.label = rec["label"]
        self.guard_log()
        return p

    def new_run(self, name: str) -> tuple[Path, Path]:
        """`<runs>/<name>/{train or gen or score, spool}` new and empty (the decided sibling layout)."""
        run = self.runs / name
        out, spool = run / "out", run / "spool"
        out.mkdir(parents=True)
        spool.mkdir()
        return out, spool

    def request(self, out: Path, spool: Path, **overrides) -> dict[str, Any]:
        req = {"area": 10, "label": "mm-a10-agentic-smoke", "host_work": str(self.host_work),
               "runtime": {"kind": "container", "image": IMAGE},
               "framework": {"name": "nemo_rl", "version": NEMO_RL_COMMIT, "commit": NEMO_RL_COMMIT},
               "training_seed": SEED, "gpus": True, "network": "none", "network_reason": None, "user": "1000:1000",
               "env": dict(TRAIN_ENV), "mounts": self.mounts(out, spool), "tmpfs": ["/tmp"], "output_dir": str(out),
               "gpu_est_h": 0.3, "argv": self.argv("smoke"), "sidecar": self.sidecar_request(spool)}
        req.update(overrides)
        return req

    def eval_request(self, script: str, out: Path, *, spool: Path | None = None, gen: Path | None = None,
                     cut: Path | None = None, limit: str | None = None, **overrides) -> dict[str, Any]:
        """A generation (gen_agentic.py, vLLM worker venv, GPU, with the sidecar) or scoring (`-I score_agentic.py`,
        the image's main venv, CPU) request in the pinned image."""
        if script == "gen_agentic.py":
            mounts = [{"host": str(self.runtime), "container": AREA10, "mode": "ro"}]
            argv = [GEN_PY, f"{AREA10}/gen_agentic.py", "--model", f"{AREA10}/{BASE_REL}"]
            if cut is not None:
                mounts.append({"host": str(cut), "container": "/inputs/cut", "mode": "ro"})
                argv += ["--adapter", "/inputs/cut"]
            mounts += [{"host": str(out), "container": "/out", "mode": "rw"},
                       {"host": str(spool), "container": "/spool", "mode": "rw"}]
            argv += ["--items", f"{AREA10}/items-test", "--out", "/out", "--socket", "/spool/mathexec.sock",
                     "--eval-seeds", "11"]
            req = {"gpus": True, "gpu_est_h": 0.2, "env": dict(GEN_ENV), "argv": argv, "mounts": mounts,
                   "sidecar": self.sidecar_request(spool)}
        else:
            argv = [TRAIN_PY, "-I", f"{AREA10}/score_agentic.py", "--gen", "/inputs/gen", "--items",
                    f"{AREA10}/items-test", "--out", "/out", "--eval-seeds", "11"]
            req = {"gpus": False, "gpu_est_h": 0, "env": dict(SCORE_ENV), "argv": argv,
                   "mounts": [{"host": str(self.runtime), "container": AREA10, "mode": "ro"},
                              {"host": str(gen), "container": "/inputs/gen", "mode": "ro"},
                              {"host": str(out), "container": "/out", "mode": "rw"}]}
        if limit is not None:
            req["argv"] += ["--limit", limit]
        req.update({"area": 10, "label": f"mm-a10-agentic-{script[:-3].replace('_', '-')}",
                    "host_work": str(self.host_work), "runtime": {"kind": "container", "image": IMAGE},
                    "framework": {"name": "nemo_rl", "version": NEMO_RL_COMMIT, "commit": NEMO_RL_COMMIT},
                    "training_seed": 0, "network": "none", "network_reason": None, "user": "1000:1000",
                    "tmpfs": ["/tmp"], "output_dir": str(out)})
        req.update(overrides)
        return req

    def launch(self, req: dict, name: str, *, profile: str = "smoke", smoke: Path | None = None,
               extra: list[str] = ()):
        path = self.root / f"{name}.request.yaml"
        path.write_text(yaml.safe_dump(req, sort_keys=False), encoding="utf-8")
        out = self.root / f"{name}.launch.json"
        args = ["ext-launch", "--adapter", ADAPTER, "--profile", profile, "--request", str(path), "--gpu-log",
                str(self.host_work / "gpu-log.md"), "--out-record", str(out)]
        if smoke is not None:
            args += ["--smoke-record", str(smoke)]
        return run_cli(args + list(extra), cwd=self.root, env=self.env()), out

    def recheck(self, record: Path, extra: list[str] = ()):
        return run_cli(["ext-launch", "--adapter", ADAPTER, "--recheck", str(record), *extra], cwd=self.root,
                       env=self.env())

    # --- collect ---

    def collect(self, runs_root: Path, *, baseline: Path | None = None, candidate: Path | None = None,
                launch: Path | None = None, timeout: int = 300):
        args = ["ext-collect", str(self.root / "spec.yaml"), "--adapter", ADAPTER, "--stage-out", str(self.out),
                "--launch-record", str(launch or self.root / "launch.json"), "--smoke-record",
                str(self.root / "smoke.json"),
                "--eval-baseline", str(baseline or self.arms["base"] / "arm-exact_match"),
                "--eval-candidate", str(candidate or self.arms["cand"] / "arm-exact_match"), "--runs-root", str(runs_root)]
        return run_cli(args, cwd=self.root, timeout=timeout, env=self.env())

    def edit_json(self, path: Path, fn: Callable[[Any], None]) -> None:
        doc = json.loads(path.read_text(encoding="utf-8"))
        fn(doc)
        write_json(path, doc)

    def edit_jsonl(self, path: Path, fn: Callable[[list[dict]], None]) -> None:
        rows = read_jsonl(path)
        fn(rows)
        write_jsonl(path, rows)

    def reseal(self, cut: Path) -> None:
        """Rewrite cut.json's file hashes after a cut file was edited (the cut stays complete)."""
        self.edit_json(cut / "cut.json", lambda d: d.update(files={n: file_sha256(cut / n) for n in d["files"]}))


# ---------------------------------------------------------------------------
# Fixture copies and the p1-a3 cutter
# ---------------------------------------------------------------------------

def probe_copy(dst: Path) -> Path:
    """A writable copy of `fixtures/agentic/p1-a3/` (the cut run's outputs plus `data/shards/`)."""
    shutil.copytree(PROBE, dst)
    return dst


def sliding_copy(dst: Path, *, env_id: str = "sliding_puzzle", max_turns: int = 6) -> Path:
    """A writable copy of `fixtures/agentic/sliding-puzzle/` plus what the launcher would write around it:
    run_args.json (env sliding_puzzle, 2 steps of 2 prompts x 4 generations, 6 turns) and a minimal cut marker of step 2
    (the consumption join reads only its step and examples_consumed)."""
    shutil.copytree(SLIDING, dst)
    write_json(dst / "run_args.json", {"stage": "agentic_grpo", "env_id": env_id, "max_steps": 2,
                                       "max_rollout_turns": max_turns,
                                       "profile_settings": {"steps": 2, "prompts": 2, "generations": 4,
                                                            "save_period": 2}})
    write_json(dst / "cuts" / "checkpoint-2" / "cut.json", {"global_step": 2, "examples_consumed": 4, "files": {}})
    return dst


# p1-a3 step -> [(group in the step, [members kept])] (module docstring); prompts 2 x members 2 per step.
PROBE_SELECTION = {1: [(12, [0, 1]), (13, [1, 4])], 2: [(4, [1, 6]), (9, [1, 2])], 3: [(0, [3, 4]), (7, [1, 2])]}
PROBE_STANDINS = ("adapter_model.safetensors", "tokenizer.json", "vocab.json", "merges.txt")
PROBE_COPIES = ("run_args.json", "env-freeze.txt", "attempts.jsonl", "cuts.jsonl")
PROBE_DROPPED = ("advantages", "generation_logprobs", "prev_logprobs")
# The fixtures name the runtime by this neutral path instead of the host path the probe ran with (run_args.json, the
# cut's adapter_config.json; the real arm's gen_meta.json and summary.json).
NEUTRAL_RUNTIME = "/srv/ptc/nemo-rl-agentic-math/runtime"
# SOURCE.json leaves out the hashes of originals that can be rebuilt from the copies here once the host path is guessed
# (adapter_config.json differs only by that path; run_args.json also by the documented profile_settings rewrite; the
# cut's cut.json by its `files` hashes, all in SOURCE.json but adapter_config.json's, and examples_consumed): such a
# hash would confirm a guessed host path.
PROBE_UNHASHED = ("cuts/checkpoint-3/adapter_config.json", "cuts/checkpoint-3/cut.json", "run_args.json")


def neutral_runtime_paths(path: Path, recorded: str) -> None:
    """Replace the recorded runtime directory `recorded` by `NEUTRAL_RUNTIME` in the text of `path`."""
    path.write_text(path.read_text(encoding="utf-8").replace(f"{recorded}/", f"{NEUTRAL_RUNTIME}/"), encoding="utf-8")


def cut_probe(src: Path, dst: Path, train_jsonl: Path) -> dict:
    """Cut `fixtures/agentic/p1-a3/` from the real probe `src` (module docstring). Returns SOURCE.json."""
    G = 8
    env = read_jsonl(src / "env_calls.jsonl")
    firsts = sorted({x["traj_call"] for x in env})
    rows_all = {r["row_id"]: r for r in read_jsonl(train_jsonl)}
    keep: dict[tuple[int, int], int] = {}                   # (traj_call, traj_pos) -> new traj_pos
    picked_rows = []
    for k, sel in PROBE_SELECTION.items():
        n = 0
        for g, members in sel:
            for m in members:
                keep[(firsts[k - 1], g * G + m)] = n
                n += 1
            picked_rows.append(next(x["row_id"] for x in env if x["traj_call"] == firsts[k - 1]
                                    and x["traj_pos"] == g * G + members[0]))
    lines = [x for x in env if (x["traj_call"], x["traj_pos"]) in keep]
    calls = sorted({x["call"] for x in lines})
    new_call = {c: i + 1 for i, c in enumerate(calls)}
    out_env, pos = [], {}
    for x in lines:
        y = dict(x)
        y["traj_pos"] = keep[(x["traj_call"], x["traj_pos"])]
        y["call"], y["traj_call"] = new_call[x["call"]], new_call[x["traj_call"]]
        y["position"] = pos.get(y["call"], 0)
        pos[y["call"]] = y["position"] + 1
        out_env.append(y)
    dst.mkdir(parents=True)
    write_jsonl(dst / "env_calls.jsonl", out_env)
    write_jsonl(dst / "trajectories.jsonl", [{k: y[k] for k in ("call", "end_reason", "env_id", "response_sha256",
                                                                 "row_id", "traj_call", "traj_pos", "turn", "utc")}
                                             for y in out_env if y["terminated"]])
    logs = dst / "nemo" / "logs" / "exp_001"
    logs.mkdir(parents=True)
    for k in PROBE_SELECTION:
        src_lines = read_jsonl(src / "nemo" / "logs" / "exp_001" / f"train_data_step{k}.jsonl")
        chosen = sorted(((p, n) for (c, p), n in keep.items() if c == firsts[k - 1]), key=lambda t: t[1])
        text = ""
        for p, n in chosen:
            rec = {key: v for key, v in src_lines[p].items() if key not in PROBE_DROPPED}
            assert rec["idx"] == p
            L = rec["input_lengths"][0]
            rec["token_ids"] = [rec["token_ids"][0][:L]]
            rec["token_loss_mask"] = [rec["token_loss_mask"][0][:L]]
            rec["idx"] = n
            text += json.dumps(rec) + "\n"
        (logs / f"train_data_step{k}.jsonl").write_text(text, encoding="utf-8")
    for name in PROBE_COPIES:
        shutil.copyfile(src / name, dst / name)
    args = json.loads((dst / "run_args.json").read_text(encoding="utf-8"))
    args["profile_settings"] = dict(args["profile_settings"], prompts=2, generations=2)
    write_json(dst / "run_args.json", args)
    recorded = args["base_model"].rsplit("/models/", 1)[0]
    neutral_runtime_paths(dst / "run_args.json", recorded)
    shard = dst / "data" / "shards" / "shard-00000.jsonl"
    shard.parent.mkdir(parents=True)
    shard.write_text("".join(json.dumps(rows_all[r], sort_keys=True, separators=(",", ":")) + "\n" for r in picked_rows),
                     encoding="utf-8")
    cut_src, cut_dst = src / "cuts" / "checkpoint-3", dst / "cuts" / "checkpoint-3"
    cut_dst.mkdir(parents=True)
    for f in sorted(cut_src.iterdir()):
        if f.name in PROBE_STANDINS:
            data = (safetensors_bytes(f"p1-a3 {f.name} stand-in") if f.name.endswith(".safetensors")
                    else f"p1-a3 {f.name} stand-in\n".encode())
            (cut_dst / f.name).write_bytes(data)
        elif f.name != "cut.json":
            shutil.copyfile(f, cut_dst / f.name)
    neutral_runtime_paths(cut_dst / "adapter_config.json", recorded)
    cut = json.loads((cut_src / "cut.json").read_text(encoding="utf-8"))
    cut["files"] = {n: file_sha256(cut_dst / n) for n in cut["files"]}
    cut["examples_consumed"] = 2 * cut["global_step"]
    write_json(cut_dst / "cut.json", cut)
    source = {"run": "p1-a3", "source_dir": "out/area10-agentic/p1-a3/train", "cutter": "tests/_nemo_rl_agentic.py cut_probe",
              "selection": {str(k): v for k, v in PROBE_SELECTION.items()}, "prompts": 2, "generations": 2,
              "dropped_fields": list(PROBE_DROPPED), "token_arrays": "cut to input_lengths",
              "stand_ins": list(PROBE_STANDINS), "rewritten": ["run_args.json profile_settings prompts/generations",
                                                               "cuts/checkpoint-3/cut.json files, examples_consumed",
                                                               "run_args.json, cuts/checkpoint-3/adapter_config.json: "
                                                               f"the runtime directory as {NEUTRAL_RUNTIME}"],
              "source_sha256": {n: file_sha256(src / n) for n in ("env_calls.jsonl", "trajectories.jsonl",
                                                                  *PROBE_COPIES)}
              | {f"nemo/logs/exp_001/train_data_step{k}.jsonl":
                 file_sha256(src / "nemo" / "logs" / "exp_001" / f"train_data_step{k}.jsonl") for k in PROBE_SELECTION}
              | {f"cuts/checkpoint-3/{n}": file_sha256(cut_src / n) for n in sorted(cut["files"]) + ["cut.json"]}}
    for name in PROBE_UNHASHED:
        del source["source_sha256"][name]
    write_json(dst / "SOURCE.json", source)
    return source
