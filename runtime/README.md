# Area "agentic RL": multi-turn NeMo RL GRPO with a Python tool on math (prototype, unpromotable)

Research draft, CPU only, written 2026-11-14 before anything was downloaded or run. Tags: **prototype**,
**unpromotable**. Scope (the maintainers' decision "Area 10 (agentic RL) direction", not published): NeMo RL multi-turn
GRPO (`run_multi_turn_rollout`), policy `Qwen/Qwen3-1.7B`, environment as configuration, first environment NeMo RL
`code_environment` (multi-turn Python execution) on math with exact-match reward, `sliding_puzzle` as the second
environment for CPU tests, one gatedtrain adapter `nemo_rl_agentic` parameterised by environment, consumption one row
per trajectory with model and environment tokens distinguished, one evaluator and item set per environment, a
registered gate binds one environment.

Sections 1-9: every fact cites a file and line of a pinned source or a URL with its read date. **Unverified**
marks an estimate or a claim not checked against a source. Section 0 records what was set up afterwards.

Pinned sources read:
- A NeMo RL clone at `4f39fd86289a2ffd5aaf5d202702f7fd7185c924` (`git rev-parse HEAD`, 2026-11-14). Paths below
  without a prefix are relative to this clone.
- Area 7's runtime (the VLM recipe, `gatedtrain-recipe-nemo-rl-vlm-clevr`, `runtime/`: README "Implementation (as run)",
  `run_vlm_nemo.py`, `vlm_env_logged.py`) and its full run's launch record (`train-full-1.{cmd,launch.json}`).
- gatedtrain's `MULTIMODAL-SPEC.md` and `ext/registry.py` (rows `nemo_rl_ext`, `nemo_rl_vlm`) as of 2026-11-14, and the
  design set (`gatedtrain-design`) `05-evaluation-and-lineage.md` (R15-1 :255, R16-1 :257).
- Hugging Face Hub API and files at pinned revisions, read 2026-11-14 with `env -u HF_TOKEN curl` (metadata, model
  card, `config.json`, `tokenizer_config.json`, `generation_config.json`, dataset cards only; no weights or data).

## 0. Runtime as set up (2026-11-14, CPU only)

Prepared on the host after the maintainers chose the two downloads (decision "Area 10 decisions", not published).
No GPU job, no docker command, no install was run. Sections 1-9 below are the research draft; where they differ
from this section (module names, D3, LoRA targets), this section is what exists.

### 0.1 Downloads (network used only for these two repositories)

| Item | Result |
|---|---|
| `fetch_models.py` | `Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` -> `models/qwen3-1.7b/`: 12 plain files, 4,079,450,110 bytes (equal to the Hub listing); `models/MODELS.json` with per-file SHA-256; the SHA-256 of both safetensors files and `tokenizer.json` equal the Hub's LFS etags kept in `hf-meta/qwen3-1.7b/` (`169ad53e...`, `912becff...`, `aeb13307...`). |
| `fetch_data.py` | `EleutherAI/hendrycks_math` @ `21a5633873b6a120296cce3e2df9d5550074f4a3` -> `raw/hendrycks_math/`: 16 files (14 parquet, README.md, .gitattributes), 4,890,090 bytes (equal to the README listing); `raw/RAW.json`. |
| Token handling | both run as `env -u HF_TOKEN <host-python> -I ...`, `token=False`, `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, fresh `HF_HOME=hf-home/` (left empty), Xet off; hub metadata moved to `hf-meta/`. `--verify` re-hashes: both ok. |

### 0.2 Data (`make_data.py`, seed 20261115, area-4 venv; two runs byte-identical)

- **Integer rule** (`make_data.integer_answer`, exact): answer = content of the last `\boxed{...}` / `\fbox{...}`
  (balanced braces) of `solution`; delete `$`, `\!`, `\,`, `\;`, `\ `, spaces; `{,}` -> `,`; a comma-grouped number
  loses its commas; the rest must be `[+-]?\d+` with an optional zero fraction; stored as `str(int(...))`.
  Yield: test 3,144 of 5,000 (62.9 %), train 4,764 of 7,500 (63.5 %); every solution has a `\boxed`.
- **Items** `items-test/items.jsonl`: 500 integer-answer test problems, largest-remainder allocation over the 35
  (subject, level) cells, seeded sample per cell; by subject algebra 135, counting_and_probability 48, geometry 36,
  intermediate_algebra 79, number_theory 77, prealgebra 99, precalculus 26; by level 53 / 95 / 116 / 120 / 116.
  `items_sha256` `sha256:cc252796f377009fbac1aa15bfedb6a3e9b37099725991c0cfed54780e5e435a`.
- **Training rows** `data/shards/shard-0000{0..3}.jsonl` (500 each) and `data/train.jsonl` (their concatenation,
  `sha256:56e9bf87889da029e2b4220360535735d62887cbcabe47c10cbb7007fd6efc69`): 2,000 integer-answer train problems,
  stratified by subject (algebra 517, c&p 212, geometry 202, int. algebra 287, number theory 331, prealgebra 356,
  precalculus 95), permuted once; row ids `math-train-00000..01999`; row `{row_id, input, messages, answer, subject,
  level, source}`.
- **Overlap** (0 required, checked by make_data.py and tests/test_data.py): training rows vs items by exact text 0
  and by normalized text (lower case, `$` and whitespace removed) 0; duplicate item texts 0. Training candidates equal
  (exact or normalized) to any of the 5,000 test problems were dropped first: 3.
- **Prompt** `prompt_math_code.txt` (one user turn, `enable_thinking=false`; the template ends with the empty think
  block): tokens per prompt train min 86 / median 125 / p99 624 / max 1,370; items max 669 (Qwen3 tokenizer).
- All hashes in `data/data_meta.json` and `items-test/item_set_meta.json`.

### 0.3 Files

| File | Lines | Role |
|---|---|---|
| `fetch_models.py`, `fetch_data.py` | 123, 81 | downloads (0.1) |
| `make_data.py`, `prompt_math_code.txt` | 264, 2 | data (0.2); the prompt file is NeMo RL's `prompt_file` (`{}` = problem); P1 p1-a2's prompt, kept as its record |
| `prompt_math_code_oneshot.txt` | 14 | the deployment prompt since 0.11 (training and evaluation): probe-tool-2's `text_oneshot` |
| `agentic_core.py` | 140 | turn protocol, observation, stateless-replay bookkeeping, reward gate; shared by trainer env and evaluator |
| `math_code_env.py` | 116 | NeMo RL processor `math_code_rowid` and Ray actor `MathCodeEnvironment` (env `math_code`) |
| `traj_log.py` | 132 | environment-agnostic trajectory stamp and `env_calls.jsonl` / `trajectories.jsonl`; `LoggedSlidingPuzzleEnv` |
| `consumption.py` | 247 | one row per trajectory: env log joined with `train_data_step<N>.jsonl` (refuses on any mismatch; logged shape in 0.10) |
| `probe_tool_use.py` | 452 | tool-use probe of prompt / protocol variants with the base model (0.10) |
| `executor/executor_server.py`, `sandbox_runner.py`, `exec_protocol.py`, `run-executor.sh` | 391, 210, 96, 45 | executor sidecar (0.5) |
| `run_agentic_nemo.py`, `ptc_agentic_hooks.py` | 260, 204 | launcher (0.6), cut writer and run record |
| `gen_agentic.py`, `score_agentic.py` | 294, 328 | evaluator (0.7) |
| `tests/` | 11 files, 1,603 | CPU tests (0.8, 0.11) |

### 0.4 Turn protocol and environment (`agentic_core.py`, `math_code_env.py`)

- Generation stops at `</code>` (per-datum `stop_strings`, which the synchronous rollout applies at every turn,
  rollouts.py:671-672). A turn ending in `</code>` with a `<code>` before it is a tool call (code = text between the
  last `<code>` and the final `</code>`, Markdown fence stripped, dedented). Below turn 4 it runs in the executor and
  the observation `<result>\n<output>\n</result>\n` (output cut to 512 characters, head and tail; `(no output)`;
  `TimeoutError: ...`; exception line `Type (line N): message`) is appended as raw tokens. Any other turn is final
  (`final`); a tool call at turn 4 is not run (`max_turns`); both terminal cases return an empty observation.
- Stateless replay: the metadata keeps the trajectory's cells `{code, status}`; each tool call sends all earlier cells
  except `syntax_error` / `timeout` / `crash` ones plus the new cell; earlier output is discarded, earlier
  exceptions ignored; `random.seed(0)` makes replays repeat.
- Reward (terminal turn only): NeMo RL's `HFVerifyWorker.verify` (in-process, `math_verify_impl="hf_math_verify"`
  passed explicitly: without it the call returns 0 for every sample, math_environment.py:131-146) on the final
  assistant turn's text against the integer reference, and only if that text contains `\boxed` (the evaluator scores
  only boxed answers); 0 otherwise. Checked with math-verify 0.9.0 (the image's version) in
  tests/test_reward_math_verify.py.
- Every env call appends one fsynced line per sample to `<out>/env_calls.jsonl` (`env_id, call, position, traj_call,
  traj_pos, turn, row_id, reward, terminated, end_reason, tool_status, obs_chars, response_sha256, utc`) and one line
  per finished trajectory to `<out>/trajectories.jsonl`. Trajectory identity: the turn-0 call and position, stamped
  into the returned metadata under `_ptc` (traj_log.py docstring). An executor fault raises (the step fails).

### 0.5 Executor sidecar (`executor/`)

- Decision D3 (a) as area 6: generated code runs only in a separate container that mounts exactly
  `area10-agentic/executor` (ro) and one spool directory (rw, holds `mathexec.sock`); never the area root, `items-test/`,
  `data/`, `models/` or a run's `/out` (MM-34a). Wire format: area 6's (8-byte length + JSON over a Unix socket, one
  batch per connection); ops `ping`, `run` (jobs `{id, cells}`).
- Per job: a fresh `python -s -P sandbox_runner.py` in its own session, cwd a fresh directory under `/tmp` (tmpfs),
  environment built by the server (no inheritance), stdin = job JSON then `/dev/null`, raw fds 1/2 to `/dev/null`
  (results go through a private fd); rlimits with soft = hard: CPU 10 s, address space 2 GiB, file size 1 MiB, 64
  open files, no core, `RLIMIT_NPROC` = the uid's current tasks + 32; wall clock 10 s for the whole replay (process
  group killed); 8 jobs in parallel; at most 16,384 output characters kept (total counted). Statuses `ok`, `error`,
  `syntax_error`, `timeout`, `crash` are observations. A Python audit hook (defence in depth, not a boundary) refuses
  sockets, subprocesses, exec/spawn/fork, kill, writes outside the work directory. The server warms numpy/sympy
  imports before it listens (cold page cache: 57 s on the test host, warm 0.2 s).
- Trust boundary (M2; executor_server.py docstring), independent of the audit hook, which ctypes bypasses: a job's
  stdin/stdout/stderr are socket pairs of that job (not reopenable through `/proc/<pid>/fd`, unlike pipes), it
  inherits no other descriptor; the server and every runner are non-dumpable (no `/proc` fd access, memory access,
  ptrace or `pidfd_getfd` by jobs of the same uid); the server is a child subreaper and kills every process a job
  left behind before it answers a batch; a socket path replaced during a batch makes the server answer
  `executor_error` and stop. Jobs can still signal each other or the server (shared uid): a failure, never a forged
  result.
- Image and interpreter: the NeMo RL image of areas 6/7 (`sha256:7a4078e7...`), `/opt/nemo_rl_venv/bin/python -I`
  (Python 3.13.14; numpy 2.4.6 and sympy 1.14.0 importable by model code; area 7 env-freeze). Container flags as
  area 6's B18 executor: `--network none --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges --read-only
  --tmpfs /tmp:rw,nosuid,nodev,noexec,size=4g --cpus 8 --memory 8g --memory-swap 8g --pids-limit 256 --ulimit
  nofile=1024:1024 --ulimit core=0`. `executor/run-executor.sh serve <spool> <label>` prints the commands.
- Note: `RLIMIT_NPROC` counts all tasks of uid 1000 on the host kernel, while the container's `/proc` shows only its
  own; inside the container the margin is therefore effectively 0 (no new threads or processes for model code).
  Accepted: the audit hook refuses fork/subprocess anyway; a model that starts a thread sees an error observation.

### 0.6 Launcher (`run_agentic_nemo.py`, `ptc_agentic_hooks.py`)

Calls the unmodified `examples/run_grpo.py` `main()` with `examples/configs/grpo_math_1B.yaml` plus overrides
(checked against the pinned yaml with hydra 1.3.8 in tests/test_launcher.py); registers `math_code_rowid` and env
`math_code` (`register_env`, `ACTOR_ENVIRONMENT_REGISTRY[fqn] = PY_EXECUTABLES.SYSTEM`); pings the executor and runs
`print(6 * 7)` there before anything else (waits up to 15 min for the warm-up). Settings: profiles `probe` 3 steps
(cut 3), `smoke` 5 (cut 5), `full` 60 (cuts 20/40/60; `--max-steps 100` for D5 b), 16 prompts x 8 generations,
`grpo.max_rollout_turns` 4, `max_total_sequence_length` 4,096 (was 2,048 until 0.11), 1,024 new tokens per turn
(was 512) with `cap_max_tokens_to_context`, prompt `prompt_math_code_oneshot.txt` (0.11), thinking off, KL 0 with the reference pass skipped, `data_plane.enabled=false`, data in
file order, sequence packing off, micro batches of 1, Automodel backend torch/SDPA/no rope fusion (area 7), lr 2e-5
with a 10-step warm-up, seed 10100. LoRA rank 32, alpha 64 on every decoder projection (`target_modules: ['*_proj']`:
q, k, v, o, gate, up, down, about 35 M parameters; not `match_all_linear`, which would also wrap the tied lm_head,
Automodel module_matcher.py:118-124). Memory rungs: `a` plain AdamW, vLLM 0.35, logprob chunk 512 (since 0.11);
`b` TE FusedAdam fp32 master + bf16 moments, logprob chunk 512, vLLM 0.4; `c` full fine-tune (outside the decision),
1,536 tokens, vLLM 0.15 (refused by the launcher since 0.11 until re-planned).
Not set: `PYTORCH_CUDA_ALLOC_CONF` (area 7 probe-b4).

Cuts `<out>/cuts/checkpoint-<N>/` (`cut.json` last, schema `ptc_mm.agentic_cut`, kind `lora_adapter`): hard links to
Automodel's PEFT files (`adapter_model.safetensors`, `adapter_config.json` checked for r 32 / alpha 64,
`automodel_peft_config.json`), tokenizer copies, `sampler.json`, `base_ref.json` (base repo, revision, file hashes).
No merged weights: the evaluator loads base + adapter with vLLM's LoRA support, the same path NeMo RL's native
refit uses during training. Also `run_args.json` (overrides, code and data hashes, executor ping incl. its code
hashes and limits), `env-freeze.txt`, `attempts.jsonl`, `cuts.jsonl`.

### 0.7 Evaluator (`gen_agentic.py`, `score_agentic.py`)

`gen_agentic.py` (vLLM worker venv, GPU) replays the training loop exactly (prompt, chat template, raw-token
observations, 1,024-token turns capped to 4,096 (0.11), stop `</code>`, overflow cut) with greedy decoding and the same
`agentic_core` and executor sidecar; `--adapter` loads a LoRA cut (`cut.json` and `base_ref.json` hashes checked, and
its `sampler.json` must match the evaluator's prompt hash, budget and stop string);
`--limit N` takes N items evenly spaced (all subjects). `score_agentic.py` (standard library, `-I`, runs no generated
code): own parser (last `\boxed{}` of the final assistant turn, normalized, integer), outcomes `correct`,
`wrong_answer`, `no_answer`, `max_turns`, `overflow`; errors `missing_output`, `sandbox_failure`, `scorer_error`;
sub-metrics `turns_used, tool_calls, tool_errors, tool_timeouts, answer_found, truncated, model_tokens, env_tokens`;
MM-1 arm `arm-exact_match/` as area 7, evaluator id `agentic.math_code_items@0.1`, adapter `nemo_rl_agentic`.

### 0.8 Verified on CPU (host) and what remains for the GPU

Host tests (`tests/`, 62 tests since 0.11; run from the area directory):

```
<host-python> -m unittest discover -s tests -t tests   # 62, 2 skipped
<venv-with-math-verify>/bin/python -m unittest discover -s tests -t tests -p test_reward_math_verify.py
<venv-with-hydra>/bin/python -m unittest discover -s tests -t tests -p test_launcher.py
```

Two environment variables (tests/_skips.py; added after the counts above, 2026-11-20):
- `GATEDTRAIN_RUNTIME_TESTS_HOST_EXECUTOR=1`: the executor runs the host venv's Python, which may lack numpy and
  sympy; `test_executor`'s numpy/sympy check is then skipped. Without it the check runs and fails when either is
  missing, as it must in the sidecar's NeMo RL image (which has both).
- `GATEDTRAIN_RUNTIME_TESTS_STRICT=1`, for the GPU host where every heavy input is present: a test that would skip
  because the base tokenizer, the training data, the NeMo RL clone or torch is absent fails instead, naming the input
  (a skipped class fails in its setUpClass), and the host-executor opt-out is ignored. Skips for packages a venv lacks
  (tokenizers, transformers, hydra, math_verify) and for absent run outputs stay skips.

All pass (2026-11-14). Covered: executor statuses and limits (wall and CPU timeout, 3 GiB allocation, 2 MiB file,
socket / subprocess / os.system / fork / kill / write outside refused, raw-fd writes cannot corrupt the result, hard
exit = crash, output cap), replay determinism and skipping, the Unix-socket server (ping, run, protocol errors, socket
mode 0700); turn parsing, termination, observations, reward gate; the Ray actor glue with stand-in `ray`/`nemo_rl`
modules (EnvironmentReturn, metadata stamps, logs); the evaluator loop incl. overflow; consumption joins for
**math_code and NeMo RL's real sliding_puzzle game** (pinned clone, loaded read-only) through the same
`TrajectoryLogger` and `consumption.build`, with refusals (reward, row order, missing step, too many steps);
data invariants; scorer self-test and an MM-1 arm; launcher overrides against the pinned yaml; the LoRA cut writer.
The NeMo RL rollout itself is a test double (`tests/_support.fake_rollout`), not NeMo RL.

Open until the GPU / docker runs (none verified here):
1. Peak GPU memory of rung a (estimate 15-19 GB) and host RAM (trainer + sidecar; area 7 peaked 25.4 GiB).
2. LoRA with Automodel's Qwen3 on the torch/SDPA backend, `target_modules ['*_proj']`, and NeMo RL's native LoRA
   refit into vLLM; LoRA lr 2e-5 is a guess.
3. Automodel's `adapter_model.safetensors` key names load in vLLM 0.29 `LoRARequest` (evaluation of a cut).
4. The executor sidecar in the NeMo RL image with `--read-only` (image entrypoint, Python startup) and the Unix
   socket between two containers (area 6 used the same pattern).
5. NeMo RL loads `data/train.jsonl` with the extra columns (`source` dict) through ResponseDataset; `math_code`
   created from `ENV_REGISTRY`; the real `train_data_step<N>.jsonl` order matches the env call order (consumption
   join, as area 7's 640/640).
6. Step time (estimate 90-130 s) and the base accuracy on the 20-item smoke arm (MM-38: must be in (0, 1)).

### 0.9 Commands (as planned before the GPU runs)

Common: `AREA=<runtime>`, outputs under `<output-root>/area10-agentic/`,
image `sha256:7a4078e70b28cc6de622fa3cf25c7596c52337fa3f1779488b402c5fa1c74e69`. Every GPU or evaluation step is a
pair: start the sidecar on a new empty spool, wait for `listening` in its log, run the paired container, stop and
remove the sidecar. GPU steps run through the operator's `gpu-step.sh` wrapper (`<HOST_WORK>/bin/`), which logs
GPU time to `<HOST_WORK>/gpu-log.md`; `GATEDTRAIN_GPU_CAP_H` is the operator's GPU-hour budget ceiling and `GATEDTRAIN_GPU_USED_H`
the running total of that log at launch. `<host-python>` is any host Python 3.13 venv with pyarrow (and torch and
transformers for the tokenizer tests).

**P1 memory ladder, rung a (probe: 3 steps, cut 3; GPU estimate 0.3 h):**

```
mkdir -p <output-root>/area10-agentic/p1-a/train <output-root>/area10-agentic/p1-a/spool
docker --context gatedtrain run -d --name gatedtrain-mathexec-mm-a10-agentic-p1-a --network none --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges --read-only --tmpfs /tmp:rw,nosuid,nodev,noexec,size=4g --cpus 8 --memory 8g --memory-swap 8g --pids-limit 256 --ulimit nofile=1024:1024 --ulimit core=0 -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -v <runtime>/executor:<runtime>/executor:ro -v <output-root>/area10-agentic/p1-a/spool:/spool:rw sha256:7a4078e70b28cc6de622fa3cf25c7596c52337fa3f1779488b402c5fa1c74e69 /opt/nemo_rl_venv/bin/python -I <runtime>/executor/executor_server.py --serve /spool/mathexec.sock --workers 8
docker --context gatedtrain logs gatedtrain-mathexec-mm-a10-agentic-p1-a
GATEDTRAIN_GPU_CAP_H=<cap> GATEDTRAIN_GPU_USED_H=<total> GATEDTRAIN_GPU_STEP_EST_H=0.3 <HOST_WORK>/bin/gpu-step.sh mm-a10-agentic-p1-a --rm --runtime=nvidia --gpus all --network none --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges --shm-size=16g --ulimit memlock=-1 --ulimit nofile=65535:65535 -w /opt/nemo-rl -e FLASHINFER_CUDA_ARCH_LIST=8.9 -e HF_HUB_OFFLINE=1 -e HOME=/tmp/home -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e RAY_memory_monitor_refresh_ms=0 -e TORCH_CUDA_ARCH_LIST=8.9 -v <runtime>:<runtime>:ro -v <output-root>/area10-agentic/p1-a/train:/out:rw -v <output-root>/area10-agentic/p1-a/spool:/spool:rw --tmpfs /tmp:rw,exec,nosuid,nodev,size=8g sha256:7a4078e70b28cc6de622fa3cf25c7596c52337fa3f1779488b402c5fa1c74e69 /opt/nemo_rl_venv/bin/python <runtime>/run_agentic_nemo.py --profile probe --out /out --seed 10100 --base <runtime>/models/qwen3-1.7b --memory a --socket /spool/mathexec.sock
docker --context gatedtrain stop gatedtrain-mathexec-mm-a10-agentic-p1-a
docker --context gatedtrain rm gatedtrain-mathexec-mm-a10-agentic-p1-a
<host-python> -I <runtime>/consumption.py --env-calls <output-root>/area10-agentic/p1-a/train/env_calls.jsonl --train-logs <output-root>/area10-agentic/p1-a/train/nemo/logs --train-file <runtime>/data/train.jsonl --max-steps 3 --run p1-a --out <output-root>/area10-agentic/p1-a/consumption.jsonl
```

Pass: peak <= 22.5 GB, 3 steps, `cuts/checkpoint-3/cut.json`, consumption 384 rows. Rung b: the same with `p1-b`,
`--memory b`; rung c only if full weights are re-planned.

**Smoke** (5 steps, cut 5; then a 20-item base arm and a 20-item cut arm): training as P1 with `smoke-1`,
`--profile smoke --memory <rung>`, label `mm-a10-agentic-smoke-1`, estimate 0.25 h. Each evaluation arm
(`eval-smoke-base-1`, `eval-smoke-cut-1`): sidecar on `<arm>/spool` (as above, container name
`gatedtrain-mathexec-mm-a10-agentic-<arm>`), then generation:

```
GATEDTRAIN_GPU_CAP_H=<cap> GATEDTRAIN_GPU_USED_H=<total> GATEDTRAIN_GPU_STEP_EST_H=0.1 <HOST_WORK>/bin/gpu-step.sh mm-a10-agentic-eval-smoke-base-1-gen --rm --runtime=nvidia --gpus all --network none --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges --shm-size=16g --ulimit memlock=-1 --ulimit nofile=65535:65535 -w /opt/nemo-rl -e HF_HUB_OFFLINE=1 -e HOME=/tmp/home -e PYTHONUNBUFFERED=1 -v <runtime>:<runtime>:ro -v <output-root>/area10-agentic/eval-smoke-base-1/gen:/out:rw -v <output-root>/area10-agentic/eval-smoke-base-1/spool:/spool:rw --tmpfs /tmp:rw,exec,nosuid,nodev,size=8g sha256:7a4078e70b28cc6de622fa3cf25c7596c52337fa3f1779488b402c5fa1c74e69 /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python <runtime>/gen_agentic.py --model <runtime>/models/qwen3-1.7b --items <runtime>/items-test --out /out --socket /spool/mathexec.sock --eval-seeds 11 --limit 20
```

The cut arm adds `-v <output-root>/area10-agentic/smoke-1/train/cuts/checkpoint-5:/inputs/cut:ro`
and `--adapter /inputs/cut`. Scoring (CPU, no sidecar, runs no generated code):

```
GATEDTRAIN_GPU_CAP_H=<cap> GATEDTRAIN_GPU_USED_H=<total> GATEDTRAIN_GPU_STEP_EST_H=0.01 <HOST_WORK>/bin/gpu-step.sh mm-a10-agentic-eval-smoke-base-1-score --rm --network none --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges --shm-size=8g -e HF_HUB_OFFLINE=1 -e HOME=/tmp/home -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -v <runtime>:<runtime>:ro -v <output-root>/area10-agentic/eval-smoke-base-1/gen:/inputs/gen:ro -v <output-root>/area10-agentic/eval-smoke-base-1/score:/out:rw --tmpfs /tmp:rw,exec,nosuid,nodev,size=8g sha256:7a4078e70b28cc6de622fa3cf25c7596c52337fa3f1779488b402c5fa1c74e69 /opt/nemo_rl_venv/bin/python -I <runtime>/score_agentic.py --gen /inputs/gen --items <runtime>/items-test --out /out --eval-seeds 11 --limit 20
```

**Full**: as smoke with `full-1`, `--profile full` (60 steps) or `--profile full --max-steps 100` (D5 b), estimate
1.8-3.3 h. **Evaluations**: base (`eval-base-1`, `--model` only) and candidate (`eval-cand-1`, `--adapter` = the
gated cut) on all 500 items (no `--limit`), each with its own sidecar, generation estimate 0.2 h, then scoring as
above without `--limit`. Consumption after each training run: `consumption.py ... --max-steps <steps> --run <run>`.

### 0.10 P1 probe p1-a2 findings and fixes (2026-11-15, CPU analysis of the run's logs)

Run `out/area10-agentic/p1-a2/train` (rung a, 3 steps, peak GPU 11.0 GB, LoRA native refit, `checkpoint-3`, reward
mean 0.47). Findings and what changed:

1. **Consumption join on real logs.** `train_data_step<N>.jsonl` is written with `make_microbatch_iterator(1)`
   (utils/logger.py:1427-1441): every field is a batch of one (`rewards` `[r]`, `input_lengths` `[L]`, `content`
   `[[prompt, assistant, observation, ...]]`, `token_ids` / `token_loss_mask` / `advantages` / logprobs `[[...]]`
   right-padded to the step's longest sequence) plus `idx` = line number. The test double had flat fields, so
   `f32(list)` raised. `consumption.logged_sample` now unwraps and checks the shape (refuses anything else), and every
   turn gets `eos` (last generated token is an EOS id). `env_calls.jsonl` has one line per sample per env call (per
   turn), `trajectories.jsonl` one per trajectory: 385 = 384 trajectories + 1 second turn (step 1 position 65, the
   only executed tool call); no duplicate. Join: 384 rows, prompt / model / env tokens 51,904 / 151,703 / 9, equal to
   an independent count of the masks. Tests: `tests/fixtures/p1-a2/` (12 trajectories cut from the real files by
   `tests/cut_fixtures.py`, provenance in `SOURCE.json`), exact per-turn counts, a tokenizer check that the derived
   segments decode to the logged texts, shape refusals, and the full run when present.
2. **The `</code>` stop string never applied in training.** With `policy.generation.stop_strings` null NeMo RL starts
   vLLM with `skip_tokenizer_init=True` (models/generation/__init__.py:227-237); without a tokenizer vLLM's
   detokenizer is a no-op and stop strings are not matched (vLLM v1/engine/detokenizer.py,
   `IncrementalDetokenizer.from_new_request`; read in vLLM 0.31 of the area-1 venv, the image has 0.29). Evidence: 7
   of the 8 replies containing `</code>` ran on past it, mostly writing their own `<result>`; the one tool call that
   ran ended with `</code><|im_end|>` (EOS, not the stop string). The evaluator (`gen_agentic.py`, vLLM with a
   tokenizer) did stop, so training and evaluation differed. Fix: the launcher sets
   `policy.generation.stop_strings=["</code>"]` (test_launcher checks it against the pinned yaml).
3. **Truncation.** 180 of 384 final turns did not end at EOS (cut at 512 new tokens; prose reasoning, thinking off);
   179 had no `\boxed` and reward 0. The environment called all of them `final`; it now returns `no_answer` for a
   terminal turn without `\boxed` (`final` keeps meaning "answered"); the reward is unchanged.
4. **Tool use is rare** (`<code>` in 6 / 2 / 1 of 128 replies; the model computes in prose and at most "verifies"
   with code). `probe_tool_use.py` measures prompt and protocol variants with the base model before more training:
   see its docstring; CPU tests in `tests/test_probe.py`.

**Tool-use probe** (GPU, vLLM only, about 0.2-0.3 h; output `out/area10-agentic/probe-tool-1/`): sidecar as in 0.9
on `probe-tool-1/spool` (container `gatedtrain-mathexec-mm-a10-agentic-probe-tool-1`), then

```
GATEDTRAIN_GPU_CAP_H=<cap> GATEDTRAIN_GPU_USED_H=<total> GATEDTRAIN_GPU_STEP_EST_H=0.3 <HOST_WORK>/bin/gpu-step.sh mm-a10-agentic-probe-tool-1 --rm --runtime=nvidia --gpus all --network none --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges --shm-size=16g --ulimit memlock=-1 --ulimit nofile=65535:65535 -w /opt/nemo-rl -e FLASHINFER_CUDA_ARCH_LIST=8.9 -e TORCH_CUDA_ARCH_LIST=8.9 -e HF_HUB_OFFLINE=1 -e HOME=/tmp/home -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -v <runtime>:<runtime>:ro -v <output-root>/area10-agentic/probe-tool-1/probe:/out:rw -v <output-root>/area10-agentic/probe-tool-1/spool:/spool:rw --tmpfs /tmp:rw,exec,nosuid,nodev,size=8g sha256:7a4078e70b28cc6de622fa3cf25c7596c52337fa3f1779488b402c5fa1c74e69 /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python <runtime>/probe_tool_use.py --model <runtime>/models/qwen3-1.7b --out /out --socket /spool/mathexec.sock --n 64 --samples 8
```

Optional second run `probe-tool-2` with `--max-new-tokens 1024 --max-model-len 4096` (truncation at a larger turn
budget). After a variant is chosen: a P1 re-run (`p1-a3`) to confirm the stop-string fix in the trainer (tool turns
must end at `</code>` with `eos` false in the consumption rows); see 0.11.

### 0.11 Deployment prompt and turn budget (2026-11-15, after probe-tool-1/2; CPU only)

**Decision** (maintainers, rule of 0.10: exact match not clearly below `text_base`, then the highest tool share):
`text_oneshot` at 1,024 new tokens per turn, context 4,096, for training and evaluation. probe-tool-2 (64 training
problems x 8 samples, temperature 1.0, 1,024 / 4,096; `out/area10-agentic/probe-tool-2/probe/summary.json`):

| Variant | Tool share | Exact match | Truncated | Note |
|---|---|---|---|---|
| text_base | 0.02 | 0.71 | 0.11 | control (the P1 prompt) |
| **text_oneshot** | **0.44** | **0.72** | **0.08** | chosen; tool calls 215 ok / 40 error / 1 syntax error; no overflow |
| text_system | 0.63 | 0.62 | 0.04 | exact match clearly lower |
| native | 0.56 | 0.58 | 0.03 | exact match clearly lower |
| native_fewshot | 0.88 | 0.43 | 0.01 | exact match clearly lower |

**Prompt file.** `prompt_math_code_oneshot.txt` (860 bytes,
`sha256:3b7145c1edf2526dcd13d5477d343d68e883b30c8e20d1a5c5e3f19af7b6230f`) is `probe_tool_use.TEXT_ONESHOT` in NeMo
RL's `prompt_file` form (a `str.format` template: `{}` = problem, the literal braces of `\boxed{answer}` and
`\boxed{263}` doubled), written from the constant by a script, not retyped. Added next to `prompt_math_code.txt`
instead of replacing it: the old file is hashed in `data/data_meta.json` (tests/test_data.py), in P1 p1-a2's
`run_args.json`, and is the probes' `text_base` arm (`probe_tool_use.TRAIN_PROMPT`); replacing it would silently
change what those records mean. `probe_tool_use.py` is unchanged (its SHA-256 `ec6ed3c5...` is the `code_sha256` of
both probe runs), so the probes stay reproducible. Checked (tests/test_prompt.py): the file formats to exactly
`TEXT_ONESHOT.replace("{problem}", p)` for all 2,000 training rows, all 500 items and brace-laden strings; the
rendered chat prompt equals `probe_tool_use.render_prompt(tok, "text_oneshot", p)` and its token count equals the
recorded `prompt_tokens` for all 64 probe-tool-2 rows; the worked example, run by `MathCodeCore` through the real
executor, yields exactly the `<result>\n263\n</result>\n` block the prompt shows, and its answer turn is `final` with
reward 1.

Used by: `gen_agentic.PROMPT_FILE` (evaluator) and `run_agentic_nemo.PROMPT = gen_agentic.PROMPT_FILE`
(`data.default.prompt_file`, processor `math_code_rowid`). Recorded: `run_args.json` `prompt_file`,
`prompt_file_sha256`, `max_new_tokens_per_turn`, `max_total_sequence_length`; `gen_agentic.SAMPLER` `prompt_file`,
`prompt_file_sha256` (so every cut's `sampler.json` and `gen_meta.json` name and hash it), `gen_meta.json`
`prompt_file`. `gen_agentic --adapter` now refuses a cut whose `sampler.json` differs from the evaluator's SAMPLER in
`prompt_file_sha256`, `max_new_tokens_per_turn`, `max_model_len`, `max_turns` or `stop` (`SAMPLER_MUST_MATCH`); the
p1-a2 cut (old prompt, 512 / 2,048) is therefore not evaluable by this evaluator.

**Budget.** `policy.generation.max_new_tokens=1024`, `policy.max_total_sequence_length=4096` (the rollout's
max_seq_len, grpo.py:3929-3932, and vLLM's `max_model_len` through the yaml interpolation, grpo_math_1B.yaml:540/588),
`cap_max_tokens_to_context` kept; `gen_agentic.SAMPLER` `max_new_tokens_per_turn` 1,024, `max_model_len` 4,096; the
cut's `sampler.json` is that SAMPLER. The launcher takes both numbers from SAMPLER (one definition; test_launcher
checks the pinned yaml with the overrides applied against SAMPLER). Prompt lengths with the one-shot prompt (Qwen3
tokenizer, chat template, thinking off): training rows min 273 / median 312 / p99 820 / max 1,557; items max 856.
The longest leaves 2,539 tokens (two full 1,024-token turns and a 491-token third), the median 3,784 (3.7 full
turns); tool turns stop at `</code>` and are short (probe-tool-2 `text_oneshot`: longest trajectory 1,527 tokens, 45
of 870 turns hit 1,024, no overflow). NeMo RL raises if an input already fills the context (vllm_worker.py:92-102);
that cannot happen here: the rollout ends a trajectory once prompt + reply + observation reach max_seq_len
(rollouts.py:1180-1196) and every prompt is shorter than 4,096. Rung `c` (1,536 tokens, below the longest prompt) is
refused by the launcher until re-planned.

**Memory (Unverified estimate; p1-a3 measures it).** P1 p1-a2 (rung a, 2,048 context) peaked at 10,447 MiB
(the run's memory samples, `p1-a2.mem.tsv`), but its longest sequences were only 682 / 752 / 858 tokens
(`train_data_step1-3.jsonl` padded length), so the 2,048 limit was never approached. Fixed part: the base loaded in
fp32 for plain AdamW (2.03 B x 4 B = 8.1 GB, automodel/setup.py:776-780) plus LoRA weights, gradients and moments
(about 0.6 GB); vLLM's 0.35 share (8.24 GiB requested, 3.21 GiB KV cache in p1-a2) sleeps during training and its KV
cache does not grow with `max_model_len` (4,096 x 112 KiB per token = 0.45 GiB per sequence, about 30 k cached tokens:
generation runs in more waves, not out of memory). The part that scales with the longest sequence of a step is the
logit side: one fp32 vocabulary-sized copy is 4,096 x 151,936 x 4 B = 2.5 GB at 4,096 tokens (0.5 GB at 858), and
the unchunked loss keeps about four of them alive (fp32 cast, log-softmax output and their gradients). Without
chunking a single 4,096-token sequence would add about 8-10 GB: 15-19 GB in all. Rung a therefore now sets
`policy.logprob_chunk_size=512` (as rung b already did): log-softmax and gather per 512 positions, recomputed per
chunk in backward (automodel/train.py:712-735, :939-985), numerically the same values, about 0.3 GB per chunk copy;
expected peak 11-13 GB for typical steps (longest sequence 1.5-2.5 k tokens) and 13-16 GB for a step containing a
4,096-token sequence; micro and logprob batches stay 1. Nothing else changed (no allocator setting, vLLM 0.35). If
p1-a3 fails in the chunked path, rerun it with rung b (same chunk, other optimizer) only after reading the error;
removing the chunk from rung a is the fallback with the higher worst case.

**Stops and end reasons, training vs evaluation.**
- Stop: training sets `policy.generation.stop_strings=["</code>"]` (so vLLM keeps its tokenizer and applies stop
  strings, models/generation/__init__.py:227-237; fix of 0.10) and the datum's `</code>` (merged, vllm_worker.py:
  854-865), `include_stop_str_in_output=True` (:891), `stop_token_ids` null (grpo_math_1B.yaml:488: EOS only);
  the evaluator passes `stop=["</code>"]`, `include_stop_str_in_output=True`, no stop token ids. Both decode the
  generated token ids with `skip_special_tokens=True` (rollouts.py:699 `batch_decode`; `gen_agentic` `tok.decode`)
  and decide on that text with the same `agentic_core.MathCodeCore`, so a token that completes `</code>` with
  trailing characters is treated identically. Per-turn limit in both: min(1,024, 4,096 - input length). Sampling
  differs by design (training temperature 1.0, evaluation greedy).
- End reasons (`MathCodeCore`, shared by `math_code_env` with the HFVerifyWorker reward, `gen_agentic` and the probe's
  text variants without one; the reward is the only difference): `final` = terminal turn with `\boxed`;
  `no_answer` = terminal turn without `\boxed` (EOS prose, cut by the budget, unclosed `<code>`); `max_turns` = a
  tool call at turn 4 (not run); a tool turn is not terminated (`end_reason` null in `env_calls.jsonl`). `overflow`
  is never decided by the core: in training NeMo RL's loop ends the trajectory when the observation is cut and
  consumption.py derives `overflow` from a non-terminated last line; in evaluation `run_trajectories` applies the
  same `>=` rule after the core. A terminal turn keeps its own reason in both even if it also filled the context.
  `tests/test_prompt.EndReasonParity` runs seven scripted trajectories (tool then final, final, no_answer,
  max_turns, overflow, text after `</code>`, unclosed `<code>`) through `env_step` + the rollout double +
  `consumption.build` and through `gen_agentic.run_trajectories` with the same token counts: identical end reasons
  and turn counts.

**Tests (2026-11-15, all pass).** Area-4 venv: 62 tests, 2 skipped (the hydra launcher test and the math-verify
test, each run in its own venv below) - new `tests/test_prompt.py` (6: prompt identity and use, formatting = probed text, worked example
through the executor, prompt lengths, equality with probe-tool-2's rendered prompts and token counts, end-reason
parity); `tests/test_launcher.py` with the area-8 venv: 5 (overrides 4,096 / 1,024 / chunk 512 / prompt file against
the pinned yaml and SAMPLER, stop token ids null, micro and logprob batch 1, rung c refused before any side effect,
cut `sampler.json` names and hashes the prompt and the evaluator refuses a p1-a2-style sampler); area-1 venv
math-verify test: 1. Consumption fixtures of p1-a2 unchanged and still joined.

**p1-a3** (rung a, 3 steps at the new budget; GPU estimate 0.15-0.25 h: p1-a2 took 0.11 h at 32-48 s per step, tool
calls and 1,024-token turns lengthen generation). With a host driver script `run_probe10.sh` (not part of this repository: it runs the commands of 0.9, adds `--out
/out --socket /spool/mathexec.sock`, starts and stops the sidecar and samples GPU memory into `p1-a3.mem.tsv`):

```
run_probe10.sh p1-a3 --profile probe --seed 10100 --base <runtime>/models/qwen3-1.7b --memory a
<host-python> -I <runtime>/consumption.py --env-calls <output-root>/area10-agentic/p1-a3/train/env_calls.jsonl --train-logs <output-root>/area10-agentic/p1-a3/train/nemo/logs --train-file <runtime>/data/train.jsonl --max-steps 3 --run p1-a3 --out <output-root>/area10-agentic/p1-a3/consumption.jsonl
<host-python> -I -c "import json,collections; R=[json.loads(x) for x in open('<output-root>/area10-agentic/p1-a3/consumption.jsonl')]; T=[t for r in R for t in r['per_turn'] if t['tool_status'] is not None]; print('trajectories', len(R), 'tool share', round(sum(any(t['tool_status'] is not None for t in r['per_turn']) or r['end_reason'] == 'max_turns' for r in R) / len(R), 4), 'executed tool turns', len(T), 'of them eos', sum(t['eos'] for t in T), 'end reasons', dict(collections.Counter(r['end_reason'] for r in R)), 'longest sequence', max(r['prompt_tokens'] + r['model_tokens'] + r['env_tokens'] for r in R))"
<host-python> -I -c "import json,glob; A=[c for f in sorted(glob.glob('<output-root>/area10-agentic/p1-a3/train/nemo/logs/exp_*/train_data_step*.jsonl')) for x in open(f) for c in json.loads(x)['content'][0][1::2]]; print('assistant turns', len(A), 'with </code>', sum('</code>' in a for a in A), 'not ending at </code>', sum('</code>' in a and not a.rstrip().endswith('</code>') for a in A), 'own <result>', sum('<result>' in a for a in A))"
```

Pass: rc 0; peak GPU memory (column 2 of the mem.tsv, MiB) <= 22.5 GB (expected 11-16 GB, see above); 3 steps;
`cuts/checkpoint-3/cut.json` with `sampler.json` naming `prompt_math_code_oneshot.txt` and 1,024 / 4,096;
`run_args.json` `prompt_file_sha256` = `sha256:3b7145c1...`; consumption 384 rows. Stop string in training: every
executed tool turn has `eos` false (p1-a2's only tool call had `eos` true), no assistant turn writes its own
`<result>` and (practically) none continues past `</code>`. Tool share in training rollouts: expected near
probe-tool-2's sampled 0.44 at the same temperature (training samples 8 per prompt as the probe did); well below
0.3 would mean training and the probe still differ and needs reading the rollouts before a smoke run. The same two
checks on p1-a2 (consumption written to a scratch directory, CPU, 2026-11-15): 384 trajectories, tool share 0.0026, 1
executed tool turn with `eos` true, longest sequence 858; 385 assistant turns, 9 with `</code>`, 8 not ending at it,
9 with an own `<result>`.

**Smoke** (after p1-a3 passes; 5 steps, cut 5; GPU estimate 0.3 h):

```
run_probe10.sh smoke-1 --profile smoke --seed 10100 --base <runtime>/models/qwen3-1.7b --memory a
```

then consumption with `--max-steps 5 --run smoke-1`, and the two 20-item evaluation arms of 0.9 unchanged
(`gen_agentic.py` now uses the one-shot prompt and 1,024 / 4,096 by itself; the cut arm's `sampler.json` must match).

## Summary

- NeMo RL's multi-turn loop is usable as is: `run_multi_turn_rollout` (`nemo_rl/experience/rollouts.py:1031`) calls
  vLLM once per turn for every active trajectory, hands the last messages to the environment actor, appends the
  environment's observation as **raw tokens without a chat template** (:1172-1179, TODO at :1174) and repeats up to
  `grpo.max_rollout_turns` (default 1, `nemo_rl/algorithms/grpo.py:362`). Only assistant messages that carry
  `generation_logprobs` get `token_loss_mask` 1; environment and prompt tokens get 0 (`grpo.py:2437-2441`).
- NeMo RL's `CodeEnvironment` is **not usable as is** for training: a syntax error in model code raises out of the
  actor (`ast.parse` outside the `try`, `nemo_rl/environments/code_environment.py:116` vs :130-138) and would crash
  the run; there is no timeout or resource limit (an infinite loop hangs the step); `print` output is not returned
  (only the last expression's value, :118-139); rewards are always 0 (:240); with `terminate_on_evaluation=false`
  nothing ever terminates (:236-239). Its "sandbox" (restricted `open` and `__import__`, :161-187) is bypassable
  (`import os.path`, `import io; io.open(...)`) and is not a security boundary. There is no math-with-code recipe in
  NeMo RL (docs/guides/environments.md:88: "We are tracking an end-to-end example ... in #858"); NeMo Gym has one
  (`3rdparty/Gym-workspace/Gym/resources_servers/math_with_code/`, subprocess per session with `SIGALRM` timeout,
  app.py:56-117), but it needs the Gym server stack. **Proposal:** an area-owned environment `MathCodeEnv` (about
  250 lines, Python standard library plus NeMo RL's `HFVerifyWorker` for the reward) with stateless replay execution
  in a subprocess with limits, used identically by the trainer and the evaluator.
- Data recommendation: `EleutherAI/hendrycks_math` (MIT, ungated, 4.9 MB, 7,500 train / 5,000 test), restricted to
  problems whose reference answer is an integer; training export 2,000 rows (4 shards x 500) from `train`, deciding
  item set 500 from `test`. Thinking mode off.
- Memory: full-parameter GRPO of the 2.03 B-parameter model does not fit 22.5 GB on this stack (estimate 25-30 GB;
  Automodel's CPU offload refuses a single GPU, `nemo_rl/models/automodel/setup.py:546-550`). **LoRA** (supported in
  this clone with native refit into vLLM, `nemo_rl/models/generation/vllm/lora_utils.py`) is the rung to probe first;
  it needs a maintainers' decision because the decision names a full model, not a method.
- Estimated area cost 3-4.5 GPU-h (probes, smoke, 60-100-step full run, three evaluation arms). Downloads to decide:
  Qwen3-1.7B (4.08 GB) and hendrycks_math (4.9 MB); optional GSM8K (2.7 MB).

## Pins (proposal; nothing downloaded yet)

| Item | Value |
|---|---|
| Trainer image | area 6/7 derived NeMo RL image `gatedtrain-a6-nemo-rl:1`, `sha256:7a4078e70b28cc6de622fa3cf25c7596c52337fa3f1779488b402c5fa1c74e69` (area 7's full-run launch record `runtime.image`; built from gatedtrain's `containers/nemo-rl-uid1000/`), NeMo RL commit `4f39fd86…`, interpreters `/opt/nemo_rl_venv/bin/python` (trainer, sandbox) and the vLLM worker venv `/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python` (evaluation generation) (`registry.py` `NEMO_RL_PYTHON`, `VLLM_WORKER_PYTHON`, area 7 block). |
| Base model | `Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (https://huggingface.co/api/models/Qwen/Qwen3-1.7B, read 2026-11-14): Apache-2.0, `gated: false`, `Qwen3ForCausalLM`, 2,031,739,904 BF16 parameters (tied embeddings), 12 files, 4,079,450,110 bytes. Download list below. |
| Data | `EleutherAI/hendrycks_math` @ `21a5633873b6a120296cce3e2df9d5550074f4a3` (https://huggingface.co/api/datasets/EleutherAI/hendrycks_math, read 2026-11-14): license MIT, `gated: false`, 7 subject configs, 14 parquet files, 4,890,090 bytes total. |
| Scorer Python | `/opt/nemo_rl_venv/bin/python -I` in the image (as area 7 `score_vqa.py`), standard library only. |

## 1. Multi-turn rollout in NeMo RL

**Dispatch.** `grpo_train` chooses the rollout per step: NeMo Gym path (not used), `run_async_multi_turn_rollout`
when `should_use_async_rollouts` is true, else `run_multi_turn_rollout` (`grpo.py:3905-3936`). For vLLM the async
path is selected exactly when `policy.generation.vllm_cfg.async_engine` is true
(`nemo_rl/models/generation/interfaces.py:287-302`). Validation uses the same two functions (`grpo.py:4994-5013`).
Proposal: `async_engine=false` (synchronous, as area 7 and the sliding-puzzle recipe,
`examples/configs/grpo_sliding_puzzle.yaml:41`) and `data_plane.enabled=false` (area 7's classic `grpo_train`;
`run_vlm_nemo.py:140-144`).

**Synchronous loop** (`rollouts.py:1031-1278`). For `turn in range(max_rollout_turns)` (:1084): flatten the active
trajectories' message logs to token ids (:1096-1102), call `generate_responses` with per-sample `stop_strings`
(:1105-1131). `generate_responses` (:660-742) calls `policy_generation.generate` (vLLM, one batched call per turn),
decodes the new tokens with `skip_special_tokens=True` (:699) and appends an `assistant` message with the vLLM
token ids and `generation_logprobs` (:704-714). Then `calculate_rewards` (:896-1028) groups samples by `task_name`
and calls `env.step.remote(messages, extra_env_info)` once per task (:930-946); messages are reduced to
`role`/`content` (:915-918), so **the environment never sees token ids**. Rewards add up over turns
(`total_rewards[active] += rewards`, :1166). The observation is tokenized raw with `add_special_tokens=False`
(:1172-1179) and appended as a message with the environment's role (:1199-1210). A sample is done when the
environment says `terminated` or the observation had to be cut to fit `max_seq_len` (:1180-1196, :1219-1221);
continuing samples take the environment's `next_stop_strings` and, when not `None`, its metadata (:1225-1243).
After the loop the batch gets `total_reward` and `truncated` (:1248-1249).

**Async variant** (`run_async_multi_turn_rollout`, :1776; per-sample `run_sample_multi_turn_rollout`, :1363-1590):
the same protocol per sample with vLLM's async engine; a sample stops also when vLLM reports the response truncated
(:1454-1457). Not proposed (no benefit at batch 128 on one GPU; area 7 lesson: keep the classic path).

**Loss masking.** `add_grpo_token_loss_masks_and_generation_logprobs` (`grpo.py:2390-2450`): `token_loss_mask` 1
only for `role == "assistant"` messages that carry `generation_logprobs`, 0 for everything else, i.e. prompt and
environment observations (:2437-2441). Every model turn is trained; every tool result is masked.

**Limits.**
- `grpo.max_rollout_turns` (default 1, `grpo.py:362`); each turn is one generate call and one env step.
- Per-turn new tokens: `policy.generation.max_new_tokens`; with `vllm_cfg.cap_max_tokens_to_context: true` each
  request is capped to `max_model_len - input_length` (`nemo_rl/models/generation/vllm/vllm_worker.py:92-102`,
  :1176-1199; config key `nemo_rl/models/generation/vllm/config.py:84`). Proposal: set it, otherwise a late turn
  can ask vLLM for more than the context and the rollout's `tokens_left_for_obs >= 0` assertion
  (`rollouts.py:1186-1190`) is the only guard.
- Total length: `policy.max_total_sequence_length` is the rollout's `max_seq_len` (`grpo.py:3929-3932`); an
  observation that would overflow it is cut and the sample ends (`rollouts.py:1181-1196`).
- A turn that hits `max_new_tokens` without a stop string only sets `sample_truncated` (:1133-1138); the sample is
  **not** ended by the loop. The environment decides (proposal below: no `</code>` at the end means final answer).

**Environment interface.** `EnvironmentReturn` (`nemo_rl/environments/interfaces.py:26-51`): `observations` (list of
`{role, content}`), `metadata` (next per-sample metadata; `None` keeps the previous one, `rollouts.py:1241`),
`next_stop_strings` (per sample, `None` means none), `rewards` (Tensor[B] or dict for multi-reward),
`terminateds` (Tensor[B] bool), `answers` (optional). `EnvironmentInterface.step(message_log_batch, metadata)` and
`global_post_process_and_metrics(batch)` are abstract (:54-90). Environments are Ray actors created by
`create_env` from `ENV_REGISTRY` (`nemo_rl/environments/utils.py:111-137`), with the driver's whole `os.environ`
(:134); `register_env` refuses an existing name (:140-144). Text datasets create the env under its own
`env_name` (`nemo_rl/data/utils.py:198-203`; only VLM is forced to `vlm`, :201).

**Per-trajectory logging available in NeMo RL.** Only batch means: `rollout_metrics` (turns, natural termination
rate, truncation rate, max-turns rate, mean generated and environment tokens, `rollouts.py:1252-1278`); per-sample
counters (:1074-1080) are local and lost. `grpo_train` writes `train_data_step<N>.jsonl` per step with `content`,
`rewards`, `input_lengths`, `token_ids`, `token_loss_mask`, `sample_loss_mask`, `advantages`, logprobs
(`grpo.py:4667-4691`).

**Consumption hook without editing NeMo RL (proposal, as area 7's `LoggedVLMEnvironment`,
`area7-vlm/vlm_env_logged.py:53-79`, MM-27).** The area's own environment actor appends one fsynced line per
(step call, sample) to `env_calls.jsonl` and, when a trajectory ends, one line to `trajectories.jsonl`. The
trajectory identity comes from the turn-0 call: with a single task and validation off, each training step makes
exactly one call whose messages hold exactly one assistant message, and its positions are the batch positions of
the step (`rollouts.py:1086-1094`, the batch is `repeated_batch` in order). The environment stamps
`{traj_step_call, traj_pos}` into the metadata it returns (kept for continuing samples, :1241-1243). Exact token
counts are not visible to the environment (:915-918), so they come from the join: line `traj_pos` of
`train_data_step<k>.jsonl` (k = index of the turn-0 call) gives `input_lengths`, `token_loss_mask` (model tokens =
its sum; per-turn model tokens = runs of 1s after the prompt) and, by difference, environment tokens. The join is
checked by float32 reward and by the SHA-256 of the concatenated assistant text (area 7: "joined in order with
`train_data_step<N>.jsonl` by float32 reward (probe: 640 of 640)", MULTIMODAL-SPEC.md row 7). Trajectories that
end by the loop (max turns, observation overflow) never get a terminal env call; their end reason is derived
(`max_turns` when the env call count equals `max_rollout_turns`, else `overflow`).

## 2. `code_environment`

**What it executes and how** (`nemo_rl/environments/code_environment.py`):
- `CodeEnvironment` (Ray actor, :190-268) spreads the batch over `num_workers` `CodeExecutionWorker` actors started
  with the image's system interpreter (`PY_EXECUTABLES.SYSTEM`, :198-203).
- Per sample the worker takes the **last message's** content, matches `<code>(.*)</code>(.*)` (DOTALL, greedy;
  :109), `ast.parse`s the code (:116), and if the last statement is an expression runs the rest with `exec` and the
  last expression with `eval` (:118-135), in the worker process, with globals = a builtins dict whose `open` and
  `__import__` are replaced (:53-58) and locals = `metadata["context"]` (state kept between turns by pickling the
  dict back through Ray, :147, sanitised of files and modules, :60-78), after `os.chdir(metadata["working_dir"])`
  (:129, :151-159).
- Observation: `{"role": "environment", "content": "<result>…</result>"}` with `pformat` of the value or the
  exception (:80-99, :137-146); empty string when there is no `<code>` block (:110-113).
- `terminated` is true when an expression was evaluated and `terminate_on_evaluation` is set (:134-136, :236-239);
  rewards are always zero (:240); `next_stop_strings` is always `["</code>"]` (:242); `return_extracted_answer`
  must be false (:244-246).

**Gaps that block training use (all verified in source):**
1. `ast.parse` is outside the `try` (:116 vs :130-138): model code with a syntax error raises in the worker,
   `ray.get` re-raises in `step` (:224) and in `calculate_rewards` (`rollouts.py:947`), and the step fails.
2. No timeout, CPU, memory or process limit: `while True: pass` or `10**10**10` blocks the step forever.
3. Only the last expression's value is returned; `print` goes to the Ray worker's log, not to the model.
4. No reward and no termination for a final answer: with `terminate_on_evaluation=false` (needed for several code
   turns, comment :38-41) a sample only ends by `max_rollout_turns` or overflow.
5. Isolation is advisory: `safe_import` blocks exact names only (:171-187), so `import os.path`,
   `import io` (`io.open`), `pathlib`, `ctypes` pass; `safe_open` checks only the builtin `open` (:161-169). Code runs
   with the trainer's uid, file system view and environment variables inside the trainer container (the actor gets
   the driver's `os.environ`, `environments/utils.py:134`).

**Safety inside our container.** `--network none`, read-only area mount, `--cap-drop ALL`, `no-new-privileges`, uid
1000 (area 7's full-run launch) bound what escaped code can do to the host, but inside the trainer it can
read everything the trainer mounts (area 6 review 11 rule: containers that run generated code never mount the area
root, which holds the held-out references; MULTIMODAL-SPEC.md :219) and write `/out` (cuts, logs) and `/tmp`.

**Recipes found.** NeMo RL: unit tests only (`tests/unit/environments/test_code_environment.py:121-224`: `<code>`
tags in the user text, stop string `</code>`, `max_rollout_turns=2`, Llama-3.2-1B, no reward); docs say there is no
end-to-end example (`docs/guides/environments.md:68-88`). NeMo Gym `math_with_code`
(`3rdparty/Gym-workspace/Gym/resources_servers/math_with_code/configs/math_with_code.yaml`: `max_execution_time: 10`,
`max_steps: 3`, dataset `open_math_reasoning_problems_tool`, Apache-2.0; `app.py:56-117` per-session subprocess and
`SIGALRM`; `app.py:213-257` boxed-answer verify with a fallback to answers printed by code). Gym is the later
"NeMo Gym tool calling" candidate of the decision, not this area.

**Proposal: area-owned `MathCodeEnv`** (module `agentic_envs.py`, registered with `register_env("math_code",
"agentic_envs.MathCodeEnv")` and `ACTOR_ENVIRONMENT_REGISTRY[fqn] = PY_EXECUTABLES.SYSTEM`, exactly as
`run_vlm_nemo.py:181-190`; NeMo RL source untouched):
- Turn protocol (the code environment's): the model writes `<code>…</code>`; stop string `</code>`; the env returns
  `<result>\n…\n</result>\n` (stdout of the new cell plus the last expression's `repr`, or the exception type and
  message, cut to 512 characters) and `next_stop_strings=["</code>"]`.
- Final answer: an assistant message that does not end with `</code>` (EOS, `max_new_tokens`, or text) ends the
  trajectory: `terminated=True`, reward = NeMo RL's `HFVerifyWorker` (`math_verify` on `\boxed{}` against the ground
  truth, `math_environment.py:86-168`) applied to the concatenated assistant text, as `MathEnvironment` does
  (`MathEnvironment.step`, `math_environment.py:456`). Reward 1/0, at the terminal turn only.
- Execution: **stateless replay**. Metadata holds the list of previous code cells (JSON-safe, no pickled Python
  objects). Each turn runs, in a fresh subprocess `python -I -S sandbox_exec.py` (the same interpreter in training and
  evaluation), all previous cells with output discarded and then the new cell with output captured; limits:
  wall-clock 10 s, `RLIMIT_CPU` 10 s, `RLIMIT_AS` 2 GiB, `RLIMIT_NPROC`, `RLIMIT_FSIZE` 1 MiB, cwd a fresh temporary
  directory under `/tmp` (tmpfs), empty environment, stdin closed. A syntax error, exception or timeout is an
  observation, never an env failure. A sandbox process that cannot be started is an env failure (raise; training
  stops; in evaluation an MM-4 error).
- `extra_env_info` from the area's processor: `{ground_truth, row_id, env_id}`; per-datum `stop_strings:
  ["</code>"]` (`DatumSpec.stop_strings`, `nemo_rl/data/interfaces.py:74`, collated at
  `nemo_rl/data/collate_fn.py:231-232`).
- Where the sandbox runs is open decision D3 (in-trainer subprocess vs area 6 style sidecar).

Prompt (proposal, user turn, `enable_thinking=false`): "Solve the problem. You may run Python by writing
`<code>…</code>`; the printed output and the value of the last line are returned in `<result>…</result>`. You may
run code at most 3 times. Put the final answer in `\boxed{}`." plus the problem.

## 3. `sliding_puzzle` (second environment, CPU tests)

`examples/run_grpo_sliding_puzzle.py` builds its own data and env and calls `grpo_train` directly: each datum is a
generated puzzle (`generate_puzzle_datum`, :59-123: random size 2..`size` and odd shuffle count, rendered board and
rules in one user message through the chat template, `extra_env_info = SlidingPuzzleMetadata(game_state, num_moves,
max_moves)`, `stop_strings ["</action>"]`, `task_name`), served by an infinite `IterablePuzzleDataset` (:126-154);
the env actor is created directly with `SlidingPuzzleEnv.options(num_gpus=0).remote(cfg)` (:170), not through
`ENV_REGISTRY` (it is listed there too, `environments/utils.py`, and in `distributed/actor_environments.py:89`).
Config `examples/configs/grpo_sliding_puzzle.yaml`: `max_rollout_turns: 50` (:7), `max_total_sequence_length: 1024`
(:22), `size: 5`, `shuffle_moves: 10`, `max_moves: 30` (:53-58).

`SlidingPuzzleEnv.step` (`nemo_rl/environments/games/sliding_puzzle.py:345-392`) runs `SlidingPuzzleRunner.process_turn`
per sample (:265-342): parses the last `<action>…</action>` (:245-263), applies it, returns
`<environment>…</environment>\n`, reward 1.0 only when solved (:192-198), `terminated` on solve or `max_moves`
(:288-300). Quirks to keep in the adapter tests: a parse failure returns `metadata None` without terminating (the
rollout then keeps the previous metadata, `rollouts.py:1241`); at `max_moves` the observation is an `<error>` and
`next_stop_strings` is `None`; generation uses the global `random` without a seed
(`run_grpo_sliding_puzzle.py:71-76`, `sliding_puzzle.py:71-73`), so item sets must be generated with a seeded RNG by
the area (row id = SHA-256 of `{size, shuffle_moves, seed}`).

Use in CPU tests (proposal): `SlidingPuzzleGameLogic` and `SlidingPuzzleRunner` are pure Python; the adapter tests
need no NeMo RL at all: fixtures of `env_calls.jsonl`, `trajectories.jsonl` and `train_data_step<N>.jsonl` for both
envs (math_code: terminal reward in {0, 1} after at most 4 turns; sliding_puzzle: reward 1.0 at the solving turn,
`terminated` by `max_moves`, `metadata None` turns) must go through the same consumption code. An optional
container smoke (NeMo RL image, CPU, `ray.init(num_cpus=4)`, a scripted "policy" that returns canned assistant
messages) can drive both env actors through `calculate_rewards` (`rollouts.py:896`) without vLLM (**unverified**
that the image's Ray starts without a GPU under `--gpus` absent; area 6/7 always ran with a GPU).

## 4. Data

HF API, `?blobs=true`, read 2026-11-14 (`env -u HF_TOKEN`). Sizes are the sum of all repo files.

| Repo @ revision | License (card) | Gated | Size | Splits | Notes |
|---|---|---|---|---|---|
| `EleutherAI/hendrycks_math` @ `21a5633873b6a120296cce3e2df9d5550074f4a3` | mit | no | 4,890,090 B (14 parquet + README) | train 7,500 / test 5,000 over 7 subjects (README `num_examples`, e.g. algebra 1,744/1,187) | Hendrycks MATH with `level`, `type`, `solution`; answer = last `\boxed{}` of `solution`. **Recommended.** |
| `openai/gsm8k` @ `740312add88f781978c0658806c59bc2815b9866` | mit | no | 5,900,352 B (main 2.73 MB) | train 7,473 / test 1,319 | Integer answers after `####`; ceiling risk (below). Alternative. |
| `HuggingFaceH4/MATH-500` @ `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` | none declared | no | 450,344 B | test 500 | Subset of MATH test (PRM800K split); no license in the card. Not needed: the recommended item set is drawn from hendrycks_math test. |
| `BytedTsinghua-SIA/DAPO-Math-17k` @ `65877096c24ffa7abc4e4fa5edb95cf3413a5674` | apache-2.0 | no | 299,366,461 B (one parquet) | train only | Integer answers, competition level; no held-out split; base reward likely near 0 for a 1.7 B non-thinking model (**unverified**). |
| `nvidia/OpenMathReasoning` @ `d3d08664755704f422af97d43a7ff0ded4bd95df` | cc-by-4.0 | no | 49,525,388,931 B (234 files) | cot/tir/genselect | Far too large; TIR subset could serve SFT later. |
| `project-numina/NuminaMath-TIR` @ `77a91d7b7a1a98ac4b1beb7d86c09d156b935dcd` | apache-2.0 | no | 147.6 MB | train / test | Tool-integrated solutions (SFT material); answers not all integers. |
| `SynthLabsAI/Big-Math-RL-Verified` @ `c75d2f117cddfecb6bd08756e61e508e59732b21` | apache-2.0 | **auto** | 33.2 MB | train | Gated: excluded. |
| `nlile/hendrycks-MATH-benchmark` @ `465bcdb36f5962aa3512891498966df785fc3c18` | none declared | no | 5.34 MB | train 12,000 / test 500 | Re-split with `answer` column; no license. Excluded. |

**Recommendation.** hendrycks_math, all 7 subjects, levels 1-5, **integer reference answers only** (after stripping
`$`, `\!`, thousands separators, a trailing `.0`; the filter keeps exact match trivially well defined for the
standard-library scorer). Training export: 2,000 rows (4 shards x 500, seeded draw from `train`, stratified by
subject), read in file order (`data.shuffle=false`); 60 steps x 16 prompts consume 960 rows (**more steps** would
use up to 2,000). Deciding item set: 500 from `test` (integer answers, stratified by subject and level, seeded),
disjoint by exact problem text from all training rows (MM-11; check at `make_data.py`). The integer share of MATH
answers is **unverified** (estimate 55-65 %; about 4,000 train and 3,000 test candidates): `make_data.py` reports
it, and if fewer than 2,500 / 600 qualify the fallback is "integer or simple fraction".

**MM-38 (binary metric, degenerate only at mean 0 or 1).** Published Qwen3-1.7B numbers (Qwen3 technical report,
arXiv:2505.09388; recalled, **unverified**, not re-read here): MATH-500 about 73 % in non-thinking mode and about
93 % in thinking mode. With the code tool, a prompt the model has not been tuned for and greedy decoding, the base
exact match on the integer subset is estimated at 0.45-0.70 (**unverified**): well inside (0, 1), with room to
improve. GSM8K for the same model is likely 0.85-0.92 non-thinking (**unverified**), which leaves little headroom
for a paired gate (MDE about 0.05 at N 500), hence MATH. The smoke's 20-item base arm and, optionally, a 100-item
base pilot settle it before registration.

**Thinking off** (`policy.tokenizer.chat_template_kwargs: {enable_thinking: false}`, applied as a partial of
`apply_chat_template`, `nemo_rl/algorithms/utils.py:465-496`; key present in `examples/configs/grpo_math_1B.yaml:158`).
Reasons: (1) length: thinking traces on MATH run to thousands of tokens (the model card recommends 32,768 output
tokens, q17 README "Best Practices" item 2), while the budget here is 2,048 tokens over up to 4 turns and 128
trajectories per step on 24 GB; (2) the model card forbids greedy decoding in thinking mode (README "enable_thinking=True"
note) and the evaluator is greedy; (3) with the flag false the template emits an empty `<think>\n\n</think>\n\n`
after the generation prompt (tokenizer_config.json `chat_template`, last lines), so turns are plain text.

## 5. Model

`Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (HF API, model card, `config.json`,
`tokenizer_config.json`, `generation_config.json`, read 2026-11-14):
- License Apache-2.0 (card `license: apache-2.0`), not gated.
- Files: `model-00001-of-00002.safetensors` 3,441,185,608 B, `model-00002-of-00002.safetensors` 622,329,984 B,
  `model.safetensors.index.json` 25,605 B, `tokenizer.json` 11,422,654 B, `vocab.json` 2,776,833 B, `merges.txt`
  1,671,853 B, `tokenizer_config.json` 9,732 B, `config.json` 726 B, `generation_config.json` 239 B, `LICENSE`,
  `README.md`, `.gitattributes`; total 4,079,450,110 B.
- Architecture: `Qwen3ForCausalLM`, 28 layers, hidden 2,048, 16 query / 8 KV heads, head dim 128, intermediate
  6,144, vocab 151,936, `tie_word_embeddings: true`, `max_position_embeddings` 40,960, bf16. KV cache per token
  28 x 2 x 8 x 128 x 2 B = 112 KiB (derived).
- Generation defaults (thinking mode): temperature 0.6, top_p 0.95, top_k 20; EOS `[151645 <|im_end|>, 151643]`,
  pad 151643. Card: non-thinking 0.7 / 0.8 / 20.
- Chat template: ChatML; tools in the system turn as `<tools>` JSON, calls as `<tool_call>{"name","arguments"}</tool_call>`,
  results as a user turn `<tool_response>…</tool_response>`. `<tool_call>`, `</tool_call>`, `<tool_response>`,
  `</tool_response>`, `<think>`, `</think>` are added tokens with `special: false` (151657, 151658, 151665, 151666,
  151667, 151668), so they survive NeMo RL's `skip_special_tokens=True` decode; `<|im_start|>`/`<|im_end|>` are
  special. Switches: `enable_thinking` argument; soft `/think`, `/no_think` only when thinking is enabled.
- Tool format choice (proposal): the code environment's `<code>…</code>` / `<result>…</result>` text protocol inside
  one assistant turn, not Qwen's `<tool_call>` JSON. NeMo RL appends observations raw without a template
  (`rollouts.py:1174`), so a native `<tool_response>` user turn would need the env to emit the template text
  (`<|im_end|>\n<|im_start|>user\n<tool_response>…`) itself; possible later, but it couples the env to the model's
  template. Open decision D6.

## 6. Memory and time on 24 GB (all estimates unless cited)

**Basis.** P = 2.03 B parameters. Automodel loads the trainable model in fp32 unless the optimizer keeps fp32 master
weights (`nemo_rl/models/automodel/setup.py:766-780`); FSDP2 mixed precision bf16 compute, fp32 reduce (:531-536).
Area 7 (0.87 B, TE FusedAdam with fp32 master via bf16 remainders and bf16 moments, micro batch 1, 1,024 tokens,
vLLM 0.2) peaked at 17.7-19.7 GB (area 7 README "Implementation (as run)"), i.e. about 9-11 GB beyond the ~8.7 GB
of model and optimizer state (248k vocabulary there, 152k here).

| Option | Model + optimizer state on GPU | Verdict |
|---|---|---|
| Full FT, default AdamW (fp32 params, grads, two moments: 16 B/P) | 32.5 GB | does not fit |
| Full FT, TE FusedAdam `master_weights`, `store_param_remainders`, bf16 moments (area 7 rung a: about 10 B/P) | 20.3 GB, plus activations, fp32 logits (2,048 x 151,936 x 4 B = 1.24 GB per sequence) and vLLM residue: 25-30 GB | does not fit |
| Full FT with Automodel `cpu_offload` | n/a | refused on one GPU: `NotImplementedError("CPUOffload doesn't work on single GPU for AutoModel")` (`setup.py:546-550`). Area 7's rung c could never have run. |
| 8-bit Adam (`torchao` 0.15/0.17 in `uv.lock:8536-8550`; presence in the image venv **unverified**) | params load fp32 because no fp32 master (`setup.py:776-780`): about 10 B/P = 20 GB | no gain |
| Megatron backend with `optimizer_cpu_offload` (`examples/configs/grpo_math_1B.yaml:350-351`, `nemo_rl/models/megatron/setup.py:170-186`) | bf16 params + fp32 main grads on GPU about 12 GB; fp32 master and moments about 24 GB in host RAM | too tight for a host with about 32 GB RAM (area 6 peaked at 29.1 GiB on such a host); Megatron path never run here |
| **LoRA** (Automodel `lora_cfg`, `examples/configs/grpo_math_1B.yaml:200-211`; native refit into vLLM, `nemo_rl/models/generation/vllm/lora_utils.py:21-160`, `lora_refit_mode: native` default :524; Qwen3 dense recipe `examples/configs/recipes/llm/grpo-qwen3-8B-base-1n8g-automodel-lora.yaml`) | base 4.1 GB (bf16, with a fp32-master optimizer) or 8.1 GB (fp32, plain AdamW); adapters rank 32 all-linear about 35 M params (1.25 M per layer x 28), with Adam about 0.6 GB | fits (training phase about 10-15 GB) |

**Memory ladder (proposal; probe in this order, the first whose smoke peaks at <= 22.5 GB is used):**
- **a (probe first):** LoRA rank 32, alpha 64, `match_all_linear`, plain AdamW lr 2e-5 (fp32 base 8.1 GB),
  activation checkpointing, `max_total_sequence_length` 2,048, `max_new_tokens` 512 per turn, `max_rollout_turns`
  4, `cap_max_tokens_to_context` true, micro batch 1 and `logprob_batch_size` 1, KL 0 with
  `skip_reference_policy_logprobs_calculation` and `data_plane.enabled=false` (area 7 lessons), vLLM
  `gpu_memory_utilization` 0.35, Automodel backend `linear=torch`, `attn=sdpa` only if the refit fails as in area 7
  (Qwen3 dense has a native vLLM path; **unverified** whether TE `_extra_state` appears for Qwen3). Estimated peak
  15-19 GB.
- **b:** a with TE FusedAdam `master_weights=true` (bf16 base, about -4 GB; LoRA plus TE FusedAdam **unverified**)
  and `policy.logprob_chunk_size` 512, vLLM 0.4 (more KV, faster rollouts).
- **c:** full fine-tune, area 7 rung a settings, `max_total_sequence_length` 1,536, vLLM 0.15: expected to exceed
  22.5 GB; probe only if full weights are required. Fallback outside the decision: `Qwen/Qwen3-0.6B`
  (@ `c1899de289a04d12100db370d81485cdf75e47ca`, Apache-2.0, 751,632,384 params, 1.52 GB) full fine-tune with area 7's
  rung a (about 7.5 GB state).

Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (breaks the CUDA IPC refit under `--cap-drop ALL`,
area 7 probe-b4).

**Throughput (estimate).** Step = 16 prompts x 8 generations = 128 trajectories, mean about 900-1,400 tokens
(prompt ~150, 1-3 code turns, tool results). vLLM decode of a 1.7 B model on an RTX 4090 with ~3-5 GB KV (25-45k
tokens, 20-40 concurrent sequences) about 2-4k tokens/s: 40-70 s generation per step, plus sandbox replay (<5 s
with 8 parallel subprocesses), plus logprob and training passes at micro batch 1 (about 30-50 s), plus refit: about
90-130 s per step. Startup about 5 min (area 7).

## 7. Evaluator (proposal)

Own two-step evaluator, as area 7 (`gen_vqa.py` + `score_vqa.py`), with the env's sandbox in the generation step
(multi-turn generation cannot be separated from execution):
1. `gen_agentic.py` (GPU, vLLM worker venv, vLLM offline `LLM.generate` with `prompt_token_ids`): per item the same
   chat template and prompt as training, `enable_thinking=false`, greedy (temperature 0), K 1, seed recorded,
   per-turn `max_tokens` 512, stop `["</code>"]` with `include_stop_str_in_output=True` (as NeMo RL,
   `vllm_worker.py:891`), the same `max_rollout_turns` 4 and 2,048-token budget; after each turn the **same**
   `sandbox_exec.py` replay (same interpreter, same limits) and the observation appended as raw tokens
   (`add_special_tokens=False`), exactly as `rollouts.py:1172-1210`, so evaluation matches the training loop token
   for token. All items advance turn by turn in one batch. Writes `gen/<item>__s<seed>.json` (turns, code cells,
   observations, sandbox status per cell, token counts per turn) and `gen_meta.json` (items hash, sampler, model
   files, sandbox code hash and limits, framework).
2. `score_agentic.py` (CPU, `python -I`, standard library): its **own** answer parser, independent of
   `math_verify`: last `\boxed{…}` with balanced braces in the final assistant text, normalise (`$`, spaces, `\!`,
   `,` thousands separators, `\text{}`, trailing `.0`, leading `+`), parse as integer, compare with the reference
   integer. Score 1/0.
   - `sub_metrics` (finite numbers): `turns_used`, `tool_calls`, `tool_errors` (exception or timeout observations),
     `tool_timeouts`, `answer_found`, `truncated` (budget exhausted), `model_tokens`, `env_tokens`.
   - MM-4 score outcomes (policy's fault, score 0): `wrong_answer`, `no_answer`, `max_turns`, `overflow`; code that
     raises or times out is part of the transcript, not an error.
   - MM-4 error classes (evaluator's fault, unpaired): `missing_output` (no generation for a registered item),
     `sandbox_failure` (the sandbox process could not start or its protocol broke: recorded by `gen_agentic.py` per
     cell), `scorer_error`.

**Tier and disclosures.** The reward uses `math_verify` (HF Math-Verify via `HFVerifyWorker`,
`math_environment.py:86-168`); the scorer is another answer parser, so under R15-1 (05 §6.1:255) the metric is **T3,
construct-aligned** (`construct_alignment`: both reward final-answer correctness). The references are held-out MATH
test answers written by the same annotation process as the training answers; under R16-1 (05 §6.1:257) the scorer
stays T3 and the registration states the convention (`shared_annotation`: `\boxed{}` integer answers of the MATH
competition corpus). Contamination note for the report: MATH train and test are public since 2021 and the base was
very likely trained on them (**unverified**); the paired design still measures the change, not absolute skill.

**Power (estimate).** Binary paired difference with discordance d 0.2-0.3: sd 0.45-0.55; N 500, one-sided alpha
0.025, power 0.8: MDE about 0.055-0.07. Expected gain from 60 LoRA steps on a model that already answers about
60 %: **unverified**; tool-format learning and answer formatting could give 0.03-0.10. Mitigations: 100 steps instead
of 60 (still cheap with LoRA), or held-out items restricted to levels 3-5 (more headroom).

## 8. gatedtrain adapter `nemo_rl_agentic` (proposal; `gatedtrain/ext/` untouched)

Sibling of `nemo_rl_vlm` (`registry.py:604-642`) and `nemo_rl_ext` (`registry.py:471-520`):
1. **Pins.** NeMo RL commit `4f39fd86…` (env-freeze.txt line, as area 7); image ID; base `{repo, revision}`
   (`base_ref.json`, MM-21); `agentic_envs.py`, `sandbox_exec.py`, processor and prompt file SHA-256s in
   `run_args.json`; the sandbox interpreter path and limits.
2. **Stage** `rl.nemo_rl_agentic@0.1`, single stage from the plain base directory. Cuts: LoRA adapter
   (`adapter_model.safetensors`, `adapter_config.json`, composed with the base hash as the LoRA areas 2a/2b/3,
   MM-18) or full weights if rung c is chosen; `cut.json` last; NeMo RL `step_N` are resume state. LoRA cut file
   layout from Automodel **unverified** (lora_cfg `restore_from` comment, `grpo_math_1B.yaml:211`, names
   `adapter_model.safetensors + adapter_config.json`).
3. **Environment parameterisation.** Registry data keyed by env id:

   | env id | NeMo RL env | evaluator id | item schema (payload) | stop / turns | reward |
   |---|---|---|---|---|---|
   | `math_code` | `agentic_envs.MathCodeEnv` | `agentic.math_code_items@0.1` (deciding, binary, T3, both disclosures) | `text` (problem), `answer` (integer string), `subject`, `level` | `</code>` / 4 | terminal 0/1 |
   | `sliding_puzzle` | `agentic_envs.LoggedSlidingPuzzleEnv` (subclass of `SlidingPuzzleEnv`) | `agentic.sliding_puzzle_items@0.1` (CPU tests only, never registered) | `text` (rendered prompt), `size`, `shuffle_moves`, `seed` | `</action>` / 30-50 | 1.0 at solve |

   The launch request names `--env <id>`; `run_args.json` records it; `ext-collect` refuses a spec whose evaluator
   is not the env's, a run whose `run_args.env_id` differs from the registration's, and an item set whose payload
   fields differ from the env's schema. A gate binds one env; another env is a new registration (decision).
4. **Consumption row schema** (precision `row`, `derivation: observed`, one record per trajectory):
   `{trajectory_id = H(run, attempt, step, traj_pos), row_id, data_ref.row = sha256(row_id), step, traj_pos,
   member (generation index within the prompt group), env_id, turns, per_turn: [{model_tokens, env_tokens,
   tool_status}], model_tokens, env_tokens, prompt_tokens, reward, end_reason ∈ {final, solved, max_moves, max_turns,
   overflow}, response_sha256}`. Built from `trajectories.jsonl` + `env_calls.jsonl` (env actor) joined with
   `train_data_step<N>.jsonl` (`token_loss_mask`, `input_lengths`, `rewards`). Checks: 128 trajectories per step,
   16 rows of the step in file order each 8 times consecutively (area 7 rule), float32 reward equality, assistant
   text hash equality, model + env + prompt tokens = `input_lengths`, no step beyond `max_steps`. Outcome
   `consumed` for every trajectory (NeMo RL trains zero-advantage groups with zero advantage; MM-28 `exposed: not
   recorded` otherwise). Failed join refuses; `iteration` fallback is an explicit operator decision.
5. **Launch facts** (mirrors area 7's full-run launch): `--runtime=nvidia --gpus all --network
   none --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges --shm-size=16g --ulimit memlock=-1 --ulimit
   nofile=65535:65535 -w /opt/nemo-rl --tmpfs /tmp:rw,exec,nosuid,nodev,size=8g`; env allow-list exactly
   `FLASHINFER_CUDA_ARCH_LIST HF_HUB_OFFLINE HOME PYTHONDONTWRITEBYTECODE PYTHONUNBUFFERED
   RAY_memory_monitor_refresh_ms TORCH_CUDA_ARCH_LIST` (no `HF_TOKEN`, no `PYTORCH_CUDA_ALLOC_CONF`). Mounts:
   training: scripts, model and `data/` read-only, `/out` rw; **not** the item sets if the sandbox runs in the
   trainer (D3 option b), or an area 6 style sidecar with a spool (D3 option a); generation: as training plus
   `/inputs/cut` ro and the item set ro; scoring: CPU, item set ro, `/inputs/gen` ro, `/out` rw.
6. **Env-swappability checks** (tests): both env fixtures pass the same consumption code; per-env registry rows
   complete (metric, direction, range, error and score classes, tier); `--env` changes nothing in the adapter but
   the row lookup; an arm of env A refused under a registration of env B; a training log whose env actor class is
   not the env id's refused.

## 9. Open decisions, risks and probe plan

**Maintainer decisions (options; recommendation first):**
- **D1 Training method.** (a) LoRA rank 32 on Qwen3-1.7B (recommended: the only option that fits 22.5 GB with a
  margin; supported with native vLLM refit at this commit); (b) full fine-tune of Qwen3-1.7B, probe rung c (expected
  OOM, about 0.3 GPU-h to confirm); (c) full fine-tune of Qwen3-0.6B (fits like area 7, but not the decided model).
- **D2 Dataset.** (a) hendrycks_math integer-answer subset, 2,000 train rows, 500 test items (recommended: MIT, tiny,
  headroom); (b) GSM8K (MIT, integer answers; ceiling risk); (c) DAPO-Math-17k train + MATH test items (hard, low
  base reward).
- **D3 Where generated code runs.** (a) area 6 style sidecar executor container (MM-34a: network none, read-only,
  limits, spool socket; never mounts the area root), used by the trainer's env actor and by `gen_agentic.py`
  (recommended for consistency with the area 6 review 11 rule; more adapter work); (b) subprocess with rlimits
  inside the trainer and generation containers, with the item sets and references kept out of those containers'
  mounts (cheaper; generated code can still write `/out` and `/tmp`); (c) NeMo RL `CodeEnvironment` as is (rejected:
  crashes on syntax errors, no timeout).
- **D4 Thinking mode.** (a) off (recommended, section 4); (b) on with a 4,096-token budget (greedy evaluation
  conflicts with the model card; memory and time about 2x).
- **D5 Gate size.** (a) N 500, K 1 greedy, 60 steps (area 7 practice); (b) N 500, K 1, 100 steps (recommended if
  the smoke shows base accuracy above 0.6: more chance to exceed the MDE, +~1 GPU-h); (c) K 4 sampled.
- **D6 Tool protocol.** (a) `<code>`/`<result>` text protocol (recommended: NeMo RL's code-env convention, no
  template coupling); (b) Qwen native `<tool_call>`/`<tool_response>` turns rendered by the env.

**[routine]** memory rung order; prompt wording; sandbox limits (10 s, 2 GiB); `max_rollout_turns` 4; lr.

**Risks.** Weak effect (base already ~0.6, short LoRA run): mitigated by D5(b). The model rarely calls the tool
(possible; the gate measures the answer, not tool use; `tool_calls` sub-metric shows it). Host RAM (Ray plus
sandbox subprocesses; area 6 peaked at 29.1 GiB on a 32 GB host): LoRA keeps it lower than full FT. vLLM KV pressure with 128
multi-turn sequences: slower steps, not failure. Join brittleness (env call order vs batch order): verified at the
smoke on all 640 trajectories.

**Probe plan (GPU-h estimates, unverified).**

| Phase | What | GPU-h |
|---|---|---|
| P0 (CPU) | `make_data.py` (integer filter, shards, items, overlap 0); `agentic_envs.py` and `sandbox_exec.py` unit tests (syntax error, timeout, memory limit, print capture, replay determinism, final-answer termination, reward); sliding-puzzle fixtures; adapter tests on fixture logs; tokenizer-only check of prompt lengths | 0 |
| P1 (GPU) | memory ladder: rung a smoke-probe 3 steps (then b, then c only if D1(b)) | 0.15-0.45 |
| Smoke | 5 steps (cut 5) + 20-item base arm + 20-item cut arm; `smoke.json` (MM-38) | 0.3 |
| Full | 60 steps (D5 a) or 100 steps (D5 b) | 1.8-3.3 |
| Evaluation | base and candidate on 500 items (greedy, multi-turn, about 10 min each incl. start) | 0.3-0.4 |
| Total | | about 2.6-4.5 |

## Download list (proposal; nothing downloaded)

| Artifact | Revision | Files | Bytes |
|---|---|---|---|
| `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | all 12 files | 4,079,450,110 |
| `EleutherAI/hendrycks_math` | `21a5633873b6a120296cce3e2df9d5550074f4a3` | 14 parquet + README | 4,890,090 |
| optional `openai/gsm8k` (D2 b) | `740312add88f781978c0658806c59bc2815b9866` | `main/*.parquet` + README | 2,733,568 |
| optional `Qwen/Qwen3-0.6B` (D1 c) | `c1899de289a04d12100db370d81485cdf75e47ca` | all 10 files | 1,519,209,243 |

## Result (2026-11-16)

- **full-1** (100 steps x 16 prompts x 8, LoRA r32, cuts every 20; 2.14 GPU-h; training reward 0.61 -> 0.86).
- **Gate PASS** (`gate/registration.yaml`, 500 items, greedy, seed 11, candidate checkpoint-100): exact_match 0.648 ->
  0.702 (delta +0.054, 95% cluster-bootstrap CI [0.014, 0.096], p 0.0039). Used-tool share 0.412 -> 0.884,
  answer-found 0.836 -> 0.914 (sub-metrics, not gated). Run `run-20261116T022020Z-96715e1a`;
  consumption precision row (12,800 trajectories); impact maps cuts 20..100 to steps 20..100.
- One training trajectory ended at `max_turns` with reward 1.0 (a boxed answer in the same turn as a turn-4 tool call;
  `MathCodeCore.step` verifies any terminal turn with `\boxed`). The evaluator scores `max_turns` 0. The adapter's
  end-reward table now allows it (the maintainers' decision "Area 10 result", not published).
