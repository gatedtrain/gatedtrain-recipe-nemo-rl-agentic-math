# gatedtrain recipe: NeMo RL agentic GRPO on MATH with a Python tool

A [gatedtrain](https://github.com/gatedtrain/gatedtrain) recipe: Qwen3-1.7B learns, with NeMo RL's synchronous
multi-turn GRPO and a LoRA adapter (r 32, alpha 64), to solve MATH problems with integer answers while calling a Python
tool (`<code>...</code>`, answered by `<result>...</result>`, up to four turns). gatedtrain gates the trained cut
against the base model on 500 held-out MATH test problems; the deciding metric is `exact_match` of the final `\boxed{}` answer, read by
the recipe's own integer parser (the training reward is NeMo RL's `math_verify`). The training environment is
configuration: the adapter `nemo_rl_agentic` knows each environment's log shape and gates one environment through its
evaluator (`math_code` now). Every model-written program runs only in the executor sidecar (`--network none`, uid 1000,
no capabilities, read-only root, CPU / memory / pids limits), paired with the trainer and with the evaluation's
generation step through a Unix socket in a shared spool. Prototype.

The runtime notes, code comments, tests and `RESULT.md` cite the maintainers' decision record ("Area 10 (agentic RL)
direction", "Area 10 decisions", "Area 10 tool-use probe and choice", "Area 10 result") and the framework's review
rounds ("Review 9 M1", "area 6 review 11"): those records are not published. The ids `D3` to `D6` are the open
decisions of `runtime/README.md` section 9.

## Layout

| Path | What |
|---|---|
| `src/gatedtrain_recipe_nemo_rl_agentic_math/` | the gatedtrain adapter `nemo_rl_agentic` (`adapter.py`) and its registry rows (`rows.py`), registered through the `gatedtrain.adapters` entry point |
| `runtime/` | what runs in the containers: `run_agentic_nemo.py` (NeMo RL launcher), `ptc_agentic_hooks.py` (cuts, run record), `math_code_env.py` and `agentic_core.py` (the environment and its turn logic, a client of the executor), `traj_log.py` (the per-trajectory log), `consumption.py` (the log join), `gen_agentic.py` (multi-turn vLLM generation), `score_agentic.py` (scoring), the prompts, the host-side `fetch_models.py`, `fetch_data.py`, `make_data.py` and `probe_tool_use.py`, the pins (`models/MODELS.json`, `raw/RAW.json`), the data metadata, the registered item set (`items-test/`) and the runtime's own CPU tests (`tests/`) |
| `runtime/executor/` | the executor: `executor_server.py` (serve, ping), `sandbox_runner.py` (one cell per process under rlimits), `exec_protocol.py` (wire protocol, statuses) and `run-executor.sh` (prints the sidecar's `docker run` commands) |
| `tests/` | the adapter's tests on fixture runs (no GPU, no model, no NeMo RL needed) and gatedtrain's conformance check |
| `results/2026-11-16/` | the gate files of the published run (`gate/`; redacted copies, host paths replaced by `<data-root>`, see `RESULT.md`) |

## Setup

```bash
pip install "gatedtrain @ git+https://github.com/gatedtrain/gatedtrain@v0.3.0"   # the framework, at the release this recipe was tested with
pip install -e .                      # the adapter, into the environment that runs gatedtrain
cd runtime
python fetch_models.py                # the base model (needs huggingface-hub; network)
python fetch_data.py                  # the hendrycks_math snapshot (needs huggingface-hub; network)
python -I make_data.py                # needs pyarrow: training shards (data/) and items-test/ from raw/
```

`make_data.py --tokenizer models/qwen3-1.7b` (also needs transformers) recomputes the prompt token counts
(`prompt_tokens` in `data/data_meta.json`: the deployment prompt under the base model's chat template, thinking off).
Without `--tokenizer`, the script keeps the committed counts when everything else in `data/data_meta.json` comes out
byte for byte the same (it prints `data_meta.json reproduced; kept its prompt_tokens`), since the counts depend only
on the rows, the prompt file and the pinned tokenizer; if anything else differs, the new `data_meta.json` has no
`prompt_tokens`. Either way a faithful run leaves the committed files unchanged (`git status runtime/`).

The trainer, the generation and scoring steps and the executor sidecar all use the NeMo RL family image
`gatedtrain-a6-nemo-rl:1`, pinned in `gatedtrain.families.nemo_rl` and built from gatedtrain's
`containers/nemo-rl-uid1000/`; the sidecar runs the image's own interpreter on `runtime/executor/`.

### Images

The image is pinned by its local image ID (`NEMO_RL_UID1000_IMAGE` = `sha256:7a4078e7...` in
`gatedtrain.families.nemo_rl`; `runtime/executor/run-executor.sh` prints the same ID), not by a registry digest: build it
yourself in two steps.

1. The NeMo RL image `nemo-rl:4f39fd8`: NeMo RL's own `docker/Dockerfile` at commit
   `4f39fd86289a2ffd5aaf5d202702f7fd7185c924` (target `release`), as described in gatedtrain's `docs/tier-b-runbook.md`
   (`NEMO_RL_IMAGE` in `gatedtrain.families.nemo_rl` is the ID of the build the published run used).
2. `gatedtrain-a6-nemo-rl:1`: from gatedtrain's `containers/nemo-rl-uid1000/` (`docker build -t
   gatedtrain-a6-nemo-rl:1 .` there); one layer on top of `nemo-rl:4f39fd8` that hands the vLLM files NeMo RL patches
   at start-up to uid 1000, because gatedtrain refuses a trainer running as root.

Check the resulting ID with `docker image inspect gatedtrain-a6-nemo-rl:1 --format '{{.Id}}'`. A rebuild can produce
another ID; gatedtrain's launcher refuses an image whose ID is not the pinned one, so change the constants
deliberately (and `GATEDTRAIN_MATHEXEC_IMAGE` for `run-executor.sh`) if you use your own build. The commands in this
repository use the Docker context `gatedtrain`, as gatedtrain's launcher prints them; the executor sidecar's container
is named `gatedtrain-mathexec-<hashed label>`.

Trust: the evaluator loads LoRA cuts (safetensors) and NeMo RL can resume from its own training checkpoints; load only
cuts and checkpoints you produced.

The adapter finds the runtime at `runtime/` of this checkout (or `GATEDTRAIN_RECIPE_NEMO_RL_AGENTIC_MATH_RUNTIME`). The
runtime is mounted read-only at its own host path, so that path is also the one the containers see; the sidecar mounts
only `runtime/executor/` (never the runtime root, which holds the items' answers). The runs' outputs go under the host's
gatedtrain output root (`<output_root>/nemo-rl-agentic-math`); a run's spool is a sibling of its trainer output
(`<run>/train`, `<run>/spool`). `runtime/README.md` has the research notes, the trainer settings, the probes, the
executor sandbox and the decisions behind the metric.

The adapter locates `runtime/` from its own source tree (`src/gatedtrain_recipe_nemo_rl_agentic_math/../../runtime`),
which exists only in a source checkout or an editable install (`pip install -e .`). After a non-editable install (a
wheel), set `GATEDTRAIN_RECIPE_NEMO_RL_AGENTIC_MATH_RUNTIME` to the `runtime/` directory of a checkout.

Some runtime names predate the project's rename to gatedtrain and are kept because the published run and the
committed metadata carry them: the hooks module `runtime/ptc_agentic_hooks.py` (each run records the SHA-256 of its
code files by these names in `run_args.json`, and the adapter checks them beside the launched script), the
trajectory stamp key `_ptc` that `traj_log.py` writes into the environment metadata (`traj_log.py` is part of the
recorded code), and the runtime's own schema ids `ptc_mm.*` (`cut.json`, the data, raw-snapshot, model and item-set
metadata, whose hashes are registered).

Note: `make_data.py` must reproduce the training shards and the committed item set byte for byte (seed 20261115;
`data/data_meta.json` and `items-test/item_set_meta.json` pin their SHA-256) to keep the registered data and item-set
hashes.

## Tests

```bash
python -m unittest discover -s tests -t tests
```

The tests pin the runtime to a neutral layout under a temporary directory in `build/` (set before gatedtrain loads the
rows). Tests that read regenerated files (the base model, the training shards) skip when they are absent;
`GATEDTRAIN_RECIPE_NEMO_RL_AGENTIC_MATH_RUNTIME` points them at a complete runtime, and
`GATEDTRAIN_RECIPE_NEMO_RL_AGENTIC_MATH_OUTPUTS` at the directory holding the GPU probe `p1-a3/` for the tests that read it.
The runtime's own CPU tests are `runtime/tests/` (run from `runtime/`; `GATEDTRAIN_NEMO_RL_DIR` names a NeMo RL clone at
4f39fd8, `GATEDTRAIN_RECIPE_NEMO_RL_AGENTIC_MATH_OUTPUTS` the run outputs `p1-a2/` and `probe-tool-2/` for the tests that
read them).

## Data and licenses

| What | Source and pin | License | Use here |
|---|---|---|---|
| MATH (Hendrycks et al., 2021) | `EleutherAI/hendrycks_math` @ `21a5633873b6a120296cce3e2df9d5550074f4a3` on the Hugging Face Hub, from https://github.com/hendrycks/math | MIT | downloaded by `runtime/fetch_data.py`; 500 test problems are redistributed in `runtime/items-test/items.jsonl`, a few training and test problems in the test fixtures (`THIRD_PARTY_NOTICES.md`) |
| Qwen3-1.7B (Qwen team, Alibaba Cloud) | `Qwen/Qwen3-1.7B` @ `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | Apache-2.0 (model card and `LICENSE` of the repository) | base model, downloaded by `runtime/fetch_models.py`; weights are never committed; the test fixture of a LoRA cut holds copies of its tokenizer configuration and chat template |
| NeMo RL (NVIDIA) | https://github.com/NVIDIA-NeMo/RL @ `4f39fd86289a2ffd5aaf5d202702f7fd7185c924` | Apache-2.0 | trainer, run in the image above; not redistributed (the runtime's tests contain a test double of its multi-turn rollout bookkeeping, written for these tests) |
| vLLM, math-verify, PyTorch and the rest of the NeMo RL image | as installed in `nemo-rl:4f39fd8` (`env-freeze.txt` of a run lists the versions) | their own licenses | run inside the image; not redistributed |

Please cite MATH as Hendrycks et al., "Measuring Mathematical Problem Solving With the MATH Dataset", NeurIPS 2021.
This repository's own code is Apache-2.0 (`LICENSE`).

## License

This repository is licensed under the Apache License, Version 2.0 (`LICENSE`); see `NOTICE`. Third-party material it
contains is listed with its license in `THIRD_PARTY_NOTICES.md`; models, data and images the recipe downloads or
builds are listed under "Data and licenses" above. Names of third-party projects and products are trademarks of
their owners; this project is not affiliated with or endorsed by them.

## Result

See `RESULT.md`.
