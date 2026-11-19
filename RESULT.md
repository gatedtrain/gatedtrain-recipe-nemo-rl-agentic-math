Prototype result of a single registered run; not a benchmark claim; not affiliated with or endorsed by the owners of the models, datasets, or frameworks used.

# Result (2026-11-16): gate PASS

Qwen/Qwen3-1.7B @ `70d244cc`, NeMo RL 4f39fd8 synchronous multi-turn GRPO with LoRA r 32 (alpha 64, every decoder
projection) in the derived image `ptc-a6-nemo-rl:1` (ptc, ptc-skel and ptc-hub are earlier names of gatedtrain; see "Rename to
gatedtrain 0.3.0"; uid 1000, `--network none`), environment `math_code`: MATH
problems with integer answers (EleutherAI/hendrycks_math @ `21a56338`) and a Python tool run in the executor sidecar,
at most four turns, the deployment prompt `prompt_math_code_oneshot.txt`. Deciding metric `exact_match` (the recipe's
own parser: the last `\boxed{}` of the final turn normalized to an integer), T3 with the construct-alignment and
shared-annotation disclosures (the training reward is NeMo RL's `math_verify` on the final turn, paid only when that
turn contains `\boxed`).

## Training and gate

full-1: 100 steps x 16 prompts x 8 generations (12,800 trajectories, rows in file order), cuts every 20 steps;
2.14 GPU-hours; training reward 0.61 at step 1, 0.86 at step 100.

| | |
|---|---|
| Deciding metric | `exact_match` on 500 held-out MATH test problems with integer answers |
| Baseline (base model) | 0.648 |
| Candidate (checkpoint-100) | 0.702 |
| Delta, 95% cluster-bootstrap interval | +0.054 [0.014, 0.096] |
| p-value | 0.0039 |
| Items, samples per item | 500, 1 (greedy, eval seed 11) |
| Achieved power | 0.999 |
| Exact overlap with training | 0 |
| Consumption precision | `row` (the environment's per-trajectory log joined with NeMo RL's step log, 12,800 records) |

Sub-metrics (not gated): used-tool share 0.412 to 0.884, answer found 0.836 to 0.914, truncated 0.108 to 0.024.
Consumption: 9,429 executed tool calls; end reasons final 11,661, no_answer 1,001, max_turns 138. The impact query on
training shard 0 maps the cuts to steps 20, 40, 60, 80 and 100, none orphaned or superseded.

One training trajectory ended at `max_turns` with reward 1.0: `agentic_core.MathCodeCore.step` verifies every terminal
turn that contains `\boxed`, a turn-4 tool call included. The adapter's end-reward table allows reward 0.0 or 1.0 at
`max_turns` for `math_code` (with a test); the evaluator scores `max_turns` 0, so training paid 1 of 12,800
trajectories for a form the evaluator does not count (disclosed, not changed).

## Re-collect with this repository's adapter (2026-11-16)

The run was re-collected with this repository's adapter (installed as a plugin of ptc, the runtime override pointing at
the original area directory, the input roots named as at launch) on the original run's outputs and compared with a
re-collect by ptc before the migration: every field of the gate report is equal except the run-id-derived
`lockfile_hash` (the lockfile differs in the run id, the ledger hash that follows from it, and the two
`checkpoint_completeness` `sources` texts, which now name files relative to this repository); the ledger (12,800 rows)
and the lineage are equal, and the impact query above gives the same cuts. Both re-collects also equal the canonical
2026-11-16 report except `lockfile_hash`.

## Published copies (redacted)

The files under results/2026-11-16/ are redacted copies (host paths replaced by `<data-root>`, references to
unpublished notes removed from the comments) of `gate/spec.yaml` and `gate/registration.yaml` (`spec.yaml` names the
original's hash); the gate-time originals are not published. `gate/trainer.yaml` is unchanged (`e365f0474a6439d045498fd4820867d2541ecbce7d09d3596ba5d768f04604f4`). The scripts in
`runtime/` whose SHA-256 the adapter pins or the run records (`agentic_core.py`, `gen_agentic.py`, `score_agentic.py`,
`math_code_env.py`, `traj_log.py`, `ptc_agentic_hooks.py`, `run_agentic_nemo.py`, `probe_tool_use.py`, `executor/*.py`,
the prompts) are byte for byte the ones that produced this result, except `executor/executor_server.py` and `executor/sandbox_runner.py` (section below) and, since the rename to ptc-hub 0.2.0, `score_agentic.py`, `run_agentic_nemo.py`, `agentic_core.py` and `executor/executor_server.py` again (last section). The other runtime files differ from the run's only
in comments and in how they find paths: `executor/run-executor.sh` derives the runtime directory from its own location
instead of a host path, and the runtime's CPU tests take the NeMo RL clone and the run outputs from environment
variables. The test fixtures copied from real outputs name the runtime directory by the neutral path
`/srv/ptc/nemo-rl-agentic-math/runtime`, with the hashes that cover those files updated.

## Executor hardening after the gate (M2)

A security review found that model code could leave the executor's Python-level sandbox (the audit hook) with
`ctypes` and then impersonate the executor (fork into a new session, replace `mathexec.sock` with its own socket and
answer later requests with forged results), write into a concurrently running job's result pipe through `/proc`, and
open the server's descriptors. `runtime/executor/executor_server.py` and `runtime/executor/sandbox_runner.py` were
changed so the boundary no longer depends on Python-level restrictions (per-job socket pairs instead of pipes,
non-dumpable server and runners, the server a child subreaper that kills every process a job left behind before it
answers, the socket path checked after every batch; `executor_server.py` docstring, "Trust boundary"). The wire
protocol and `exec_protocol.py` are unchanged.

| File | SHA-256 after M2 |
|---|---|
| `executor/executor_server.py` | `fc58950c62bc01c13e5028cd0733c51d7ce98584ba8eff497b972353efe01162` |
| `executor/sandbox_runner.py` | `4b656e183b51c8604fe09cdd092c6e98fdfbee50a9f5b8a32d4322586cd353f0` |

The versions the 2026-11-16 gate run used are not published (`rows.py` pins their hashes, below).

All historical runs (the p1 probes and the 2026-11-16 gate run) used the old version, and their records, the
registration under `results/2026-11-16/` and the test fixtures copied from them keep its hashes. The adapter pins both
versions (`rows.py` `MATHEXEC_CODE_SHA256` and `MATHEXEC_CODE_SHA256_GATE_2026_11_16`), requires one of them used
throughout a run (trainer ping, registered sampler, both arms' generation), and on collecting a run of the old version
raises `PreM2ExecutorWarning` (printed on the collect command's stderr). The warning is not a report field: it
adds nothing to a re-collected report.

## Rename to ptc-hub 0.2.0 (2026-11-17)

ptc-hub 0.2.0 renamed the persisted identifiers (schema ids, ledger writer id, analysis code version). The files under
results/2026-11-16/ and the byte-copied fixtures under tests/fixtures/agentic/real-arm/ keep the names ptc-skel 0.1.0
wrote; they are read by ptc `78eb913` (this and the other framework commits cited here, `e20736e` among them, are
unpublished pre-release commits of the framework, earlier named ptc-skel and ptc-hub; the published framework is
`gatedtrain` `v0.3.0`), with which the historical verdict remains
reproducible from the original `registration.yaml`/`spec.yaml` (and the pins before this change). The gate definition
is carried forward by results/2026-11-16/gate/registration-ptc-hub.yaml (amends
sha256:7ffdbe37e585203a5fa2faa76f5cb5514fd829a4b4cf2217cffbddbca45277cb, the published `registration.yaml` next to
it; the gate-time original is not published, and the published copy differs from the gate-time file only in redacted comments; post_hoc: false) and spec-ptc-hub.yaml. Re-pinned for the rename
(only the new pin is accepted):

| File | Old SHA-256 | New SHA-256 | Why |
|---|---|---|---|
| `runtime/score_agentic.py` (scorer) | `a750f5be01871a61e2ba08d5569d58634eaf4aeb35faeb636df790f5254c2ab2` | `f3a9e563fafd15e679567b324b13efddf97fa7b60e54c82638065a8b83713ca8` | name map only: writes `"schema": "ptc_hub.item_eval"` |
| `runtime/executor/executor_server.py` | `fc58950c62bc01c13e5028cd0733c51d7ce98584ba8eff497b972353efe01162` | `d1f3256d3f18e85b38950247cb2d1b6ae93c1a18a500e5b63d51bce8c6a9a6dc` | one docstring line without host-specific wording; code unchanged (the M2 executor; `MATHEXEC_CODE_SHA256`) |
| `runtime/run_agentic_nemo.py` | not published | `de95485149db2abbfc95d53a663e562ceaa44c9f716bf798aa49b54c79a185d0` | one docstring line without internal process wording; not pinned by the adapter, recorded by each run (`script_sha256`) |

The historical runs used the old versions (the gate run the pre-M2 executor, `MATHEXEC_CODE_SHA256_GATE_2026_11_16`,
which is unchanged). The scorer's composite `code_sha256` moves from
`sha256:f8e356f400bcdf7af9fecd0c1cd43926b416a1d8def8563cd77f621e59c952bb` to
`sha256:2d90f842065073974d1b00053780c7e08ed5e26cd884ddc74061bef07b74556c`, and the evaluator
`agentic.math_code_items@0.1` gets a new identity: `manifest_hash`
`sha256:5a09b30ea3eb1198b1771b5091044afa343affb47757859ddd1f8e9bfb6dba47` ->
`sha256:a2eac88bfec1af38d81bfb88fb4eb4407eef29374a2690f2f60b3446df1a5b82`. The amendment's `rename_proof` (design
revision 17, R17-3; `composite: true`) shows this is the rename only: `results/2026-11-16/gate/scorer-before/score_agentic.py`
is the gate-time scorer byte for byte (from git, SHA-256 `a750f5be...`), and ptc's `renamed.apply_name_map` of it is
today's `runtime/score_agentic.py`. No other registered value changed: the sampler (with the pre-M2
`executor_code_sha256`), `eval_distribution` and the other evaluator fields are as registered; `agentic_core.py` and
`gen_agentic.py` keep the hashes the gate run's training and generation recorded in the pinned gate version (below).

`runtime/agentic_core.py` lost an internal document reference in its docstring (the code is the same), so the turn
logic and generator have two named versions, as the executor has since M2 (`adapter.py`):

| Version | `agentic_core.py` | `gen_agentic.py`, `executor/exec_protocol.py` |
|---|---|---|
| `GEN_CODE_SHA256` (current) | `8c5028886847b5a19b70e9be07b1c728408e58c9f3647574fc06f6a43235bfc3` | unchanged |
| `GEN_CODE_SHA256_GATE_2026_11_16` | the gate-time file, not published (`adapter.py` pins its hash) | `8f32dcd2...`, `cbd8df69...` |

The gate version is the one the 2026-11-16 run recorded (checked against `full-1/train/run_args.json` and the four
`gen_meta.json` files of `eval-base-1` and `eval-cand-1`). The adapter accepts either, the same one throughout a run
(trainer `run_args.json` and both arms' generation), and on collecting a run of the gate version raises
`GateGenCodeWarning` on the collect command's stderr (not a report field). Since the gate-time file is not published,
a training run of the gate version (`full-1`, the probe `p1-a3`) also binds the current `agentic_core.py` beside its
launched script: the launch check (EXT-83) accepts that one difference, and only for a run whose recorded hashes are
one pinned version and a file beside the script that is the current version's (added 2026-11-18). The pinned byte copy
`tests/fixtures/agentic/runtime/agentic_core.py` is a test fixture of the current file (a test compares it with the
runtime's), so it follows the edit.

- Files: `registration-ptc-hub.yaml` (SHA-256 `49caad3be66a1a9ebcb93ece6cd1a977f9b3fcb3f89cdd0f867dd1d0e161058f`)
  differs from `registration.yaml` only in `analysis_code_version: ptc-hub 0.2.0`, `amends`, `created_at`
  (2026-11-17T17:51:11Z), the evaluator's `manifest_hash` and its comment, `post_hoc: false`, `rename_proof` and the
  amendment comment (which also names the gate-time original's hash); `spec-ptc-hub.yaml` (SHA-256 `a9d1e7e6e96564d0a70d1ba3db325716386e955d72c17eeceb57f6f55fe35266`)
  differs from `spec.yaml` only in the registration uri and the name map in a comment. `ptc-hub validate` accepts it
  (with `<data-root>` filled in); the original spec under ptc-hub 0.2.0 is refused with
  `error[REG-FIELD] registration.analysis_code_version` naming the amendment.
- Re-judged on a scratch replay with ptc `e20736e`: copies of the run's launch record (`train-full-1.launch.json`),
  smoke record (`train-smoke-1.smoke.json`) and both scoring directories had only their schema ids replaced
  (`ptc_hub.launch_record`, `ptc_hub.smoke_record`, `ptc_hub.item_eval`) and the proved scorer hashes: in each arm's
  `eval_meta.json` `scorer.code_sha256` `f8e356f4...` -> `2d90f842...`, and in the `env-runtime.json` beside it (which
  the adapter checks against its pin) the same pair in `scorer_code_sha256` and the member `score_agentic.py`
  `a750f5be...` -> `f3a9e563...`. The launch record's label is a checksum of the record and gpu-step.sh logs that label
  in the host's `gpu-guard.log`, so the replay record was moved to a scratch host-work directory holding a copy of the
  host's `gpu-step.sh` (its `WORK=` line changed) and the run's two guard-log lines under the re-hashed label;
  `host_work`, `gpu_step_sha256`, `host_config_sha256` (original input roots) and `label` were recomputed with ptc's
  own functions. The stage output is the original; the runtime override names the original runtime directory.
  `ext-collect` on `spec-ptc-hub.yaml` reported `registration_amendment: version_only`, `registration_amends:
  sha256:7ffdbe37...`, `post_hoc: false`, and (with `PreM2ExecutorWarning` and `GateGenCodeWarning` on stderr) against the canonical
  2026-11-16 report: status PASS, no status reasons, delta_hat 0.054, interval [0.014, 0.096], p value 0.00388,
  achieved power 0.99917, n_items 500, baseline 0.648, candidate 0.702, item-set and sampler hashes equal (65 report
  fields equal). The differences are exactly `analysis_code_version`, `created_at`, `run_id`, `schema`,
  `lockfile_hash`, `registration_hash` (`49caad3b...`), `registration_amends`, `registration_amendment` (new), and
  `evaluator_hash` / `evaluator_hash_registered` (`5a09b30e...` -> `a2eac88b...`, the pair the rename proof proves).
  The ledger (12,800 rows) is equal except `run_id`, the run-id-derived `sample_id` and `writer_id`
  (`ptc-hub.ledger@0.2.0`); the lineage (5 checkpoints, 4 shards, no orphans) is equal except the run id, and its
  edges except the run id, the registration hash and the schema id.

## Rename to gatedtrain 0.3.0 (2026-11-18)

The framework was renamed again, from ptc-hub 0.2.0 to gatedtrain 0.3.0 (distribution and import `gatedtrain`, schema
ids `gatedtrain.*`, ledger writer id `gatedtrain.ledger@0.3.0`); this repository is now
`gatedtrain-recipe-nemo-rl-agentic-math` (import `gatedtrain_recipe_nemo_rl_agentic_math`, entry point group
`gatedtrain.adapters`, runtime override `GATEDTRAIN_RECIPE_NEMO_RL_AGENTIC_MATH_RUNTIME`, runtime tests
`GATEDTRAIN_NEMO_RL_DIR` and `GATEDTRAIN_RECIPE_NEMO_RL_AGENTIC_MATH_OUTPUTS`). The files under results/2026-11-16/
keep their bytes (`registration.yaml`/`spec.yaml` as published, the `-ptc-hub` files as the ptc-hub 0.2.0 amendment,
`scorer-before/` as the gate-time scorer), and so do the fixtures copied from real outputs under tests/fixtures/agentic/
and runtime/tests/fixtures/ (including the neutral path `/srv/ptc/nemo-rl-agentic-math/runtime` they name). The
sections above describe what was done at the time and keep the names of their time (ptc, ptc-hub,
`ptc-a6-nemo-rl:1`).

Kept names: `runtime/ptc_agentic_hooks.py` (each run records the SHA-256 of its code files by these names in
`run_args.json`, and the adapter checks them beside the launched script), the trajectory stamp key `_ptc` that
`runtime/traj_log.py` writes into the environment metadata (`traj_log.py` is part of the recorded code), and the
runtime's own schema ids `ptc_mm.*` (`ptc_mm.agentic_cut` in the published cuts; `agentic_data`, `agentic_items`,
`agentic_models`, `agentic_raw` in committed metadata whose hashes are registered). The executor sidecar's container
name prefix is now `gatedtrain-mathexec-` (row `name_prefix` and `run-executor.sh serve`, which also prints `docker
--context gatedtrain` and reads `GATEDTRAIN_MATHEXEC_IMAGE` / `GATEDTRAIN_MATHEXEC_PYTHON`), as the framework prints it
(`<name_prefix><hashed label>`); the name is not part of any record.

Re-pinned (only the new pin is accepted): `runtime/score_agentic.py`
`sha256:f3a9e563fafd15e679567b324b13efddf97fa7b60e54c82638065a8b83713ca8` ->
`sha256:839de6f29b2b10f1dc9052563d1f5c31a1b0fcbca63c931e98045685d269a2c1` (name map only: it writes
`gatedtrain.item_eval`), so its composite `code_sha256` moves from
`sha256:2d90f842065073974d1b00053780c7e08ed5e26cd884ddc74061bef07b74556c` to
`sha256:3ede3a5a8b1bdced8b7eabd832139ddd3f489446016088e293fd5ac2df64febe` and the evaluator
`agentic.math_code_items@0.1` gets the identity `sha256:dcad26644bc590738e4f02dcba95881675e9676a09fe4ab9cd7bcb37f89d6bca`
(was `sha256:a2eac88bfec1af38d81bfb88fb4eb4407eef29374a2690f2f60b3446df1a5b82`). The two named pin versions are
unchanged and both still accepted: the executor (`MATHEXEC_CODE_SHA256`, current, and
`MATHEXEC_CODE_SHA256_GATE_2026_11_16`, pre-M2) and the turn logic and generator (`GEN_CODE_SHA256` and
`GEN_CODE_SHA256_GATE_2026_11_16`); no file of either set changed. `runtime/run_agentic_nemo.py` changed in one
comment (each run records its own `script_sha256`; the adapter does not pin it); `runtime/executor/run-executor.sh`
changed as above (not pinned).

- Amendment: results/2026-11-16/gate/registration-gatedtrain.yaml (SHA-256
  `7d61d026c8a5d4e29c1545a91d31ab77140d0650bb553b46cd89e3354e4263b8`, amends
  sha256:49caad3be66a1a9ebcb93ece6cd1a977f9b3fcb3f89cdd0f867dd1d0e161058f, the ptc-hub amendment; post_hoc: false)
  differs from `registration-ptc-hub.yaml` only in `analysis_code_version: gatedtrain 0.3.0`, `amends`, `created_at`
  (2026-11-18T09:53:45Z), the evaluator's `manifest_hash` and its comment, `rename_proof` (now from the ptc-hub-era
  scorer, `code_sha256_before` `2d90f842...`, file `scorer-ptc-hub/score_agentic.py`, a byte copy from git that hashes
  to `f3a9e563...`; the framework's `apply_name_map` of it is today's `score_agentic.py`), and comments (the amendment
  header and the name map); `spec-gatedtrain.yaml` (SHA-256
  `a10342d6adc218e3847f56044ecea9c07a3f5e888e28da086c1e3235c460a40e`) differs from `spec-ptc-hub.yaml` only in the
  registration uri, comments and the placeholder `<mm-root>` (was `<data-root>/ptc-mm`). `gatedtrain validate` accepts
  it (with `<mm-root>` filled in); `spec-ptc-hub.yaml` under gatedtrain 0.3.0 is refused with `error[REG-FIELD]
  registration.analysis_code_version` naming the amendment.
- Re-judged on a scratch replay with gatedtrain 0.3.0 (framework HEAD of 2026-11-18), as for the ptc-hub rename:
  copies of the run's launch record (`train-full-1.launch.json`), smoke record (`train-smoke-1.smoke.json`) and both
  scoring directories, made from the original records (ptc-skel ids), had only their schema ids replaced
  (`gatedtrain.launch_record`, `gatedtrain.smoke_record`, `gatedtrain.item_eval`) and the proved scorer hashes (each
  arm's `scorer.code_sha256` and the `env-runtime.json` `scorer_code_sha256` `f8e356f4...` -> `3ede3a5a...`, its member
  `score_agentic.py` `a750f5be...` -> `839de6f2...`); the launch record moved to a scratch host-work directory (a copy
  of the host's `gpu-step.sh` with its `WORK=` line changed, the run's two guard-log lines under the re-hashed label),
  with `host_work`, `gpu_step_sha256`, `host_config_sha256` and `label` recomputed by the framework's own functions.
  The stage output is the original; the runtime override names the original runtime directory. `ext-collect` on
  `spec-gatedtrain.yaml` reported `registration_amendment: version_only`, `registration_amends: sha256:49caad3b...`,
  `post_hoc: false`, and (with `PreM2ExecutorWarning` and `GateGenCodeWarning` on stderr, the gate run's pin versions)
  against the canonical 2026-11-16 report: status PASS, no status reasons, delta_hat 0.054, interval [0.014, 0.096],
  p value 0.00388, achieved power 0.99917, n_items 500, item-set and sampler hashes equal. The differences are exactly
  `analysis_code_version`, `created_at`, `run_id`, `schema`, `lockfile_hash`, `registration_hash` (`7d61d026...`),
  `registration_amends`, `registration_amendment` (new), and `evaluator_hash` / `evaluator_hash_registered`
  (`5a09b30e...` -> `dcad2664...`, the identity the two rename proofs carry forward). The ledger (12,800 rows) is equal
  except `run_id`, the run-id-derived `sample_id` and `writer_id` (`gatedtrain.ledger@0.3.0`); the lineage (5
  checkpoints, 4 shards, no orphans) is equal except the run id.
