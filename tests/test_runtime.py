"""The runtime code that runs inside the NeMo RL trainer and generation containers (runtime/ptc_agentic_hooks.py,
runtime/math_code_env.py, runtime/traj_log.py, runtime/gen_agentic.py, runtime/run_agentic_nemo.py) on CPU, without
NeMo RL, Ray, torch, vLLM, transformers, a GPU, a model or the training data: the modules they import are stand-ins in
sys.modules that record what they are handed, the executor sidecar is a stand-in that speaks exec_protocol over a real
Unix socket, and the problems are the committed test items in the row format make_data.py writes.

runtime/tests holds the area's own CPU tests; most need the model's tokenizer, the training data, a NeMo RL clone,
torch or sympy and skip without them. These run in CI from a fresh clone.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path

import _nemo_rl_agentic as V
from gatedtrain_recipe_nemo_rl_agentic_math import adapter as A

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "runtime"
for _p in (RUNTIME, RUNTIME / "executor"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import agentic_core  # noqa: E402
import exec_protocol  # noqa: E402
import gen_agentic  # noqa: E402
import math_code_env  # noqa: E402
import ptc_agentic_hooks as hooks  # noqa: E402
import run_agentic_nemo  # noqa: E402
import traj_log  # noqa: E402

ITEMS = RUNTIME / "items-test" / "items.jsonl"


def items(n: int = 500) -> list[dict]:
    return [json.loads(x) for x in ITEMS.read_text(encoding="utf-8").splitlines()[:n]]


def train_row(k: int, it: dict) -> dict:
    """A training row as make_data.py writes it (make_data.py:228-232) for a committed test item's problem."""
    return {"row_id": f"math-train-{k:05d}", "input": it["problem"],
            "messages": [{"role": "user", "content": it["problem"]}, {"role": "assistant", "content": it["answer"]}],
            "answer": it["answer"], "subject": it["subject"], "level": it["level"], "source": it["source"]}


def stub_modules(test: unittest.TestCase, mods: dict[str, types.ModuleType]) -> None:
    """Put `mods` (and empty parent packages) into sys.modules until the test ends."""
    full = dict(mods)
    for name in list(mods):
        parts = name.split(".")
        for i in range(1, len(parts)):
            full.setdefault(".".join(parts[:i]), types.ModuleType(".".join(parts[:i])))
    saved = {k: sys.modules.get(k) for k in full}
    sys.modules.update(full)

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    test.addCleanup(restore)


def module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]


class FakeExecutor:
    """The executor sidecar's protocol (exec_protocol.py) over a real Unix socket, without running code: `print(6 *
    7)` prints 42, a cell with `1/0` raises, any other cell prints how many cells the job replayed."""

    def __init__(self, test: unittest.TestCase, self_check: str = "42\n") -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="fx-", dir="/tmp"))
        test.addCleanup(shutil.rmtree, self.dir, True)
        self.path = str(self.dir / "x.sock")
        self.self_check = self_check
        self.requests: list[dict] = []
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(self.path)
        self.srv.listen(8)
        self.srv.settimeout(0.1)
        self.stop = False
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()
        test.addCleanup(self.close)

    def close(self) -> None:
        self.stop = True
        self.thread.join(5)
        self.srv.close()

    def serve(self) -> None:
        while not self.stop:
            try:
                conn, _ = self.srv.accept()
            except (socket.timeout, OSError):
                continue
            with conn:
                req = exec_protocol.recv_msg(conn)
                self.requests.append(req)
                exec_protocol.send_msg(conn, self.answer(req))

    def answer(self, req: dict) -> dict:
        if req["op"] == "ping":
            return {"ok": True, "limits": {"wall_s": 6.0}, "versions": {"python": "3.13.14"}}
        results = []
        for j in req["jobs"]:
            code = j["cells"][-1]
            if code == "print(6 * 7)":
                out, st = self.self_check, "ok"
            elif "1/0" in code:
                out, st = "ZeroDivisionError (line 1): division by zero\n", "error"
            else:
                out, st = f"{len(j['cells'])} cells\n", "ok"
            results.append({"id": j["id"], "status": st, "output": out, "output_chars": len(out), "wall_s": 0.01})
        return {"results": results}

    def jobs(self) -> list[dict]:
        return [j for r in self.requests if r["op"] == "run" for j in r["jobs"]]


class FakeTensor(list):
    def tolist(self):
        return list(self)

    def float(self):
        return FakeTensor(float(x) for x in self)

    def mean(self):
        return types.SimpleNamespace(item=lambda: sum(self) / len(self))


# ------------------------------------------------------------------------------------------- trajectory log ---
class TestTrajectoryLogger(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)

    def test_stamps_and_lines(self):
        tlog = traj_log.TrajectoryLogger(str(self.dir / "logs"), "math_code")
        user = {"role": "user", "content": "p"}
        convs = [[user, {"role": "assistant", "content": "<code>x</code>"}],
                 [user, {"role": "assistant", "content": "\\boxed{3}"}],
                 [user, {"role": "assistant", "content": "<code>y</code>"}]]
        md_in = [{"row_id": "r0", "ground_truth": "1"}, {"row_id": "r1"}, {"row_id": "r2"}]
        md = tlog.record(convs, md_in, [{"row_id": "r0", "cells": [1]}, None, None], [0.0, 1.0, 0.0],
                         [False, True, False], ["<result>\n1\n</result>\n", "", "obs"], [None, "final", None],
                         ["ok", None, "error"])
        self.assertEqual(md[0], {"row_id": "r0", "cells": [1], "_ptc": {"traj_call": 1, "traj_pos": 0}})
        self.assertIsNone(md[1])                                     # terminated: no metadata back
        self.assertEqual(md[2], {"row_id": "r2", "_ptc": {"traj_call": 1, "traj_pos": 2}})   # None kept as NeMo RL does
        lines = read_jsonl(self.dir / "logs" / "env_calls.jsonl")
        self.assertEqual([(x["call"], x["position"], x["traj_pos"], x["turn"], x["row_id"]) for x in lines],
                         [(1, 0, 0, 1, "r0"), (1, 1, 1, 1, "r1"), (1, 2, 2, 1, "r2")])
        self.assertEqual([x["end_reason"] for x in lines], [None, "final", None])
        self.assertEqual([x["obs_chars"] for x in lines], [len("<result>\n1\n</result>\n"), 0, 3])
        self.assertEqual(lines[1]["response_sha256"], V.sha_text("\\boxed{3}"))
        self.assertEqual([x["tool_status"] for x in lines], ["ok", None, "error"])
        [end] = read_jsonl(self.dir / "logs" / "trajectories.jsonl")
        self.assertEqual((end["traj_call"], end["traj_pos"], end["row_id"], end["end_reason"]), (1, 1, "r1", "final"))
        # turn 2 of trajectories 0 and 2, now at positions 0 and 1: the stamps carry their identity
        conv0 = convs[0] + [{"role": "environment", "content": "o"}, {"role": "assistant", "content": "\\boxed{1}"}]
        tlog.record([conv0, convs[2]], [md[0], md[2]], [None, None], [1.0, 0.0], [True, True], ["", ""])
        lines = read_jsonl(self.dir / "logs" / "env_calls.jsonl")[3:]
        self.assertEqual([(x["call"], x["position"], x["traj_call"], x["traj_pos"], x["turn"]) for x in lines],
                         [(2, 0, 1, 0, 2), (2, 1, 1, 2, 1)])
        self.assertEqual(lines[0]["response_sha256"], V.sha_text("<code>x</code>\\boxed{1}"))
        self.assertEqual([x["end_reason"] for x in lines], ["terminated", "terminated"])
        self.assertEqual(len(read_jsonl(self.dir / "logs" / "trajectories.jsonl")), 3)

    def test_without_log_dir_nothing_is_written(self):
        tlog = traj_log.TrajectoryLogger(None, "math_code")
        md = tlog.record([[{"role": "assistant", "content": "a"}]], [None], [None], [0.0], [True], [""])
        self.assertEqual(md, [None])
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_sliding_end_reason(self):
        self.assertEqual((traj_log.sliding_end_reason(1.0), traj_log.sliding_end_reason(0.0)), ("solved", "max_moves"))


# -------------------------------------------------------------------------------------------- environment ---
class TestRowIdProcessor(unittest.TestCase):
    """math_code_rowid_processor against a stand-in `math_hf_data_processor` doing what NeMo RL's does
    (processors.py:553-614: `prompt.format(problem)` through the chat template, ground truth = messages[1].content)."""

    def test_rows(self):
        seen = []

        def math_hf_data_processor(datum, spec, tokenizer, max_seq_length, idx):
            content = spec.prompt.format(datum["messages"][0]["content"])
            seen.append(content)
            return {"message_log": [{"role": "user", "content": content}], "idx": idx, "loss_multiplier": 1.0,
                    "extra_env_info": {"ground_truth": datum["messages"][1]["content"]}}

        stub_modules(self, {"nemo_rl.data.processors": module("nemo_rl.data.processors",
                                                              math_hf_data_processor=math_hf_data_processor)})
        spec = types.SimpleNamespace(prompt=run_agentic_nemo.PROMPT.read_text(encoding="utf-8"), system_prompt=None)
        for k, it in enumerate(items(20)):
            row = train_row(k, it)
            out = math_code_env.math_code_rowid_processor(row, spec, None, run_agentic_nemo.MAX_SEQ, k)
            self.assertEqual(out["extra_env_info"], {"ground_truth": it["answer"], "row_id": row["row_id"]})
            self.assertEqual(out["stop_strings"], ["</code>"])
            self.assertTrue(seen[-1].endswith("Problem: " + it["problem"]))       # LaTeX braces kept as they are
            self.assertIn("\\boxed{answer}", seen[-1])                             # the template's {{ }} unescaped


class EnvCase(unittest.TestCase):
    def setUp(self):
        self.ex = FakeExecutor(self)
        self.log = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.log)

    def batch(self):
        user = {"role": "user", "content": "prompt"}
        convs = [[user, {"role": "assistant", "content": "Let me compute.\n<code>\nprint(2**10)\n</code>"}],
                 [user, {"role": "assistant", "content": "It is \\boxed{7}."}],
                 [user, {"role": "assistant", "content": "I think it is 7."}]]
        md = [{"ground_truth": "1024", "row_id": "r0"}, {"ground_truth": "7", "row_id": "r1"},
              {"ground_truth": "7", "row_id": "r2"}]
        return convs, md


class TestEnvStep(EnvCase):
    """math_code_env.env_step, the body of MathCodeEnvironment.step, over the stand-in executor."""

    def test_turns(self):
        core = math_code_env.build_core({"socket": self.ex.path, "timeout_s": 10, "max_turns": 4},
                                        verifier=lambda text, gt: float(f"\\boxed{{{gt}}}" in text))
        tlog = traj_log.TrajectoryLogger(str(self.log), "math_code")
        convs, md = self.batch()
        obs, md_out, stops, rewards, terms = math_code_env.env_step(core, tlog, convs, md)
        self.assertEqual(obs, ["<result>\n1 cells\n</result>\n", "", ""])
        self.assertEqual((rewards, terms), ([0.0, 1.0, 0.0], [False, True, True]))
        self.assertEqual(stops, [["</code>"], None, None])
        self.assertEqual(self.ex.jobs(), [{"id": "1:0", "cells": ["print(2**10)"]}])
        self.assertNotIn("1024", json.dumps(self.ex.requests))                 # no reference data to the executor
        self.assertEqual(md_out[0]["cells"], [{"code": "print(2**10)", "status": "ok"}])
        self.assertEqual(md_out[0]["_ptc"], {"traj_call": 1, "traj_pos": 0})
        self.assertEqual([x["end_reason"] for x in read_jsonl(self.log / "env_calls.jsonl")],
                         [None, "final", "no_answer"])
        # turn 2: the earlier cell is replayed before the new one, the stamp stays
        conv = convs[0] + [{"role": "environment", "content": obs[0]},
                           {"role": "assistant", "content": "<code>print(3)</code>"}]
        obs2, md2, *_ = math_code_env.env_step(core, tlog, [conv], [md_out[0]])
        self.assertEqual(self.ex.jobs()[-1]["cells"], ["print(2**10)", "print(3)"])
        self.assertEqual(obs2, ["<result>\n2 cells\n</result>\n"])
        self.assertEqual(md2[0]["_ptc"], {"traj_call": 1, "traj_pos": 0})

    def test_executor_fault_raises(self):
        core = math_code_env.build_core({"socket": str(self.log / "missing.sock"), "timeout_s": 1})
        tlog = traj_log.TrajectoryLogger(str(self.log), "math_code")
        convs, md = self.batch()
        with self.assertRaises(exec_protocol.ExecutorUnavailable):
            math_code_env.env_step(core, tlog, convs, md)
        self.assertFalse((self.log / "env_calls.jsonl").exists())


class TestEnvironmentActor(EnvCase):
    """MathCodeEnvironment, math_code_env.py loaded under stand-in ray, torch and nemo_rl modules; the reward goes
    through the HFVerifyWorker stand-in, which returns 0 unless called with math_verify_impl="hf_math_verify" as NeMo
    RL's does (math_environment.py:131-146)."""

    def setUp(self):
        super().setUp()
        self.verify_calls = []
        calls = self.verify_calls

        class HFVerifyWorker:
            def verify(self, preds, gts, return_extracted_answer=False, **kw):
                calls.append(kw)
                if kw.get("math_verify_impl") != "hf_math_verify":
                    return [0.0] * len(preds)
                return [float(f"\\boxed{{{g}}}" in p) for p, g in zip(preds, gts)]

        ret_fields = ("observations", "metadata", "next_stop_strings", "rewards", "terminateds", "answers")
        stub_modules(self, {
            "ray": module("ray", remote=lambda **kw: (lambda cls: cls)),
            "torch": module("torch", tensor=lambda x, dtype=None: FakeTensor(x), float32="float32", bool="bool"),
            "nemo_rl.environments.interfaces": module(
                "nemo_rl.environments.interfaces", EnvironmentInterface=object,
                EnvironmentReturn=__import__("collections").namedtuple("EnvironmentReturn", ret_fields)),
            "nemo_rl.environments.math_environment": module(
                "nemo_rl.environments.math_environment",
                HFVerifyWorker=types.SimpleNamespace(__ray_actor_class__=HFVerifyWorker))})
        import importlib.util
        spec = importlib.util.spec_from_file_location("math_code_env_under_ray", RUNTIME / "math_code_env.py")
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_step(self):
        env = self.mod.MathCodeEnvironment({"socket": self.ex.path, "log_dir": str(self.log), "max_turns": 4})
        convs, md = self.batch()
        ret = env.step(convs, md)
        self.assertEqual(ret.observations[0], {"role": "environment", "content": "<result>\n1 cells\n</result>\n"})
        self.assertEqual(ret.rewards.tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(ret.terminateds.tolist(), [False, True, True])
        self.assertEqual(ret.next_stop_strings, [["</code>"], None, None])
        self.assertIsNone(ret.answers)
        self.assertEqual(self.verify_calls, [{"math_verify_impl": "hf_math_verify"}])   # only the boxed answer
        self.assertEqual(env.global_post_process_and_metrics({"total_reward": ret.rewards})[1],
                         {"math_code_reward": 1 / 3})
        self.assertEqual(env.global_post_process_and_metrics({})[1], {"math_code_reward": 0.0})
        self.assertIsNone(env.shutdown())
        self.assertEqual(len(read_jsonl(self.log / "env_calls.jsonl")), 3)
        self.assertEqual(len(read_jsonl(self.log / "trajectories.jsonl")), 2)

    def test_host_import_has_no_actor(self):
        self.assertIsNone(math_code_env.MathCodeEnvironment)


# -------------------------------------------------------------------------------------------- cut writer ---
class CutCase(unittest.TestCase):
    """A NeMo RL step directory (PEFT adapter files or consolidated weights), the base directory and MODELS.json."""

    STEP = 5

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.out = self.tmp / "out"
        self.out.mkdir()
        self.base = V.build_base(self.tmp / "models" / "qwen3-1.7b")
        V.write_json(self.tmp / "models" / "MODELS.json", V.models_json(self.base))
        self.meta = {"seed": 10100, "profile": "smoke", "stage": "agentic_grpo", "env_id": "math_code",
                     "memory_rung": "a", "lora": {"dim": 32, "alpha": 64}, "init_checkpoint": None,
                     "script_sha256": "sha256:" + "1" * 64, "examples_consumed": lambda s: 16 * s}

    def step_dir(self, *, lora: bool = True, adapter_cfg: dict | None = None, weights=("model.safetensors",),
                 index: bool = False, tokenizer=("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                                                 "chat_template.jinja"), in_weights=()) -> Path:
        d = self.out / "nemo" / "checkpoints" / f"step_{self.STEP}"
        w = d / "policy" / "weights" / "model"
        w.mkdir(parents=True)
        if lora:
            (w / "adapter_model.safetensors").write_bytes(V.safetensors_bytes("lora"))
            V.write_json(w / "adapter_config.json", adapter_cfg or V.adapter_config())
            V.write_json(w / "automodel_peft_config.json", {"dim": 32, "alpha": 64})
        else:
            cons = w / "consolidated"
            cons.mkdir()
            for name in weights:
                (cons / name).write_bytes(V.safetensors_bytes(name))
            if index:
                V.write_json(cons / "model.safetensors.index.json", {"weight_map": {n: n for n in weights}})
            V.write_json(cons / "config.json", {"architectures": ["Qwen3ForCausalLM"]})
        tok = d / "policy" / "tokenizer"
        tok.mkdir(parents=True)
        for name in tokenizer:
            (tok / name).write_text(f"tokenizer {name}\n", encoding="utf-8")
        for name in in_weights:
            (w / name).write_text(f"saved with the adapter {name}\n", encoding="utf-8")
        (d / "training_info.json").write_text(json.dumps({"current_step": self.STEP}), encoding="utf-8")
        return d

    def write(self, sd: Path) -> Path:
        return hooks.write_cut(sd, self.STEP, self.out, self.base, self.meta)


class TestWriteCut(CutCase):
    def test_lora_cut_is_accepted_by_adapter_and_evaluator(self):
        sd = self.step_dir()
        cut = self.write(sd)
        self.assertEqual(cut, self.out / "cuts" / f"checkpoint-{self.STEP}")
        self.assertIsNone(A.NemoRlAgenticAdapter().cut_problem(cut, self.STEP))
        doc = json.loads((cut / "cut.json").read_text())
        self.assertEqual((doc["kind"], doc["lora"], doc["examples_consumed"]), ("lora_adapter", {"dim": 32, "alpha": 64},
                                                                                80))
        for name in hooks.LORA_FILES:
            self.assertTrue(os.path.samefile(cut / name, sd / "policy" / "weights" / "model" / name), name)
        self.assertFalse(os.path.samefile(cut / "tokenizer.json", sd / "policy" / "tokenizer" / "tokenizer.json"))
        self.assertEqual(doc["file_origin"]["tokenizer.json"], "policy/tokenizer/tokenizer.json")
        self.assertEqual(json.loads((cut / "sampler.json").read_text()), gen_agentic.SAMPLER)
        self.assertEqual(gen_agentic.check_cut(cut, self.base)["global_step"], self.STEP)
        self.assertEqual([x["cut"] for x in read_jsonl(self.out / "cuts.jsonl")], [cut.name])
        self.assertEqual(sorted(p.name for p in (self.out / "cuts").iterdir()), [cut.name])

    def test_same_layout_as_the_adapter_fixture_cut(self):
        # tests/_nemo_rl_agentic.build_cut stands in for write_cut in the adapter's scenarios: both must agree.
        mine = self.write(self.step_dir())
        other = V.build_cut(self.tmp / "fixture-run", self.STEP, profile="smoke", seed=10100,
                            script_sha256=self.meta["script_sha256"], base=self.base)
        self.assertEqual(sorted(p.name for p in mine.iterdir()), sorted(p.name for p in other.iterdir()))
        a, b = (json.loads((d / "cut.json").read_text()) for d in (mine, other))
        self.assertEqual({k: v for k, v in a.items() if k != "files"}, {k: v for k, v in b.items() if k != "files"})
        self.assertEqual(json.loads((mine / "base_ref.json").read_text()),
                         json.loads((other / "base_ref.json").read_text()))
        sampler = json.loads((other / "sampler.json").read_text())
        self.assertEqual({k: sampler[k] for k in gen_agentic.SAMPLER_MUST_MATCH},
                         {k: gen_agentic.SAMPLER[k] for k in gen_agentic.SAMPLER_MUST_MATCH})

    def test_tokenizer_saved_with_the_adapter(self):
        cut = self.write(self.step_dir(tokenizer=("tokenizer.json",), in_weights=("tokenizer_config.json",)))
        origin = json.loads((cut / "cut.json").read_text())["file_origin"]
        self.assertEqual(origin, {"tokenizer.json": "policy/tokenizer/tokenizer.json",
                                  "tokenizer_config.json": "policy/weights/model/tokenizer_config.json"})

    def test_lora_refusals(self):
        cases = [({"adapter_cfg": {**V.adapter_config(), "r": 8}}, "adapter_config"),
                 ({"tokenizer": ("tokenizer.json",)}, "tokenizer_config.json")]
        for kw, msg in cases:
            with self.subTest(msg):
                shutil.rmtree(self.out)
                self.out.mkdir()
                with self.assertRaisesRegex(RuntimeError, msg):
                    self.write(self.step_dir(**kw))
                self.assertFalse((self.out / "cuts" / f"checkpoint-{self.STEP}").exists())
                self.assertFalse((self.out / "cuts.jsonl").exists())
        shutil.rmtree(self.out)
        self.out.mkdir()
        sd = self.step_dir()
        (sd / "policy" / "weights" / "model" / "adapter_model.safetensors").unlink()
        with self.assertRaisesRegex(RuntimeError, "no adapter_model"):
            self.write(sd)

    def test_full_weight_cuts(self):
        self.meta["lora"] = None
        cases = [(("model.safetensors",), False, None), (("a.safetensors", "b.safetensors"), True, None),
                 (("a.safetensors", "b.safetensors"), False, "no index"), ((), False, "no consolidated")]
        for weights, index, refused in cases:
            with self.subTest(weights=weights, index=index):
                shutil.rmtree(self.out)
                self.out.mkdir()
                sd = self.step_dir(lora=False, weights=weights, index=index)
                if refused:
                    with self.assertRaisesRegex(RuntimeError, refused):
                        self.write(sd)
                    continue
                cut = self.write(sd)
                doc = json.loads((cut / "cut.json").read_text())
                self.assertEqual((doc["kind"], doc["lora"]), ("full_weights", None))
                self.assertEqual(index, "model.safetensors.index.json" in doc["files"])
                for name in (*weights, "config.json"):
                    self.assertTrue(os.path.samefile(cut / name, sd / "policy/weights/model/consolidated" / name))
                self.assertIn("config.json", doc["deployable"])
                self.assertNotIn("base_ref.json", doc["deployable"])

    def test_stale_partial_is_replaced(self):
        partial = self.out / "cuts" / f".checkpoint-{self.STEP}.partial"
        partial.mkdir(parents=True)
        (partial / "junk").write_text("x")
        cut = self.write(self.step_dir())
        self.assertIsNone(A.NemoRlAgenticAdapter().cut_problem(cut, self.STEP))


class TestEvaluatorCutChecks(CutCase):
    def test_check_cut(self):
        cut = self.write(self.step_dir())
        gen_agentic.check_cut(cut, self.base)
        (cut / "stray.txt").write_text("x")
        with self.assertRaisesRegex(SystemExit, "unlisted files"):
            gen_agentic.check_cut(cut, self.base)
        (cut / "stray.txt").unlink()
        (self.base / "config.json").write_text("{}")
        with self.assertRaisesRegex(SystemExit, "base_ref"):
            gen_agentic.check_cut(cut, self.base)

    def test_sampler_of_another_prompt_refused(self):
        cut = self.write(self.step_dir())
        sampler = json.loads((cut / "sampler.json").read_text())
        old = dict(sampler, max_new_tokens_per_turn=512, max_model_len=2048,
                   prompt_file_sha256=hooks.sha256_file(RUNTIME / "prompt_math_code.txt"))
        (cut / "sampler.json").write_text(json.dumps(old))
        with self.assertRaisesRegex(SystemExit, "prompt_file_sha256"):
            gen_agentic.check_sampler(cut)


class TestInstallCutWriter(CutCase):
    def test_publish_builds_the_cut(self):
        root = self.out / "nemo" / "checkpoints"

        class Storage:
            def publish(self, path):
                os.rename(path, Path(path).parent / Path(path).name.removeprefix("tmp_"))

        class CheckpointManager:
            def __init__(self, config):
                self._storage = Storage()

        stub_modules(self, {"nemo_rl.utils.checkpoint": module("nemo_rl.utils.checkpoint",
                                                               CheckpointManager=CheckpointManager)})
        hooks.install_cut_writer(self.out, self.base, self.meta)
        tmp = root / f"tmp_step_{self.STEP}"
        os.rename(self.step_dir(), tmp)
        CheckpointManager({"checkpoint_dir": str(root)})._storage.publish(str(tmp))
        self.assertIsNone(A.NemoRlAgenticAdapter().cut_problem(self.out / "cuts" / f"checkpoint-{self.STEP}",
                                                               self.STEP))


class TestRunRecord(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.out)

    def test_attempts(self):
        self.assertEqual([hooks.record_attempt(self.out, e, stage="agentic_grpo") for e in ("start", "end", "start")],
                         [1, 1, 2])

    def test_run_args(self):
        hooks.write_run_args(self.out, {"stage": "agentic_grpo", "p": Path("/x")})
        doc = json.loads((self.out / "run_args.json").read_text())
        self.assertEqual((doc["stage"], doc["p"], doc["framework"]["name"]), ("agentic_grpo", "/x", "nemo_rl"))
        self.assertIn("# nemo-rl commit", (self.out / "env-freeze.txt").read_text())


# -------------------------------------------------------------------------------------------- generation ---
class TestGenerationHelpers(unittest.TestCase):
    def test_select_items_covers_the_file(self):
        its = items()
        self.assertIs(gen_agentic.select_items(its, None), its)
        self.assertIs(gen_agentic.select_items(its, 600), its)
        picked = gen_agentic.select_items(its, 20)
        self.assertEqual([p["item"] for p in picked], [its[i * 25]["item"] for i in range(20)])
        self.assertEqual({p["subject"] for p in picked}, {it["subject"] for it in its})

    def test_wait_for_executor(self):
        ex = FakeExecutor(self)
        self.assertTrue(gen_agentic.wait_for_executor(ex.path)["ok"])
        sleeps = []
        old = gen_agentic.time.sleep
        gen_agentic.time.sleep = sleeps.append
        self.addCleanup(setattr, gen_agentic.time, "sleep", old)
        with self.assertRaisesRegex(SystemExit, "did not answer"):
            gen_agentic.wait_for_executor(str(ex.dir / "missing.sock"), deadline_s=0.05)

    def test_executor_fault_retried_once_then_sandbox_failure(self):
        calls = []

        def flaky(jobs):
            calls.append(len(jobs))
            if len(calls) in (1, 3, 4):
                raise exec_protocol.ExecutorUnavailable("down")
            return [{"id": j["id"], "status": "ok", "output": "5\n", "output_chars": 2} for j in jobs]

        core = agentic_core.MathCodeCore(execute=flaky)
        states = [{"item": "a", "ids": [0] * 4, "turns": [], "cells": [], "done": False, "end_reason": None,
                   "error": None}]
        gen = lambda ids, mt: [([1, 1], "<code>print(5)</code>", "stop") for _ in ids]
        gen_agentic.run_trajectories(states, gen, lambda t: [2] * len(t), core, max_model_len=4096)
        self.assertEqual(calls, [1, 1, 1, 1])
        self.assertEqual(states[0]["error"]["class"], "sandbox_failure")
        self.assertEqual(states[0]["turns"][0]["tool_status"], "ok")      # turn 1 succeeded on the retry
        self.assertIsNone(states[0]["end_reason"])

    def test_overflow_cuts_the_observation(self):
        core = agentic_core.MathCodeCore(execute=lambda jobs: [{"id": j["id"], "status": "ok", "output": "x" * 50,
                                                                 "output_chars": 50} for j in jobs])
        states = [{"item": "a", "ids": [0] * 10, "turns": [], "cells": [], "done": False, "end_reason": None,
                   "error": None}]
        seen = []

        def gen(ids, max_tokens):
            seen.append(list(max_tokens))
            return [([1] * 5, "<code>p</code>", "stop") for _ in ids]
        gen_agentic.run_trajectories(states, gen, lambda t: [2] * len(t), core, max_model_len=40)
        self.assertEqual(seen, [[30]])
        self.assertEqual((states[0]["end_reason"], len(states[0]["ids"])), ("overflow", 40))


class FakeTokenizer:
    """Character-level token ids; records chat-template and encode calls."""

    def __init__(self):
        self.template_calls, self.encode_flags = [], []

    def apply_chat_template(self, msgs, **kw):
        self.template_calls.append((msgs, kw))
        return f"<|im_start|>user\n{msgs[0]['content']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __call__(self, text, add_special_tokens=True):
        self.encode_flags.append(add_special_tokens)
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


class TestGenerationMain(CutCase):
    """gen_agentic.main with stand-in transformers and vLLM modules that record the requests, a LoRA cut written by
    write_cut, the stand-in executor and two committed items."""

    SCRIPTS = (["I will compute.\n<code>\nprint(6 * 7)\n</code>", "So the answer is \\boxed{42}."],
               ["<code>1/0</code>", "<code>print(1)</code>", "<code>print(2)</code>", "<code>print(3)</code>"])

    def setUp(self):
        super().setUp()
        self.ex = FakeExecutor(self)
        self.items = items(2)
        d = self.tmp / "items"
        d.mkdir()
        (d / "items.jsonl").write_text("".join(json.dumps(it) + "\n" for it in self.items))
        self.items_dir = d
        self.tok = FakeTokenizer()
        self.template = gen_agentic.PROMPT_FILE.read_text(encoding="utf-8")
        self.llm_kwargs, self.generate_calls, self.loras = [], [], []
        test = self

        class LLM:
            def __init__(self, **kw):
                test.llm_kwargs.append(kw)

            def generate(self, prompts, sps, lora_request=None, use_tqdm=True):
                test.generate_calls.append((prompts, sps, lora_request))
                outs = []
                for p, sp in zip(prompts, sps):
                    text = "".join(chr(i) for i in p["prompt_token_ids"])
                    k = next(i for i, it in enumerate(test.items) if it["problem"] in text)
                    turn = text.count("<result>") - test.template.count("<result>")    # the example has one
                    reply = test.SCRIPTS[k][turn][:sp.max_tokens]
                    outs.append(types.SimpleNamespace(outputs=[types.SimpleNamespace(
                        token_ids=[ord(c) for c in reply], finish_reason="stop")]))
                return outs

        stub_modules(self, {
            "transformers": module("transformers", __version__="0-stub", AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda path, local_files_only=False: self.tok)),
            "vllm": module("vllm", __version__="0.29.0-stub", LLM=LLM,
                           SamplingParams=lambda **kw: types.SimpleNamespace(**kw)),
            "vllm.lora.request": module("vllm.lora.request", LoRARequest=lambda *a: self.loras.append(a) or a)})

    def run_main(self, *extra: str) -> Path:
        out = self.tmp / "gen-out"
        argv = ["--model", str(self.base), "--items", str(self.items_dir), "--out", str(out), "--socket",
                self.ex.path, *extra]
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gen_agentic.main(argv), 0)
        return out

    def test_requests_and_outputs_with_a_lora_cut(self):
        cut = self.write(self.step_dir())
        out = self.run_main("--adapter", str(cut))
        tmpl = gen_agentic.PROMPT_FILE.read_text(encoding="utf-8")
        self.assertEqual([m for m, _ in self.tok.template_calls],
                         [[{"role": "user", "content": tmpl.format(it["problem"])}] for it in self.items])
        for _, kw in self.tok.template_calls:
            self.assertEqual((kw["tokenize"], kw["add_generation_prompt"], kw["enable_thinking"]), (False, True, False))
        self.assertEqual(set(self.tok.encode_flags), {False})
        [kw] = self.llm_kwargs
        self.assertEqual((kw["max_model_len"], kw["enable_lora"], kw["max_lora_rank"], kw["max_loras"]),
                         (4096, True, 32, 1))
        self.assertEqual(kw["model"], str(self.base))
        self.assertEqual(self.loras, [("cut", 1, str(cut))])
        prompts, sps, lora = self.generate_calls[0]
        self.assertEqual([c[2] for c in self.generate_calls], [("cut", 1, str(cut))] * len(self.generate_calls))
        for p, sp in zip(prompts, sps):
            self.assertEqual((sp.temperature, sp.top_p, sp.seed, sp.stop, sp.include_stop_str_in_output),
                             (0.0, 1.0, 11, ["</code>"], True))
            self.assertEqual(sp.max_tokens, min(1024, 4096 - len(p["prompt_token_ids"])))
        # turn 2 continues from the running ids: prompt + turn 1 + observation tokens
        first = "".join(chr(i) for i in self.generate_calls[0][0][0]["prompt_token_ids"])
        second = "".join(chr(i) for i in self.generate_calls[1][0][0]["prompt_token_ids"])
        self.assertEqual(second, first + self.SCRIPTS[0][0] + "<result>\n42\n</result>\n")
        self.assertEqual([j["cells"] for j in self.ex.jobs() if j["cells"][0] == "1/0"],
                         [["1/0"], ["1/0", "print(1)"], ["1/0", "print(1)", "print(2)"]])
        docs = [json.loads((out / "gen" / f"{it['item']}__s11.json").read_text()) for it in self.items]
        self.assertEqual([d["end_reason"] for d in docs], ["final", "max_turns"])
        self.assertEqual(docs[0]["final_text"], self.SCRIPTS[0][1])
        self.assertEqual([t["tool_status"] for t in docs[1]["turns"]], ["error", "ok", "ok", None])
        self.assertEqual(docs[0]["total_tokens"], docs[0]["prompt_tokens"] + docs[0]["model_tokens"]
                         + docs[0]["env_tokens"])
        meta = json.loads((out / "gen_meta.json").read_text())
        cut_doc = json.loads((cut / "cut.json").read_text())
        self.assertEqual(meta["adapter"]["files"], {f: h for f, h in cut_doc["files"].items()
                                                    if f in cut_doc["deployable"]})
        self.assertEqual((meta["adapter"]["global_step"], meta["cut"]), (self.STEP, None))
        self.assertEqual(meta["sampler"], gen_agentic.SAMPLER)
        self.assertEqual(meta["item_ids"], [it["item"] for it in self.items])
        self.assertEqual(meta["executor"]["versions"], {"python": "3.13.14"})
        self.assertEqual(set(meta["code_sha256"]), {"gen_agentic.py", "agentic_core.py", "executor/exec_protocol.py"})

    def test_base_model_without_adapter(self):
        self.run_main("--limit", "1")
        [kw] = self.llm_kwargs
        self.assertNotIn("enable_lora", kw)
        self.assertIsNone(self.generate_calls[0][2])
        self.assertEqual(len(self.generate_calls[0][0]), 1)

    def test_refusals(self):
        self.meta["lora"] = None
        full = self.write(self.step_dir(lora=False))
        with self.assertRaisesRegex(SystemExit, "full_weights cut"):
            self.run_main("--adapter", str(full))
        (self.tmp / "gen-out" / "gen" / "x.json").write_text("{}")
        with self.assertRaisesRegex(SystemExit, "not empty"):
            self.run_main()


# ------------------------------------------------------------------------------------------------ launcher ---
class TestLauncher(unittest.TestCase):
    """run_agentic_nemo.main with stand-ins for the NeMo RL registries and the example entry point."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.ex = FakeExecutor(self)
        self.out = self.tmp / "out"
        self.base = V.build_base(self.tmp / "models" / "qwen3-1.7b")
        train = self.tmp / "train.jsonl"
        train.write_text("".join(json.dumps(train_row(k, it)) + "\n" for k, it in enumerate(items(8))))
        old = run_agentic_nemo.DATA
        run_agentic_nemo.DATA = train
        self.addCleanup(setattr, run_agentic_nemo, "DATA", old)
        self.registered, self.envs, self.actors, self.argv = {}, {}, {}, []
        entry = module("run_grpo", __file__="/opt/nemo-rl/examples/run_grpo.py",
                       main=lambda: self.argv.append(list(sys.argv)))
        stub_modules(self, {
            "nemo_rl.data.processors": module("nemo_rl.data.processors",
                                              register_processor=lambda n, f: self.registered.__setitem__(n, f)),
            "nemo_rl.distributed.ray_actor_environment_registry": module(
                "nemo_rl.distributed.ray_actor_environment_registry", ACTOR_ENVIRONMENT_REGISTRY=self.actors),
            "nemo_rl.distributed.virtual_cluster": module(
                "nemo_rl.distributed.virtual_cluster", PY_EXECUTABLES=types.SimpleNamespace(SYSTEM="system-python")),
            "nemo_rl.environments.utils": module("nemo_rl.environments.utils",
                                                 register_env=lambda n, fqn: self.envs.__setitem__(n, fqn)),
            "nemo_rl.utils.checkpoint": module("nemo_rl.utils.checkpoint", CheckpointManager=type(
                "CheckpointManager", (), {"__init__": lambda self, config: None})),
            "run_grpo": entry})
        saved_argv, saved_path, saved_pp = list(sys.argv), list(sys.path), os.environ.get("PYTHONPATH")
        self.addCleanup(setattr, sys, "argv", saved_argv)
        self.addCleanup(setattr, sys, "path", saved_path)
        self.addCleanup(lambda: os.environ.__setitem__("PYTHONPATH", saved_pp) if saved_pp is not None
                        else os.environ.pop("PYTHONPATH", None))

    def main(self, *extra: str) -> int:
        return run_agentic_nemo.main(["--profile", "smoke", "--out", str(self.out), "--seed", "10100", "--base",
                                      str(self.base), "--memory", "a", "--socket", self.ex.path, *extra])

    def test_wiring_and_overrides(self):
        self.assertEqual(self.main("--max-steps", "2"), 0)
        self.assertEqual(self.registered, {"math_code_rowid": math_code_env.math_code_rowid_processor})
        self.assertEqual(self.envs, {"math_code": "math_code_env.MathCodeEnvironment"})
        self.assertTrue(hasattr(math_code_env, "MathCodeEnvironment"))
        self.assertEqual(self.actors, {"math_code_env.MathCodeEnvironment": "system-python"})
        self.assertTrue({str(RUNTIME), str(RUNTIME / "executor")} <= set(os.environ["PYTHONPATH"].split(":")))
        [argv] = self.argv
        self.assertEqual(argv[1:3], ["--config", "/opt/nemo-rl/examples/configs/grpo_math_1B.yaml"])
        ov = dict(o.split("=", 1) for o in argv[3:])
        self.assertEqual(ov["data.default.processor"], "math_code_rowid")
        self.assertEqual(ov["++env.math_code.socket"], self.ex.path)
        self.assertEqual(ov["++env.math_code.log_dir"], str(self.out))
        self.assertEqual(ov["grpo.max_num_steps"], "2")
        self.assertEqual(ov["checkpointing.save_period"], "2")
        self.assertEqual(ov["data.default.prompt_file"], str(gen_agentic.PROMPT_FILE))
        self.assertEqual(ov["policy.max_total_sequence_length"], str(gen_agentic.SAMPLER["max_model_len"]))
        self.assertEqual(ov["policy.generation.max_new_tokens"], str(gen_agentic.SAMPLER["max_new_tokens_per_turn"]))
        self.assertEqual(json.loads(ov["policy.generation.stop_strings"]), gen_agentic.SAMPLER["stop"])
        self.assertEqual(self.ex.jobs(), [{"id": "launcher-check", "cells": ["print(6 * 7)"]}])
        args = json.loads((self.out / "run_args.json").read_text())
        self.assertEqual((args["max_steps"], args["lora"], args["train_file_sha256"]),
                         (2, {"dim": 32, "alpha": 64}, hooks.sha256_file(run_agentic_nemo.DATA)))
        self.assertEqual(args["code_sha256"]["math_code_env.py"], hooks.sha256_file(RUNTIME / "math_code_env.py"))
        self.assertEqual(args["executor"]["ping"]["ok"], True)
        self.assertEqual([x["event"] for x in read_jsonl(self.out / "attempts.jsonl")], ["start", "end"])

    def test_failed_self_check_stops_before_the_trainer(self):
        self.ex.self_check = "41\n"
        with self.assertRaisesRegex(SystemExit, "self-check failed"):
            self.main()
        self.assertEqual(self.argv, [])


class TestPromptAndLength(unittest.TestCase):
    def test_longest_prompt_and_a_turn_fit(self):
        # make_data.py measured the chat-templated prompt tokens with the base tokenizer, but with the P1 prompt file
        # (make_data.PROMPT_FILE), not the deployment prompt. The deployment prompt adds at most its own length in
        # tokens (a BPE token is at least one byte), so the longest prompt plus one full turn still fits.
        import make_data
        meta = json.loads((RUNTIME / "data" / "data_meta.json").read_text())
        self.assertEqual(meta["prompt_file_sha256"], hooks.sha256_file(make_data.PROMPT_FILE))
        measured = max(meta["prompt_tokens"]["train"]["max"], meta["prompt_tokens"]["items-test"]["max"])
        bound = measured + len(gen_agentic.PROMPT_FILE.read_bytes())
        self.assertLessEqual(1557, bound)          # README "Prompt file": the deployment prompt's measured maximum
        self.assertLessEqual(bound + gen_agentic.SAMPLER["max_new_tokens_per_turn"],
                             gen_agentic.SAMPLER["max_model_len"])
        for rung in ("a", "b"):
            self.assertEqual(run_agentic_nemo.MEMORY[rung]["max_seq"], gen_agentic.SAMPLER["max_model_len"])

    def test_every_problem_formats(self):
        tmpl = run_agentic_nemo.PROMPT.read_text(encoding="utf-8")
        for it in items():
            text = tmpl.format(it["problem"])
            self.assertTrue(text.endswith(it["problem"]))
            self.assertNotIn("{{", text.replace(it["problem"], ""))


if __name__ == "__main__":
    unittest.main()
