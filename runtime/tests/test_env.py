"""math_code environment: turn parsing, termination, observations, reward gating, the NeMo RL actor glue (with
stand-in ray/nemo_rl modules) and the evaluator's multi-turn loop against the real executor (host CPU)."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from _skips import HAVE_NEMO_RL, HAVE_TORCH, NO_NEMO_RL, NO_TORCH
from _support import StubVerifyWorker, install_nemo_stubs, load_fresh, read_jsonl

import agentic_core
import executor_server
import traj_log


class ExtractCode(unittest.TestCase):
    def test_cases(self):
        cases = [
            ("Let me compute.\n<code>print(1+1)</code>", "print(1+1)"),
            ("<code>\nx = 3\nx</code>\n", "x = 3\nx"),
            ("<code>```python\nprint(2)\n```</code>", "print(2)"),
            ("<code>a=1</code> then <code>b=2</code>", "b=2"),
            ("The answer is \\boxed{4}.", None),
            ("<code>print(1)</code> so \\boxed{1}", None),       # text after the last </code>: final
            ("no open tag</code>", None),
            ("<code>while True:\n  pass", None),                   # budget hit inside code: no_answer
            ("<code></code>", ""),
        ]
        for text, want in cases:
            self.assertEqual(agentic_core.extract_code(text), want, text)

    def test_observation_format_and_cut(self):
        self.assertEqual(agentic_core.observation({"status": "ok", "output": "42\n"}), "<result>\n42\n</result>\n")
        self.assertEqual(agentic_core.observation({"status": "ok", "output": ""}), "<result>\n(no output)\n</result>\n")
        self.assertIn("TimeoutError", agentic_core.observation({"status": "timeout", "output": ""}))
        self.assertIn("crashed", agentic_core.observation({"status": "crash", "output": ""}))
        big = agentic_core.observation({"status": "ok", "output": "y" * 5000, "output_chars": 9000})
        body = big[len("<result>\n"):-len("\n</result>\n")]
        self.assertLessEqual(len(body), agentic_core.OBS_MAX_CHARS + 60)
        self.assertIn("characters omitted", body)
        self.assertTrue(body.endswith("of 9000 characters]"))


class Core(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="a10-env-")
        cls.ex = executor_server.Executor(workers=4, tmp_root=cls.tmp)

    @classmethod
    def tearDownClass(cls):
        cls.ex.pool.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def core(self, verifier=None, max_turns=4):
        self.jobs = []

        def execute(jobs):
            self.jobs.append(jobs)
            return self.ex.run_jobs(jobs)
        return agentic_core.MathCodeCore(execute=execute, verifier=verifier, max_turns=max_turns)

    def test_turn_decisions_and_reward_gate(self):
        v = StubVerifyWorker()
        core = self.core(verifier=lambda t, g: v.verify([t], [g], math_verify_impl="hf_math_verify")[0])
        st = {"cells": [], "ground_truth": "42", "row_id": "r"}
        res = core.step([
            {"turn": 1, "text": "<code>x = 6 * 7\nx</code>", "state": st},
            {"turn": 1, "text": "\\boxed{42}", "state": st},
            {"turn": 1, "text": "42", "state": st},                              # right number, not boxed: 0
            {"turn": 4, "text": "<code>print(1)</code>", "state": st},           # tool call at the turn limit
            {"turn": 2, "text": "<code>def f(:</code>", "state": st},
        ])
        self.assertEqual([r["terminated"] for r in res], [False, True, True, True, False])
        self.assertEqual([r["end_reason"] for r in res], [None, "final", "no_answer", "max_turns", None])
        self.assertEqual([r["reward"] for r in res], [None, 1.0, 0.0, 0.0, None])
        self.assertEqual(res[0]["observation"], "<result>\n42\n</result>\n")
        self.assertEqual([r["tool_status"] for r in res], ["ok", None, None, None, "syntax_error"])
        self.assertEqual(len(self.jobs), 1)                                     # one executor batch per call
        self.assertEqual(len(self.jobs[0]), 2)
        self.assertEqual(res[0]["state"]["cells"], [{"code": "x = 6 * 7\nx", "status": "ok"}])
        self.assertEqual(res[0]["state"]["ground_truth"], "42")

    def test_stateless_replay_skips_failed_cells(self):
        core = self.core()
        cells = [{"code": "a = 5", "status": "ok"}, {"code": "def f(:", "status": "syntax_error"},
                 {"code": "while True: pass", "status": "timeout"}, {"code": "b = a + 1\n1/0", "status": "error"}]
        res = core.step([{"turn": 3, "text": "<code>print(a, b)</code>", "state": {"cells": cells}}])
        self.assertEqual(self.jobs[0][0]["cells"], ["a = 5", "b = a + 1\n1/0", "print(a, b)"])
        self.assertEqual(res[0]["observation"], "<result>\n5 6\n</result>\n")
        self.assertEqual(len(res[0]["state"]["cells"]), 5)

    def test_eval_core_has_no_reward(self):
        res = self.core().step([{"turn": 1, "text": "\\boxed{3}", "state": {"cells": []}}])
        self.assertEqual((res[0]["reward"], res[0]["end_reason"]), (0.0, "final"))


@unittest.skipUnless(HAVE_NEMO_RL, NO_NEMO_RL)
@unittest.skipUnless(HAVE_TORCH, NO_TORCH)
class ActorGlue(unittest.TestCase):
    """MathCodeEnvironment as NeMo RL would call it (stand-in ray / nemo_rl; real torch and executor)."""

    def setUp(self):
        install_nemo_stubs()
        self.mce = load_fresh("math_code_env.py", "math_code_env_stubbed")
        self.tmp = tempfile.mkdtemp(prefix="a10-actor-")
        self.ex = executor_server.Executor(workers=2, tmp_root=self.tmp)
        self.mce.exec_protocol = type("P", (), {"call_executor": staticmethod(
            lambda sock, jobs, timeout: self.ex.run_jobs(jobs))})

    def tearDown(self):
        self.ex.pool.shutdown()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_step_returns_environment_return_and_logs(self):
        env_cls = self.mce.MathCodeEnvironment
        self.assertIsNotNone(env_cls)
        env = env_cls({"socket": "/unused", "log_dir": self.tmp, "max_turns": 4})
        user = {"role": "user", "content": "prompt"}
        convs = [[user, {"role": "assistant", "content": "<code>print(2**10)</code>"}],
                 [user, {"role": "assistant", "content": "It is \\boxed{7}."}]]
        md = [{"ground_truth": "1024", "row_id": "r0"}, {"ground_truth": "7", "row_id": "r1"}]
        ret = env.step(convs, md)
        self.assertEqual(ret.observations[0], {"role": "environment", "content": "<result>\n1024\n</result>\n"})
        self.assertEqual(ret.observations[1]["content"], "")
        self.assertEqual(ret.rewards.tolist(), [0.0, 1.0])
        self.assertEqual(ret.terminateds.tolist(), [False, True])
        self.assertEqual(ret.next_stop_strings, [["</code>"], None])
        self.assertIsNone(ret.metadata[1])
        self.assertEqual(ret.metadata[0][traj_log.STAMP], {"traj_call": 1, "traj_pos": 0})
        self.assertEqual(ret.metadata[0]["row_id"], "r0")
        # turn 2 of the continuing trajectory: same stamp, reward through the HFVerifyWorker stand-in
        conv = convs[0] + [ret.observations[0], {"role": "assistant", "content": "So \\boxed{1024}"}]
        ret2 = env.step([conv], [ret.metadata[0]])
        self.assertEqual((ret2.rewards.tolist(), ret2.terminateds.tolist()), ([1.0], [True]))
        lines = read_jsonl(Path(self.tmp) / "env_calls.jsonl")
        self.assertEqual([(ln["call"], ln["traj_call"], ln["traj_pos"], ln["turn"]) for ln in lines],
                         [(1, 1, 0, 1), (1, 1, 1, 1), (2, 1, 0, 2)])
        self.assertEqual([ln["end_reason"] for ln in lines], [None, "final", "final"])
        self.assertEqual(lines[0]["tool_status"], "ok")
        self.assertEqual(len(read_jsonl(Path(self.tmp) / "trajectories.jsonl")), 2)
        self.assertEqual(env.global_post_process_and_metrics({"total_reward": ret.rewards})[1]["math_code_reward"], 0.5)


class EvalLoop(unittest.TestCase):
    """gen_agentic.run_trajectories with a scripted 'model' (token ids = characters) and the real executor."""

    def test_loop(self):
        import gen_agentic
        tmp = tempfile.mkdtemp(prefix="a10-gen-")
        ex = executor_server.Executor(workers=2, tmp_root=tmp)
        try:
            core = agentic_core.MathCodeCore(execute=ex.run_jobs)
            script = {"a": ["<code>print(6*7)</code>", "So \\boxed{42}"],
                      "b": ["<code>print('z' * 400)</code>"] * 4,
                      "c": ["<code>1/0</code>", "<code>print(1)</code>", "<code>2</code>", "<code>3</code>"],
                      "d": ["\\boxed{1}"]}
            states = [{"item": k, "ids": [0] * 10, "turns": [], "cells": [], "done": False, "end_reason": None,
                       "error": None} for k in script]

            def generate(id_lists, max_tokens):
                out = []
                for s, m in zip([s for s in states if not s["done"]], max_tokens):
                    text = script[s["item"]][len(s["turns"])][:m]
                    out.append(([1] * len(text), text, "stop"))
                return out

            gen_agentic.run_trajectories(states, generate, lambda t: [2] * len(t), core, max_model_len=1200)
            by = {s["item"]: s for s in states}
            self.assertEqual(by["a"]["end_reason"], "final")
            self.assertEqual(by["a"]["turns"][0]["observation"], "<result>\n42\n</result>\n")
            self.assertEqual(by["b"]["end_reason"], "overflow")
            self.assertEqual(len(by["b"]["ids"]), 1200)
            self.assertEqual(by["c"]["end_reason"], "max_turns")
            self.assertEqual([t["tool_status"] for t in by["c"]["turns"]], ["error", "ok", "ok", None])
            self.assertEqual(by["d"]["end_reason"], "final")
            self.assertEqual(len(by["d"]["turns"]), 1)
        finally:
            ex.pool.shutdown()
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
