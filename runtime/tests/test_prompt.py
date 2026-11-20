"""The deployment prompt `prompt_math_code_oneshot.txt` (README 0.11): byte-identical, once formatted, to the
`text_oneshot` variant probe_tool_use.py ran in probe-tool-1/2; used by the trainer and the evaluator; its worked
example is a true exchange of this area's own protocol (real in-process executor); prompt lengths leave room for the
turns within the 4,096-token context; and the same scripted turns end with the same end reason in the training
loop (env_step + a NeMo RL rollout double + consumption) and in the evaluator's loop. Host CPU."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

from _skips import HAVE_BASE_TOKENIZER, HAVE_TRAIN, NO_BASE, NO_TRAIN, TRAIN
from _support import AREA, OUTPUTS, fake_rollout, ntok, read_jsonl

import agentic_core
import consumption
import executor_server
import gen_agentic
import math_code_env
import probe_tool_use as probe
import run_agentic_nemo as launcher
import score_agentic
import traj_log

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None

PROMPT = AREA / "prompt_math_code_oneshot.txt"
PROMPT_SHA256 = "sha256:3b7145c1edf2526dcd13d5477d343d68e883b30c8e20d1a5c5e3f19af7b6230f"
PROBE2 = OUTPUTS / "probe-tool-2" / "probe"


def problems() -> list[str]:
    """The training problems (when make_data.py has written them) and the committed test items' problems."""
    return [r["input"] for r in (read_jsonl(TRAIN) if HAVE_TRAIN else [])] + \
        [i["problem"] for i in read_jsonl(AREA / "items-test" / "items.jsonl")]


class PromptFile(unittest.TestCase):
    def test_identity_and_use(self):
        self.assertEqual("sha256:" + hashlib.sha256(PROMPT.read_bytes()).hexdigest(), PROMPT_SHA256)
        self.assertEqual(gen_agentic.PROMPT_FILE, PROMPT)
        self.assertEqual(launcher.PROMPT, PROMPT)
        self.assertEqual((gen_agentic.SAMPLER["prompt_file"], gen_agentic.SAMPLER["prompt_file_sha256"]),
                         (PROMPT.name, PROMPT_SHA256))
        self.assertEqual((gen_agentic.SAMPLER["max_new_tokens_per_turn"], gen_agentic.SAMPLER["max_model_len"]),
                         (1024, 4096))

    def test_formats_to_the_probed_text(self):
        tmpl = PROMPT.read_text(encoding="utf-8")
        for p in problems() + ["{}", "{problem}", "a {{x}} b", "\\frac{1}{2}", ""]:
            self.assertEqual(tmpl.format(p), probe.TEXT_ONESHOT.replace("{problem}", p), p[:60])

    def test_worked_example_runs_as_shown(self):
        """The example's code, run by MathCodeCore through the real executor, gives exactly the `<result>` block the
        prompt shows, and the example's answer turn is a final, boxed, correct answer."""
        text = PROMPT.read_text(encoding="utf-8").format("X")
        example = text[text.index("Example:\n") + len("Example:\n"):text.index("\n\nNow solve")]
        call, rest = example.split("</code>\n", 1)
        call += "</code>"
        shown_result, answer = rest[:rest.index("</result>\n") + len("</result>\n")], rest.split("</result>\n", 1)[1]
        tmp = tempfile.mkdtemp(prefix="a10-prompt-")
        ex = executor_server.Executor(workers=1, tmp_root=tmp)
        try:
            core = agentic_core.MathCodeCore(execute=ex.run_jobs,
                                             verifier=lambda t, g: float(score_agentic.parse_answer(t) == int(g)))
            r1 = core.step([{"turn": 1, "text": call, "state": {"cells": [], "ground_truth": "263"}}])[0]
            self.assertEqual((r1["terminated"], r1["tool_status"], r1["code"]), (False, "ok", probe.EXAMPLE_CODE))
            self.assertEqual(r1["observation"], shown_result)
            r2 = core.step([{"turn": 2, "text": answer, "state": r1["state"]}])[0]
            self.assertEqual((r2["terminated"], r2["end_reason"], r2["reward"]), (True, "final", 1.0))
        finally:
            ex.pool.shutdown()
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipIf(AutoTokenizer is None, "transformers not installed in this venv")
@unittest.skipUnless(HAVE_BASE_TOKENIZER, NO_BASE)
class PromptTokens(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(str(AREA / "models" / "qwen3-1.7b"), local_files_only=True)
        cls.tmpl = PROMPT.read_text(encoding="utf-8")

    def render(self, problem: str) -> str:
        """As gen_agentic.main and NeMo RL's math_hf_data_processor render one user turn."""
        return self.tok.apply_chat_template([{"role": "user", "content": self.tmpl.format(problem)}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)

    @unittest.skipUnless(HAVE_TRAIN, NO_TRAIN)
    def test_lengths_fit_the_context(self):
        lens = sorted(len(self.tok(self.render(p), add_special_tokens=False)["input_ids"]) for p in problems())
        self.assertEqual(lens[-1], 1557)                         # longest training prompt; items at most 856
        # the longest prompt still leaves two full 1,024-token turns plus a short third; the median (about 310)
        # leaves 3.7 full turns; tool-call turns stop at </code> and are short (probe-tool-2 max total 1,527)
        self.assertGreaterEqual(4096 - lens[-1], 2 * 1024 + 256)

    @unittest.skipUnless((PROBE2 / "text_oneshot.jsonl").exists(), "probe-tool-2 output not present")
    def test_equals_probe_tool_2(self):
        train = {r["row_id"]: r["input"] for r in read_jsonl(AREA / "data" / "train.jsonl")}
        seen = {d["row_id"]: d["prompt_tokens"] for d in read_jsonl(PROBE2 / "text_oneshot.jsonl")}
        self.assertEqual(len(seen), 64)
        for rid, n in seen.items():
            s = self.render(train[rid])
            self.assertEqual(s, probe.render_prompt(self.tok, "text_oneshot", train[rid]))
            self.assertEqual(len(self.tok(s, add_special_tokens=False)["input_ids"]), n, rid)


class EndReasonParity(unittest.TestCase):
    """The same scripted turns through the training path (math_code_env.env_step under the NeMo RL rollout double,
    end reasons as consumption.build derives them) and through gen_agentic.run_trajectories, with the same
    whitespace token counts and context: identical end reasons and turn counts."""

    SCRIPTS = {
        "tool_then_final": ["calc <code> print(6 * 7) </code>", "so \\boxed{42}"],
        "final": ["it is \\boxed{5}"],
        "no_answer": ["prose without an answer"],
        "max_turns": ["<code> 1/0 </code>"] * 4,
        "overflow": ["<code> print(' '.join(['w'] * 300)) </code>"] * 4,
        "text_after_code": ["<code> print(1) </code> hence \\boxed{1}"],
        "unclosed_code": ["<code> while True: pass"],
    }

    def test_same_end_reasons(self):
        names = list(self.SCRIPTS)
        max_seq = 160
        tmp = Path(tempfile.mkdtemp(prefix="a10-parity-"))
        ex = executor_server.Executor(workers=2, tmp_root=str(tmp))
        try:
            core = agentic_core.MathCodeCore(execute=ex.run_jobs, verifier=lambda t, g: 0.0, max_turns=4)
            tlog = traj_log.TrajectoryLogger(str(tmp), "math_code")
            prompt = "problem what is six times seven"
            lines = fake_rollout([prompt] * len(names), [{"ground_truth": "42", "row_id": "r0"}] * len(names),
                                 lambda i, turn, log: self.SCRIPTS[names[i]][turn],
                                 lambda m, md: math_code_env.env_step(core, tlog, m, md), max_turns=4, max_seq=max_seq)
            train = consumption.build(read_jsonl(tmp / "env_calls.jsonl"), {1: lines}, None, 1, len(names),
                                      max_steps=1, max_turns=4, run="t", attempt=1)
            want = {n: (r["end_reason"], r["turns"]) for n, r in zip(names, sorted(train, key=lambda r: r["traj_pos"]))}

            states = [{"item": n, "ids": [0] * ntok(prompt), "turns": [], "cells": [], "done": False,
                       "end_reason": None, "error": None} for n in names]

            def generate(id_lists, max_tokens):
                out = []
                for s, m in zip([s for s in states if not s["done"]], max_tokens):
                    words = self.SCRIPTS[s["item"]][len(s["turns"])].split()[:m]
                    out.append(([1] * len(words), " ".join(words), "stop"))
                return out

            ecore = agentic_core.MathCodeCore(execute=ex.run_jobs, max_turns=4)
            gen_agentic.run_trajectories(states, generate, lambda t: [2] * ntok(t), ecore, max_model_len=max_seq)
            got = {s["item"]: (s["end_reason"], len(s["turns"])) for s in states}
            self.assertEqual(got, want)
            self.assertEqual({n: e for n, (e, _) in got.items()},
                             {"tool_then_final": "final", "final": "final", "no_answer": "no_answer",
                              "max_turns": "max_turns", "overflow": "overflow", "text_after_code": "final",
                              "unclosed_code": "no_answer"})
        finally:
            ex.pool.shutdown()
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
