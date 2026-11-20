"""probe_tool_use.py on the host CPU: native tool-call parsing, NativeCore against the real in-process executor, the
turn loop with a scripted 'model' for both protocols, and (when transformers is installed, area-4 venv) the chat
template: every variant renders, and the native observation continues the token history exactly as the template
renders a tool message."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest

from _skips import HAVE_BASE_TOKENIZER, NO_BASE, skip_unless
from _support import AREA

import agentic_core
import executor_server
import probe_tool_use as probe

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None


def call(code: str) -> str:
    return "<tool_call>\n" + json.dumps({"name": "python", "arguments": {"code": code}}) + "\n</tool_call>"


class ParseToolCalls(unittest.TestCase):
    def test_cases(self):
        self.assertIsNone(probe.parse_tool_calls("The answer is \\boxed{3}."))
        self.assertIsNone(probe.parse_tool_calls(call("1") + " so \\boxed{1}"))          # text after: terminal
        self.assertEqual(probe.parse_tool_calls("Plan.\n" + call("  x = 1\n  x")), [{"code": "x = 1\nx"}])
        two = probe.parse_tool_calls(call("a = 2") + "\n" + call("a * 3"))
        self.assertEqual([c["code"] for c in two], ["a = 2", "a * 3"])
        s = '<tool_call>\n{"name": "python", "arguments": "{\\"code\\": \\"print(5)\\"}"}\n</tool_call>'
        self.assertEqual(probe.parse_tool_calls(s), [{"code": "print(5)"}])            # arguments as a JSON string
        bad = probe.parse_tool_calls("<tool_call>\n{not json}\n</tool_call>")
        self.assertIsNone(bad[0]["code"])
        self.assertIsNone(probe.parse_tool_calls('<tool_call>{"name": "search", "arguments": {}}</tool_call>')[0]
                          ["code"])

    def test_observation_text(self):
        obs = probe.native_observation(["42", "(no output)"])
        self.assertEqual(obs, "\n<|im_start|>user\n<tool_response>\n42\n</tool_response>\n<tool_response>\n"
                              "(no output)\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        self.assertEqual(probe.tool_response_text({"status": "ok", "output": "7\n"}, 512), "7")


class WithExecutor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="a10-probe-")
        self.ex = executor_server.Executor(workers=2, tmp_root=self.tmp)

    def tearDown(self):
        self.ex.pool.shutdown()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_worked_example_is_true(self):
        res = self.ex.run_jobs([{"id": "x", "cells": [probe.EXAMPLE_CODE]}])
        self.assertEqual(res[0]["output"].strip(), probe.EXAMPLE_RESULT)

    def test_native_core(self):
        core = probe.NativeCore(self.ex.run_jobs, max_turns=4)
        res = core.step([
            {"turn": 1, "text": "x" + call("a = 6") + "\n" + call("a * 7"), "state": {"cells": []}},
            {"turn": 1, "text": "So \\boxed{4}.", "state": {"cells": []}},
            {"turn": 1, "text": "Hmm, let me think", "state": {"cells": []}},
            {"turn": 4, "text": call("print(1)"), "state": {"cells": []}},
            {"turn": 2, "text": "<tool_call>{oops}</tool_call>", "state": {"cells": []}},
        ])
        self.assertEqual([r["terminated"] for r in res], [False, True, True, True, False])
        self.assertEqual([r["end_reason"] for r in res], [None, "final", "no_answer", "max_turns", None])
        self.assertEqual(res[0]["tool_statuses"], ["ok", "ok"])
        self.assertIn("<tool_response>\n42\n</tool_response>", res[0]["observation"])  # the second call replays a
        self.assertEqual([c["code"] for c in res[0]["state"]["cells"]], ["a = 6", "a * 7"])
        self.assertEqual(res[4]["tool_statuses"], ["parse_error"])
        self.assertIn("could not be parsed", res[4]["observation"])
        self.assertTrue(res[0]["observation"].endswith(probe.NATIVE_NEXT))

    def test_loop_both_protocols(self):
        scripts = {"text": {"a": ["Plan <code>print(6*7)</code>", "So \\boxed{42}"], "b": ["prose only"],
                            "c": ["<code>1/0</code>"] * 4},
                   "native": {"a": [call("print(6*7)"), "So \\boxed{42}"], "b": ["prose only"],
                              "c": [call("1/0")] * 4}}
        for proto, script in scripts.items():
            core = probe.NativeCore(self.ex.run_jobs) if proto == "native" else \
                agentic_core.MathCodeCore(execute=self.ex.run_jobs)
            states = [{"row_id": k, "answer": "42", "sampler": "greedy", "sample": 0, "seed": 0, "prompt_tokens": 5,
                       "ids": [0] * 5, "turns": [], "cells": [], "done": False, "end_reason": None, "error": None}
                      for k in script]

            def generate(active, max_tokens):
                out = []
                for s, m in zip(active, max_tokens):
                    text = script[s["row_id"]][len(s["turns"])]
                    out.append(([1] * min(len(text), m), text, "stop"))
                return out

            probe.run_trajectories(states, generate, lambda t: [2] * len(t), core, max_model_len=2000,
                                   max_new_tokens=512)
            variant = "native" if proto == "native" else "text_base"
            docs = {s["row_id"]: probe.trajectory_doc(s, variant) for s in states}
            self.assertEqual((docs["a"]["correct"], docs["a"]["tool_calls_executed"], docs["a"]["end_reason"]),
                             (1, 1, "final"), proto)
            self.assertEqual((docs["b"]["tool_calls_attempted"], docs["b"]["end_reason"]), (0, "no_answer"), proto)
            self.assertEqual((docs["c"]["tool_calls_attempted"], docs["c"]["tool_calls_executed"],
                              docs["c"]["end_reason"]), (4, 3, "max_turns"), proto)
            summ = probe.summarize(list(docs.values()))
            self.assertEqual(summ["tool_share"], round(2 / 3, 4))
            self.assertEqual(summ["exact_match"], round(1 / 3, 4))
            self.assertEqual(summ["tool_statuses"]["error"], 3)


@unittest.skipIf(AutoTokenizer is None, "transformers not installed in this venv")
@skip_unless(HAVE_BASE_TOKENIZER, NO_BASE)
class ChatTemplate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = AutoTokenizer.from_pretrained(str(AREA / "models" / "qwen3-1.7b"), local_files_only=True)

    def test_variants_render(self):
        problem = "What is $1+1$?"
        for v in probe.VARIANTS:
            text = probe.render_prompt(self.tok, v, problem)
            self.assertTrue(text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), v)
            self.assertIn(problem, text, v)
            if v.startswith("native"):
                self.assertIn('"name": "python"', text)
                self.assertIn("<tool_call>", text)
        self.assertEqual(probe.render_prompt(self.tok, "text_base", problem),
                         self.tok.apply_chat_template([{"role": "user", "content": probe.TRAIN_PROMPT.format(problem)}],
                                                      tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=False))
        few = probe.render_prompt(self.tok, "native_fewshot", problem)
        self.assertIn("<tool_response>\n263\n</tool_response>", few)

    def test_native_observation_continues_like_the_template(self):
        """prompt + generated reply + <|im_end|> + observation == the template's rendering of the same conversation
        with a tool message, except the empty think block of the earlier assistant turn, which the template drops
        for non-final assistant turns while the token history keeps what the model was prompted with."""
        problem = "What is $6 \\times 7$?"
        msgs, tools = probe.messages_for("native", problem)
        prompt = probe.render_prompt(self.tok, "native", problem)
        reply = "I will compute it.\n" + call("6 * 7")
        history = prompt + reply + "<|im_end|>" + probe.native_observation(["42"])
        conv = msgs + [{"role": "assistant", "content": "I will compute it.",
                        "tool_calls": [{"type": "function", "function": {"name": "python",
                                                                         "arguments": {"code": "6 * 7"}}}]},
                       {"role": "tool", "content": "42"}]
        rendered = self.tok.apply_chat_template(conv, tools=tools, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        think = "<think>\n\n</think>\n\n"
        self.assertEqual(history.replace("<|im_start|>assistant\n" + think + reply,
                                         "<|im_start|>assistant\n" + reply), rendered)
        # the observation's special-token strings become the special tokens (NeMo RL tokenizes observations so)
        ids = self.tok(probe.native_observation(["42"]), add_special_tokens=False)["input_ids"]
        self.assertEqual(ids.count(self.tok.convert_tokens_to_ids("<|im_start|>")), 2)
        self.assertEqual(ids.count(self.tok.convert_tokens_to_ids("<|im_end|>")), 1)
        self.assertIn(self.tok.convert_tokens_to_ids("<tool_response>"), ids)


if __name__ == "__main__":
    unittest.main()
