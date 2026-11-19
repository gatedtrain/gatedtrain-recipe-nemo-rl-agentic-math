"""score_agentic.py: parser self-test and an end-to-end arm on a synthetic generation directory (host CPU)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from _support import AREA, read_jsonl

import score_agentic


def gen_doc(item, final, end="final", tool="ok", error=None):
    turns = [{"text": "<code>print(1)</code>", "n_tokens": 6, "finish_reason": "stop", "code": "print(1)",
              "tool_status": tool, "observation": "<result>\n1\n</result>\n", "obs_tokens": 7},
             {"text": final, "n_tokens": 5, "finish_reason": "stop", "code": None, "tool_status": None,
              "observation": "", "obs_tokens": 0}]
    return {"item": item, "seed": 11, "repeat": 0, "prompt_sha256": "sha256:x", "prompt_tokens": 50, "turns": turns,
            "end_reason": end, "final_text": final, "model_tokens": 11, "env_tokens": 7, "total_tokens": 68,
            "error": error}


class Scorer(unittest.TestCase):
    def test_self_test(self):
        self.assertEqual(score_agentic.self_test(), 0)

    def test_arm(self):
        tmp = Path(tempfile.mkdtemp(prefix="a10-score-"))
        try:
            items = [{"item": f"math-test-{k:04d}", "text": f"p{k}", "problem": f"p{k}", "answer": str(k),
                      "subject": "algebra", "level": "Level 1", "source": {}} for k in range(6)]
            (tmp / "items").mkdir()
            (tmp / "items" / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
            gen = tmp / "gen"
            (gen / "gen").mkdir(parents=True)
            docs = {0: gen_doc("math-test-0000", "\\boxed{0}"),                          # correct
                    1: gen_doc("math-test-0001", "\\boxed{7}", tool="timeout"),          # wrong
                    2: gen_doc("math-test-0002", "I give up"),                           # no answer
                    3: gen_doc("math-test-0003", "<code>x</code>", end="max_turns"),     # max turns
                    4: gen_doc("math-test-0004", "", error={"class": "sandbox_failure", "message": "down"})}
            for k, d in docs.items():                                                    # item 5: missing output
                (gen / "gen" / f"{d['item']}__s11.json").write_text(json.dumps(d))
            meta = {"backend": "vllm", "env_id": "math_code", "model_dir": "/m", "sampler": {"strategy": "greedy"},
                    "eval_seeds": [11], "items": 6, "item_ids": [i["item"] for i in items],
                    "items_sha256": score_agentic.sha256_file(tmp / "items" / "items.jsonl"),
                    "prompt_file_sha256": "sha256:p", "executor": {"code_sha256": {}}, "versions": {"vllm": "0.29.0"}}
            (gen / "gen_meta.json").write_text(json.dumps(meta))
            out = tmp / "score"
            r = subprocess.run([sys.executable, "-I", str(AREA / "score_agentic.py"), "--gen", str(gen), "--items",
                                str(tmp / "items"), "--out", str(out)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            rows = read_jsonl(out / "scores.jsonl")
            self.assertEqual([x["score"] for x in rows], [1.0, 0.0, 0.0, 0.0, None, None])
            self.assertEqual([x["outcome"] for x in rows], ["correct", "wrong_answer", "no_answer", "max_turns", None,
                                                            None])
            self.assertEqual([x["error"]["class"] if x["error"] else None for x in rows],
                             [None, None, None, None, "sandbox_failure", "missing_output"])
            self.assertEqual(rows[1]["sub_metrics"]["tool_timeouts"], 1)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["exact_match"]["mean"], 0.25)
            self.assertTrue(summary["degeneracy_mm38"]["exact_match"]["pass"])
            arm = read_jsonl(out / "arm-exact_match" / "items.jsonl")
            self.assertEqual(len(arm), 6)
            em = json.loads((out / "arm-exact_match" / "eval_meta.json").read_text())
            self.assertEqual((em["evaluator_id"], em["metric"], em["adapter"]),
                             ("agentic.math_code_items@0.1", "exact_match", "nemo_rl_agentic"))
            for x in arm:
                self.assertTrue(all(isinstance(v, (int, float)) for v in (x["sub_metrics"] or {}).values()))
            # a generation of another environment or another item selection is refused
            meta2 = dict(meta, env_id="sliding_puzzle")
            (gen / "gen_meta.json").write_text(json.dumps(meta2))
            r2 = subprocess.run([sys.executable, "-I", str(AREA / "score_agentic.py"), "--gen", str(gen), "--items",
                                 str(tmp / "items"), "--out", str(tmp / "score2")], capture_output=True, text=True)
            self.assertNotEqual(r2.returncode, 0)
            self.assertIn("env_id", r2.stderr)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_limit_selection_matches_generator(self):
        import gen_agentic
        items = [{"item": str(i)} for i in range(500)]
        for n in (1, 20, 100, 499, 500, 600):
            self.assertEqual(score_agentic.select_items(items, n), gen_agentic.select_items(items, n))
        self.assertEqual(len({i["item"] for i in score_agentic.select_items(items, 20)}), 20)


if __name__ == "__main__":
    unittest.main()
