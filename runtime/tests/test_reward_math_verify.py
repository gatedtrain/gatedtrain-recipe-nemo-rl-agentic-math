"""Training reward with the real math_verify library (skipped when it is not importable).

The suite runs in the area-4 venv, which has no math_verify; run this file additionally with a venv that has
math-verify 0.9.0 (the version in the NeMo RL image, env-freeze of area 7), e.g. the area-1 venv:
  <area-1 venv>/bin/python -m unittest discover -s tests -t tests -p test_reward_math_verify.py
It builds the same `math_metric` as NeMo RL's HFVerifyWorker (math_environment.py:89-100: gold LatexExtraction,
prediction ExprExtraction + LatexExtraction, ground truth wrapped as `\\boxed{gt}`, :131-135) and checks the reward
the environment gives through agentic_core's `\\boxed` gate.
"""

from __future__ import annotations

import unittest

from _support import AREA  # noqa: F401  (sys.path)

import agentic_core

try:
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:  # pragma: no cover
    math_metric = None


@unittest.skipIf(math_metric is None, "math_verify not installed in this venv")
class RewardMathVerify(unittest.TestCase):
    def setUp(self):
        f = math_metric(gold_extraction_target=(LatexExtractionConfig(),),
                        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()))

        def verifier(text, gt):
            try:
                return float(f(["\\boxed{" + gt + "}"], [text])[0])
            except Exception:  # noqa: BLE001 - as HFVerifyWorker: any failure scores 0
                return 0.0
        self.core = agentic_core.MathCodeCore(execute=lambda jobs: [], verifier=verifier)

    def reward(self, text, gt):
        return self.core.step([{"turn": 2, "text": text, "state": {"cells": [], "ground_truth": gt}}])[0]["reward"]

    def test_rewards(self):
        self.assertEqual(self.reward("Therefore the answer is $\\boxed{42}$.", "42"), 1.0)
        self.assertEqual(self.reward("\\boxed{1,000}", "1000"), 1.0)
        self.assertEqual(self.reward("\\boxed{41}", "42"), 0.0)
        self.assertEqual(self.reward("The answer is 42.", "42"), 0.0)          # gate: no \boxed, no reward
        self.assertEqual(self.reward("\\boxed{-3}", "-3"), 1.0)


if __name__ == "__main__":
    unittest.main()
