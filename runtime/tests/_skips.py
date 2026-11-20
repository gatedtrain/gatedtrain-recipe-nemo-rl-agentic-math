"""Skip conditions of area "agentic RL"'s CPU tests for the heavy inputs a fresh clone lacks: the base model's
tokenizer files (fetch_models.py), the training data (make_data.py), the NeMo RL clone (`GATEDTRAIN_NEMO_RL_DIR`),
and torch. Tests that need one skip with the reason (skip_unless, require). The repository's tests/test_runtime.py
covers the hooks, the environment, the trajectory log and the generation loop without any of them, in CI.

Strict mode, `GATEDTRAIN_RUNTIME_TESTS_STRICT=1` (the GPU host, where every heavy input is present): each of these
skips is a failure naming the input instead, and the host-executor opt-out below is ignored.

numpy and sympy in the interpreter the executor runs model code with: the sidecar's NeMo RL image has both, so
test_executor's check of them fails without them; a host venv may lack them, and
`GATEDTRAIN_RUNTIME_TESTS_HOST_EXECUTOR=1` skips that one test there.

Kept apart from _support.py, whose SHA-256 the sliding-puzzle fixture records (tests/fixtures/agentic/sliding-puzzle/
SOURCE.json).
"""

from __future__ import annotations

import functools
import importlib.util
import os
import unittest

from _support import AREA, NEMO_RL

BASE = AREA / "models" / "qwen3-1.7b"
TRAIN = AREA / "data" / "train.jsonl"
HAVE_BASE_TOKENIZER = all((BASE / n).is_file() for n in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                                                          "merges.txt"))
HAVE_TRAIN = TRAIN.is_file()
HAVE_NEMO_RL = (NEMO_RL / "nemo_rl/environments/games/sliding_puzzle.py").is_file()
HAVE_TORCH = importlib.util.find_spec("torch") is not None
NO_BASE = f"base model tokenizer absent ({BASE}; fetch_models.py)"
NO_TRAIN = f"training data absent ({TRAIN}; make_data.py)"
NO_NEMO_RL = f"NeMo RL clone absent ({NEMO_RL}; set GATEDTRAIN_NEMO_RL_DIR)"
NO_TORCH = "torch not installed in this venv"


STRICT_ENV = "GATEDTRAIN_RUNTIME_TESTS_STRICT"
STRICT = os.environ.get(STRICT_ENV) == "1"
HOST_EXECUTOR_ENV = "GATEDTRAIN_RUNTIME_TESTS_HOST_EXECUTOR"
HOST_EXECUTOR = os.environ.get(HOST_EXECUTOR_ENV) == "1" and not STRICT
NO_EXECUTOR_MODULES = (f"{HOST_EXECUTOR_ENV}=1: the executor runs this host venv's Python, which may lack numpy and "
                       "sympy (the sidecar's NeMo RL image has both)")


def _strict_message(reason: str) -> str:
    return f"{reason} ({STRICT_ENV}=1: a heavy input must be present)"


def skip_unless(have: bool, reason: str):
    """unittest.skipUnless(have, reason) for a test method or class; in strict mode a missing input fails the test
    (a class: its setUpClass) with the reason instead."""
    if have or not STRICT:
        return unittest.skipUnless(have, reason)

    def decorate(obj):
        if isinstance(obj, type):
            def setUpClass(cls):
                raise AssertionError(_strict_message(reason))
            obj.setUpClass = classmethod(setUpClass)
            return obj

        @functools.wraps(obj)
        def fail(self, *args, **kwargs):
            self.fail(_strict_message(reason))
        return fail
    return decorate


def require(test: unittest.TestCase, have: bool, reason: str) -> None:
    """test.skipTest(reason) when `have` is false, inside a test; in strict mode test.fail with the reason instead."""
    if not have:
        if STRICT:
            test.fail(_strict_message(reason))
        test.skipTest(reason)
