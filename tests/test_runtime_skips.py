"""runtime/tests/_skips.py's strict mode, in CI. runtime/tests skip a test whose heavy input (base tokenizer, training
data, NeMo RL clone, torch) is absent; on the GPU host every one is present, and there GATEDTRAIN_RUNTIME_TESTS_STRICT=1
turns each such skip into a failure that names the input, so a lost input cannot pass as "OK (skipped=...)". Here the
inputs are absent (a fresh clone; GATEDTRAIN_NEMO_RL_DIR points nowhere) and _skips is loaded fresh with and without
the variable.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

RUNTIME_TESTS = Path(__file__).resolve().parents[1] / "runtime" / "tests"
STRICT = "GATEDTRAIN_RUNTIME_TESTS_STRICT"
HOST_EXECUTOR = "GATEDTRAIN_RUNTIME_TESTS_HOST_EXECUTOR"


def load_skips(strict: str | None, host_executor: str | None = None):
    """A fresh runtime/tests/_skips.py (and the _support it imports) with the strict and host-executor variables set
    to `strict` and `host_executor` (None: unset) and no NeMo RL clone; sys.modules, sys.path and the environment are
    restored afterwards."""
    env = {"GATEDTRAIN_NEMO_RL_DIR": "/nonexistent/nemo-rl"}
    saved_path = list(sys.path)
    with mock.patch.dict(os.environ, env), mock.patch.dict(sys.modules):
        for name, value in ((STRICT, strict), (HOST_EXECUTOR, host_executor)):
            os.environ.pop(name, None)
            if value is not None:
                os.environ[name] = value
        for name in ("_skips", "_support"):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(RUNTIME_TESTS))
        try:
            spec = importlib.util.spec_from_file_location("_skips", RUNTIME_TESTS / "_skips.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        finally:
            sys.path[:] = saved_path
    return mod


def run(case: type) -> unittest.TestResult:
    result = unittest.TestResult()
    unittest.defaultTestLoader.loadTestsFromTestCase(case).run(result)
    return result


def cases(sk) -> dict:
    """The three ways runtime/tests skip on a heavy input, each on an absent one, and one on a present input."""

    class Method(unittest.TestCase):
        @sk.skip_unless(sk.HAVE_NEMO_RL, sk.NO_NEMO_RL)
        def test_needs_nemo_rl(self):
            pass

    @sk.skip_unless(sk.HAVE_NEMO_RL, sk.NO_NEMO_RL)
    class Class(unittest.TestCase):
        def test_one(self):
            pass

        def test_two(self):
            pass

    class InTest(unittest.TestCase):
        def test_needs_base(self):
            sk.require(self, False, sk.NO_BASE)

    class Present(unittest.TestCase):
        @sk.skip_unless(True, sk.NO_TORCH)
        def test_runs(self):
            sk.require(self, True, sk.NO_TRAIN)

    return {"method": Method, "class": Class, "in_test": InTest, "present": Present}


class TestStrictMode(unittest.TestCase):
    def test_strict_fails_each_absent_input_with_its_reason(self):
        sk = load_skips("1")
        self.assertFalse(sk.HAVE_NEMO_RL)
        for name, case in cases(sk).items():
            r = run(case)
            self.assertEqual(r.skipped, [], name)
            problems = [text for _, text in r.failures + r.errors]
            if name == "present":
                self.assertEqual((problems, r.testsRun), ([], 1))
                continue
            self.assertEqual(len(problems), 1, (name, problems))
            self.assertIn(f"{STRICT}=1", problems[0])
            self.assertIn(sk.NO_BASE if name == "in_test" else sk.NO_NEMO_RL, problems[0])

    def test_default_skips_with_the_reason(self):
        for value in (None, "0", ""):
            sk = load_skips(value)
            for name, case in cases(sk).items():
                r = run(case)
                self.assertEqual(r.failures + r.errors, [], (value, name))
                reasons = [why for _, why in r.skipped]
                want = {"method": [sk.NO_NEMO_RL], "class": [sk.NO_NEMO_RL] * 2, "in_test": [sk.NO_BASE],
                        "present": []}[name]
                self.assertEqual(reasons, want, (value, name))

    def test_host_executor_opt_out(self):
        # test_executor's numpy/sympy check runs (and fails without them) unless a host venv opts out explicitly;
        # strict mode ignores the opt-out
        self.assertEqual([load_skips(None, v).HOST_EXECUTOR for v in (None, "0", "", "1")], [False, False, False, True])
        self.assertFalse(load_skips("1", "1").HOST_EXECUTOR)

    def test_loading_leaves_no_module_behind(self):
        load_skips("1")
        self.assertNotIn("_skips", sys.modules)
        self.assertNotIn(str(RUNTIME_TESTS), sys.path)


if __name__ == "__main__":
    unittest.main()
