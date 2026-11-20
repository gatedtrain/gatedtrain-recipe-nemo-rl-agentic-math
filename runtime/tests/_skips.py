"""Skip conditions of area "agentic RL"'s CPU tests for the heavy inputs a fresh clone lacks: the base model's
tokenizer files (fetch_models.py), the training data (make_data.py), the NeMo RL clone (`GATEDTRAIN_NEMO_RL_DIR`),
torch, and numpy and sympy in the interpreter the executor runs model code with (the sidecar's NeMo RL image has both;
a host venv may not). Tests that need one skip with the reason. The repository's tests/test_runtime.py covers the
hooks, the environment, the trajectory log and the generation loop without any of them, in CI.

Kept apart from _support.py, whose SHA-256 the sliding-puzzle fixture records (tests/fixtures/agentic/sliding-puzzle/
SOURCE.json).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

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


def executor_python_has(*modules: str, python: str = sys.executable) -> bool:
    """Whether the interpreter the executor runs model code with (executor_server.Executor: `sys.executable -s -P`)
    imports `modules`."""
    code = "; ".join(f"import {m}" for m in modules)
    return subprocess.run([python, "-s", "-P", "-c", code], capture_output=True).returncode == 0
