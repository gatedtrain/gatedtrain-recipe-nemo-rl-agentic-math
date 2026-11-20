"""Consumption log and join are environment-agnostic: math_code and NeMo RL's sliding_puzzle (the pinned clone's
real game logic and SlidingPuzzleEnv) both go through traj_log.TrajectoryLogger and consumption.build unchanged,
driven by a double of NeMo RL's synchronous multi-turn rollout (tests/_support.fake_rollout). Host CPU."""

from __future__ import annotations

import copy
import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from _skips import HAVE_BASE_TOKENIZER, HAVE_NEMO_RL, HAVE_TORCH, NO_BASE, NO_NEMO_RL, NO_TORCH
from _support import AREA, OUTPUTS, fake_rollout, install_nemo_stubs, load_fresh, read_jsonl, write_jsonl

import agentic_core
import consumption
import executor_server
import math_code_env
import traj_log

P, G = 4, 4          # prompts per step, generations per prompt
B = P * G


def run_steps(tmp: Path, steps: int, rows: list[dict], make_info, policy, env_step, max_turns: int, max_seq: int):
    logs_dir = tmp / "nemo" / "logs" / "exp_001"
    logs_dir.mkdir(parents=True)
    for k in range(1, steps + 1):
        group_rows = [rows[((k - 1) * P + g) % len(rows)] for g in range(P)]
        batch = [r for r in group_rows for _ in range(G)]            # repeated_batch: each prompt G times in a row
        lines = fake_rollout([r["prompt"] for r in batch], [make_info(r) for r in batch], policy, env_step,
                             max_turns, max_seq)
        write_jsonl(logs_dir / f"train_data_step{k}.jsonl", lines)
    return logs_dir


class MathCodeJoin(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="a10-cons-math-"))
        self.ex = executor_server.Executor(workers=4, tmp_root=str(self.tmp))

    def tearDown(self):
        self.ex.pool.shutdown()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_join_all_end_reasons(self):
        core = agentic_core.MathCodeCore(execute=self.ex.run_jobs,
                                         verifier=lambda t, g: float(f"\\boxed{{{g}}}" in t), max_turns=4)
        tlog = traj_log.TrajectoryLogger(str(self.tmp), "math_code")

        def env_step(msgs, md):
            return math_code_env.env_step(core, tlog, msgs, md)

        scripts = [["<code> print(6 * 7) </code>", "so \\boxed{42}"],                  # tool, then right answer
                   ["\\boxed{5}"],                                                       # direct, wrong
                   ["<code> 1/0 </code>"] * 4,                                           # max_turns (env decides)
                   ["<code> print(' '.join(['w'] * 300)) </code>"] * 4]                  # overflow (loop decides)

        def policy(i, turn, log):
            return scripts[i % G][turn]

        rows = [{"row_id": f"math-train-{k:05d}", "prompt": f"problem {k} what is six times seven", "answer": "42"}
                for k in range(6)]
        logs = run_steps(self.tmp, 2, rows, lambda r: {"ground_truth": r["answer"], "row_id": r["row_id"]},
                         policy, env_step, max_turns=4, max_seq=160)
        out = consumption.build(read_jsonl(self.tmp / "env_calls.jsonl"), consumption.load_step_logs(logs),
                                [r["row_id"] for r in rows], P, G, max_steps=2, max_turns=4, run="t", attempt=1)
        self.assertEqual(len(out), 2 * B)
        by_member = {}
        for r in out:
            by_member.setdefault(r["member"], set()).add((r["end_reason"], r["reward"], r["turns"]))
        self.assertEqual(by_member[0], {("final", 1.0, 2)})
        self.assertEqual(by_member[1], {("final", 0.0, 1)})
        self.assertEqual(by_member[2], {("max_turns", 0.0, 4)})
        self.assertEqual({e for e, _, _ in by_member[3]}, {"overflow"})
        r0 = next(r for r in out if r["member"] == 0)
        self.assertEqual([t["tool_status"] for t in r0["per_turn"]], ["ok", None])
        self.assertGreater(r0["per_turn"][0]["env_tokens"], 0)
        self.assertEqual(r0["per_turn"][1]["env_tokens"], 0)                  # terminal observation is empty
        self.assertEqual(r0["prompt_tokens"] + r0["model_tokens"] + r0["env_tokens"],
                         sum(t["model_tokens"] + t["env_tokens"] for t in r0["per_turn"]) + r0["prompt_tokens"])
        self.assertEqual([r["row_id"] for r in out if r["step"] == 2][::G], ["math-train-00004", "math-train-00005",
                                                                             "math-train-00000", "math-train-00001"])
        self.assertEqual(len({r["trajectory_id"] for r in out}), 2 * B)
        # refusals: wrong reward, wrong row order, missing step log, too many steps
        steps = consumption.load_step_logs(logs)
        calls = read_jsonl(self.tmp / "env_calls.jsonl")
        bad = copy.deepcopy(steps)
        bad[1][0]["rewards"] = [0.5]
        for args, msg in (((calls, bad, None, P, G, 2, 4), "reward"),
                          ((calls, steps, [r["row_id"] for r in rows][::-1], P, G, 2, 4), "rows"),
                          ((calls, {1: steps[1]}, None, P, G, 2, 4), "no train_data_step2"),
                          ((calls, steps, None, P, G, 1, 4), "max_steps")):
            with self.assertRaises(consumption.JoinError, msg=msg) as cm:
                consumption.build(*args, run="t", attempt=1)
            self.assertIn(msg.split()[0], str(cm.exception))


FIX = Path(__file__).resolve().parent / "fixtures" / "p1-a2"
REAL = OUTPUTS / "p1-a2" / "train"


class RealLogJoin(unittest.TestCase):
    """The join on fixtures cut from P1 run p1-a2's real env_calls.jsonl and train_data_step<N>.jsonl
    (tests/cut_fixtures.py, tests/fixtures/p1-a2/SOURCE.json): 3 steps x 2 prompts x 2 generations."""

    def load(self):
        rows = json.loads((FIX / "train_rows.json").read_text(encoding="utf-8"))
        return read_jsonl(FIX / "env_calls.jsonl"), consumption.load_step_logs(FIX / "nemo" / "logs"), rows

    def test_rows_and_exact_token_counts(self):
        calls, steps, train_rows = self.load()
        self.assertEqual(len(calls), 13)                       # 12 trajectories, one of them 2 turns
        out = consumption.build(calls, steps, train_rows, 2, 2, max_steps=3, max_turns=4, run="p1-a2", attempt=1)
        self.assertEqual(len(out), 12)
        got = [(r["step"], r["traj_pos"], r["row_id"], r["prompt_tokens"],
                [(t["model_tokens"], t["env_tokens"], t["eos"], t["tool_status"]) for t in r["per_turn"]],
                r["reward"], r["end_reason"]) for r in out]
        self.assertEqual(got, [
            (1, 0, "math-train-00000", 116, [(512, 0, False, None)], 0.0, "final"),
            (1, 1, "math-train-00000", 116, [(512, 0, False, None)], 0.0, "final"),
            (1, 2, "math-train-00008", 93, [(203, 9, True, "ok"), (7, 0, True, None)], 1.0, "final"),
            (1, 3, "math-train-00008", 93, [(166, 0, True, None)], 0.0, "final"),
            (2, 0, "math-train-00021", 117, [(129, 0, True, None)], 0.0, "final"),
            (2, 1, "math-train-00021", 117, [(173, 0, True, None)], 1.0, "final"),
            (2, 2, "math-train-00024", 109, [(512, 0, False, None)], 0.0, "final"),
            (2, 3, "math-train-00024", 109, [(301, 0, True, None)], 1.0, "final"),
            (3, 0, "math-train-00034", 92, [(197, 0, True, None)], 1.0, "final"),
            (3, 1, "math-train-00034", 92, [(200, 0, True, None)], 1.0, "final"),
            (3, 2, "math-train-00040", 96, [(419, 0, True, None)], 0.0, "final"),
            (3, 3, "math-train-00040", 96, [(512, 0, False, None)], 1.0, "final")])
        for r in out:
            line = steps[r["step"]][r["traj_pos"]]
            self.assertEqual(r["prompt_tokens"] + r["model_tokens"] + r["env_tokens"], line["input_lengths"][0])
            self.assertEqual(r["model_tokens"], sum(line["token_loss_mask"][0]))

    def test_token_segments_decode_to_logged_text(self):
        """Independent check with the Qwen3 tokenizer: the prompt / model / environment segments the join derives from
        the mask decode to exactly the logged message contents."""
        try:
            from tokenizers import Tokenizer
        except ImportError:  # pragma: no cover
            self.skipTest("tokenizers not installed in this venv")
        if not HAVE_BASE_TOKENIZER:
            self.skipTest(NO_BASE)
        tok = Tokenizer.from_file(str(AREA / "models" / "qwen3-1.7b" / "tokenizer.json"))
        calls, steps, train_rows = self.load()
        out = consumption.build(calls, steps, train_rows, 2, 2, max_steps=3, max_turns=4, run="p1-a2", attempt=1)
        for r in out:
            line = steps[r["step"]][r["traj_pos"]]
            ids, content = line["token_ids"][0], line["content"][0]
            at = r["prompt_tokens"]
            self.assertEqual(tok.decode(ids[:at], skip_special_tokens=False), content[0])
            for t, (asst, obs) in zip(r["per_turn"], zip(content[1::2], content[2::2])):
                self.assertEqual(tok.decode(ids[at:at + t["model_tokens"]], skip_special_tokens=True), asst)
                at += t["model_tokens"]
                self.assertEqual(tok.decode(ids[at:at + t["env_tokens"]], skip_special_tokens=False), obs)
                at += t["env_tokens"]

    def test_logged_shape_is_enforced(self):
        calls, steps, train_rows = self.load()
        for key, value in (("rewards", 1.0), ("token_loss_mask", [[0, 1], [0, 1]]), ("idx", 3)):
            bad = copy.deepcopy(steps)
            bad[1][0][key] = value
            with self.assertRaises(consumption.JoinError, msg=key):
                consumption.build(calls, bad, train_rows, 2, 2, max_steps=3, max_turns=4, run="t", attempt=1)

    @unittest.skipUnless((REAL / "env_calls.jsonl").exists(), "P1 run p1-a2 absent (GATEDTRAIN_RECIPE_NEMO_RL_AGENTIC_MATH_OUTPUTS)")
    def test_full_p1_a2_run(self):
        """The whole first GPU probe (read only): 385 env lines (384 trajectories, one with 2 turns), 384 rows."""
        rows = [r["row_id"] for r in read_jsonl(AREA / "data" / "train.jsonl")]
        calls = read_jsonl(REAL / "env_calls.jsonl")
        self.assertEqual(len(calls), 385)
        self.assertEqual(len(read_jsonl(REAL / "trajectories.jsonl")), 384)
        out = consumption.build(calls, consumption.load_step_logs(REAL / "nemo" / "logs"), rows, 16, 8, max_steps=3,
                                max_turns=4, run="p1-a2", attempt=1)
        self.assertEqual(len(out), 384)
        self.assertEqual(sum(r["turns"] for r in out), 385)
        self.assertEqual((sum(r["prompt_tokens"] for r in out), sum(r["model_tokens"] for r in out),
                          sum(r["env_tokens"] for r in out)), (51904, 151703, 9))


@unittest.skipUnless(HAVE_NEMO_RL, NO_NEMO_RL)
@unittest.skipUnless(HAVE_TORCH, NO_TORCH)
class SlidingPuzzleJoin(unittest.TestCase):
    """The same logger and join code with NeMo RL's sliding_puzzle environment."""

    def setUp(self):
        install_nemo_stubs()
        self.tl = load_fresh("traj_log.py", "traj_log_stubbed")
        import nemo_rl.environments.games.sliding_puzzle as sp
        self.sp = sp
        self.tmp = Path(tempfile.mkdtemp(prefix="a10-cons-sp-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_join_sliding_puzzle(self):
        sp = self.sp
        env = self.tl.LoggedSlidingPuzzleEnv({"log_dir": str(self.tmp)})
        self.assertIsInstance(env, sp.SlidingPuzzleEnv)

        def env_step(msgs, md):
            ret = env.step(msgs, md)
            return ([o["content"] for o in ret.observations], ret.metadata, ret.next_stop_strings,
                    ret.rewards.tolist(), [bool(t) for t in ret.terminateds.tolist()])

        rows = []
        for k in range(4):                     # seeded puzzles (NeMo RL's generator uses the global random)
            random.seed(f"sp:{k}")
            gs = sp.SlidingPuzzleGameLogic.generate({"size": 2, "shuffle_moves": 1})
            rows.append({"row_id": f"sp-{k}", "prompt": f"puzzle {k} slide tiles", "game_state": gs})
        moves = ["up", "down", "left", "right"]

        def policy(i, turn, log):
            if i % G == 3 and turn == 0:
                return "I am not sure"                                # parse failure: metadata None, continues
            return f"<action>{moves[(i + turn) % 4]}</action>"

        logs = run_steps(self.tmp, 2, rows,
                         lambda r: {"game_state": copy.deepcopy(r["game_state"]), "num_moves": 0, "max_moves": 2,
                                    "row_id": r["row_id"]},
                         policy, env_step, max_turns=6, max_seq=400)
        calls = read_jsonl(self.tmp / "env_calls.jsonl")
        out = consumption.build(calls, consumption.load_step_logs(logs), [r["row_id"] for r in rows], P, G,
                                max_steps=2, max_turns=6, run="t", attempt=1)
        self.assertEqual(len(out), 2 * B)
        self.assertEqual({r["env_id"] for r in out}, {"sliding_puzzle"})
        ends = {r["end_reason"] for r in out}
        self.assertIn("solved", ends)
        self.assertIn("max_moves", ends)
        self.assertTrue(all(r["reward"] in (0.0, 1.0) for r in out))
        self.assertTrue(any(r["member"] == 3 and r["turns"] >= 2 for r in out))   # survived the parse failure
        self.assertTrue(all(ln["row_id"] is not None for ln in calls))           # stamp and row id kept on None


if __name__ == "__main__":
    unittest.main()
