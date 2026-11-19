#!/usr/bin/env python
"""Observed consumption of an agentic GRPO run: one row per trajectory (standard library only, environment-agnostic).

Joins the environment's own log (`env_calls.jsonl`, written by `traj_log.TrajectoryLogger` in any environment
actor: one line per sample per environment call, i.e. per model turn; `trajectories.jsonl` has one line per finished
trajectory, so a run with T_i turns per trajectory has sum(T_i) env lines and one trajectory line each) with NeMo
RL's per-step training log `train_data_step<N>.jsonl` (written by the classic `grpo_train`, grpo.py:4667-4691,
through `Logger.log_batched_dict_as_jsonl`, utils/logger.py:1409-1441).

Shape of a logged line (verified on P1 run p1-a2, 2026-11-15): the batch is written with
`make_microbatch_iterator(1)`, so every field is a batch of ONE (a one-element list) and `idx` is the line number:
`content` = [[prompt (chat-templated), assistant, observation, assistant, observation, ...]] (the flattened message
contents, llm_message_utils.py:336-341; the terminal observation is ''), `rewards` = [total reward] (the sum over
turns, rollouts.py:1166), `input_lengths` = [L], `token_ids` / `token_loss_mask` / `advantages` /
`generation_logprobs` / `prev_logprobs` = [[... padded to the step's longest sequence ...]], `sample_loss_mask` = [1.0],
`idx` = i. `token_loss_mask` is 1 exactly on the tokens the policy generated (grpo.py:2437-2441).

Rules (any violation refuses the whole run; the `iteration` fallback is the operator's decision, never automatic):
- Step k's trajectories are those first seen in the k-th environment call whose samples are all in their first turn
  (stamp `traj_call` = that call); there must be exactly prompts x generations of them, positions 0..B-1.
- Steps 1..max_steps only; `train_data_step<k>.jsonl` exists for each and has B lines; line i has idx i and every
  logged field is a batch of one; none beyond.
- Row order: the positions g*G .. g*G+G-1 share one row_id (the prompt group), and the groups of step k are the
  training rows (k-1)*P .. k*P-1 of the training file in file order (wrapping around at the end of the file).
- Per trajectory: the env's lines are consecutive turns 1..T; only the last may be terminated; the float32 sum of
  the env rewards equals the logged reward; the SHA-256 of the concatenated assistant contents (`content[1::2]`)
  equals the env's last `response_sha256`; the 1-runs of `token_loss_mask[:L]` number exactly T (one per model
  turn) and give per-turn model tokens; the 0-runs after each give the environment tokens of that turn; the leading
  0-run is the prompt; prompt + model + environment tokens = L exactly.
- Per turn `eos`: the turn's last generated token is an EOS id (`--eos-ids`, Qwen3 generation_config.json
  151645, 151643); false means the generation stopped at a stop string or ran out of tokens (truncated).
- End reason: the env's own (`final`, `no_answer`, `max_turns`, `solved`, `max_moves`, ...) when its last line
  terminated; else `max_turns` if T == max_rollout_turns (NeMo RL's loop ended it, rollouts.py:1245) or `overflow`
  (the observation was cut at max_total_sequence_length, rollouts.py:1180-1196).

Row: {trajectory_id, row_id, data_ref.row, step, traj_pos, member, env_id, turns, per_turn [{model_tokens,
env_tokens, eos, tool_status}], prompt_tokens, model_tokens, env_tokens, reward, end_reason, response_sha256,
outcome}.

Usage: consumption.py --env-calls ENV_CALLS --train-logs DIR --train-file data/train.jsonl --prompts 16
       --generations 8 --max-steps N --max-turns 4 --run RUN --attempt 1 --out consumption.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path


class JoinError(Exception):
    pass


def sha256_text(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


def f32(x: float) -> bytes:
    return struct.pack("<f", float(x))


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


EOS_IDS = (151645, 151643)          # Qwen3 generation_config.json eos_token_id
LOGGED = ("content", "rewards", "input_lengths", "token_ids", "token_loss_mask")


def logged_sample(rec: dict, k: int, line: int) -> dict:
    """One line of train_data_step<k>.jsonl unwrapped from its batch of one (logger.py:1427-1441)."""
    if rec.get("idx") != line:
        raise JoinError(f"step {k} line {line}: idx {rec.get('idx')!r}")
    out = {}
    for key in LOGGED:
        v = rec.get(key)
        if not (isinstance(v, list) and len(v) == 1):
            raise JoinError(f"step {k} line {line}: {key} is not a batch of one")
        out[key] = v[0]
    c, r, n, ids, m = (out[x] for x in LOGGED)
    if not (isinstance(c, list) and len(c) >= 3 and len(c) % 2 == 1 and all(isinstance(x, str) for x in c)):
        raise JoinError(f"step {k} line {line}: content is not [prompt, (assistant, observation)+]")
    if isinstance(r, bool) or not isinstance(r, (int, float)):
        raise JoinError(f"step {k} line {line}: rewards {r!r}")
    if isinstance(n, bool) or not isinstance(n, int) or not (isinstance(m, list) and isinstance(ids, list)) \
            or not 0 < n <= len(m) or len(ids) != len(m) or any(v not in (0, 1) for v in m):
        raise JoinError(f"step {k} line {line}: input_lengths / token_ids / token_loss_mask shapes")
    return out


def runs(mask: list[int]) -> list[tuple[int, int]]:
    """[(value, length)] run-length encoding."""
    out: list[list[int]] = []
    for v in mask:
        v = 1 if v else 0
        if out and out[-1][0] == v:
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return [(v, n) for v, n in out]


def build(env_calls: list[dict], step_logs: dict[int, list[dict]], train_rows: list[str] | None, prompts: int,
          generations: int, max_steps: int, max_turns: int, run: str, attempt: int,
          eos_ids: tuple[int, ...] = EOS_IDS) -> list[dict]:
    B = prompts * generations
    by_call: dict[int, list[dict]] = defaultdict(list)
    trajs: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for ln in env_calls:
        by_call[ln["call"]].append(ln)
        trajs[(ln["traj_call"], ln["traj_pos"])].append(ln)
    first_calls = sorted(c for c, lines in by_call.items()
                         if all(ln["traj_call"] == c and ln["turn"] == 1 for ln in lines))
    mixed = [c for c, lines in by_call.items()
             if c not in first_calls and any(ln["traj_call"] == c for ln in lines)]
    if mixed:
        raise JoinError(f"calls {mixed[:5]} start trajectories but are not pure first-turn calls")
    if len(first_calls) > max_steps:
        raise JoinError(f"{len(first_calls)} steps in the env log, max_steps is {max_steps}")
    extra = sorted(set(step_logs) - set(range(1, len(first_calls) + 1)))
    if extra:
        raise JoinError(f"train_data_step files without env steps: {extra}")
    rows = []
    for k, call in enumerate(first_calls, start=1):
        starts = sorted(ln["position"] for ln in by_call[call])
        if starts != list(range(B)):
            raise JoinError(f"step {k} (call {call}): {len(starts)} trajectories, expected positions 0..{B - 1}")
        if k not in step_logs:
            raise JoinError(f"step {k}: no train_data_step{k}.jsonl")
        if len(step_logs[k]) != B:
            raise JoinError(f"step {k}: train_data_step has {len(step_logs[k])} lines, expected {B}")
        log = [logged_sample(rec, k, i) for i, rec in enumerate(step_logs[k])]
        groups = []
        for g in range(prompts):
            ids = {trajs[(call, p)][0]["row_id"] for p in range(g * generations, (g + 1) * generations)}
            if len(ids) != 1:
                raise JoinError(f"step {k} group {g}: row ids {sorted(map(str, ids))} differ within the group")
            groups.append(ids.pop())
        if train_rows is not None:
            want = [train_rows[((k - 1) * prompts + g) % len(train_rows)] for g in range(prompts)]
            if groups != want:
                raise JoinError(f"step {k}: rows {groups[:3]}... are not the training file's rows {want[:3]}...")
        for pos in range(B):
            lines = sorted(trajs[(call, pos)], key=lambda ln: ln["call"])
            T = len(lines)
            if [ln["turn"] for ln in lines] != list(range(1, T + 1)):
                raise JoinError(f"step {k} pos {pos}: turns {[ln['turn'] for ln in lines]}")
            if any(ln["terminated"] for ln in lines[:-1]):
                raise JoinError(f"step {k} pos {pos}: a turn after a terminated one")
            last = lines[-1]
            if last["terminated"]:
                end = last["end_reason"]
            else:
                end = "max_turns" if T == max_turns else "overflow"
            reward = sum(ln["reward"] for ln in lines)
            rec = log[pos]
            if f32(reward) != f32(rec["rewards"]):
                raise JoinError(f"step {k} pos {pos}: env reward {reward} != logged {rec['rewards']}")
            content = rec["content"]
            if sha256_text("".join(str(c) for c in content[1::2])) != last["response_sha256"]:
                raise JoinError(f"step {k} pos {pos}: assistant text hash differs")
            L = rec["input_lengths"]
            r = runs(rec["token_loss_mask"][:L])
            prompt_tokens = r[0][1] if r and r[0][0] == 0 else 0
            body = r[1:] if r and r[0][0] == 0 else r
            per_turn = []
            i, at = 0, prompt_tokens
            while i < len(body):
                v, n = body[i]
                if v != 1:
                    raise JoinError(f"step {k} pos {pos}: mask runs {r}")
                env = body[i + 1][1] if i + 1 < len(body) else 0
                per_turn.append({"model_tokens": n, "env_tokens": env,
                                 "eos": rec["token_ids"][at + n - 1] in eos_ids})
                at += n + env
                i += 2
            if len(per_turn) != T:
                raise JoinError(f"step {k} pos {pos}: {len(per_turn)} model-token runs for {T} turns")
            for t, ln in zip(per_turn, lines):
                t["tool_status"] = ln.get("tool_status")
            model = sum(t["model_tokens"] for t in per_turn)
            envt = sum(t["env_tokens"] for t in per_turn)
            if prompt_tokens + model + envt != L:
                raise JoinError(f"step {k} pos {pos}: token counts do not add up to {L}")
            rows.append({
                "trajectory_id": sha256_text(f"{run}:{attempt}:{k}:{pos}"), "row_id": last["row_id"],
                "data_ref": {"row": sha256_text(str(last["row_id"]))}, "step": k, "traj_pos": pos,
                "member": pos % generations, "env_id": last["env_id"], "turns": T, "per_turn": per_turn,
                "prompt_tokens": prompt_tokens, "model_tokens": model, "env_tokens": envt, "reward": reward,
                "end_reason": end, "response_sha256": last["response_sha256"], "outcome": "consumed"})
    return rows


def load_step_logs(d: Path) -> dict[int, list[dict]]:
    out = {}
    for p in sorted(d.rglob("train_data_step*.jsonl")):
        k = int(p.name[len("train_data_step"):-len(".jsonl")])
        if k in out:
            raise JoinError(f"two train_data_step{k}.jsonl under {d}")
        out[k] = read_jsonl(p)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--env-calls", type=Path, required=True)
    p.add_argument("--train-logs", type=Path, required=True)
    p.add_argument("--train-file", type=Path)
    p.add_argument("--prompts", type=int, default=16)
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--max-turns", type=int, default=4)
    p.add_argument("--run", required=True)
    p.add_argument("--attempt", type=int, default=1)
    p.add_argument("--eos-ids", default=",".join(map(str, EOS_IDS)), help="comma-separated EOS token ids")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    train_rows = [json.loads(x)["row_id"] for x in open(a.train_file, encoding="utf-8")] if a.train_file else None
    try:
        rows = build(read_jsonl(a.env_calls), load_step_logs(a.train_logs), train_rows, a.prompts, a.generations,
                     a.max_steps, a.max_turns, a.run, a.attempt,
                     eos_ids=tuple(int(x) for x in a.eos_ids.split(",")))
    except JoinError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    a.out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    ends: dict[str, int] = {}
    for r in rows:
        ends[r["end_reason"]] = ends.get(r["end_reason"], 0) + 1
    print(json.dumps({"trajectories": len(rows), "steps": len({r["step"] for r in rows}), "end_reasons": ends,
                      "turns": sum(r["turns"] for r in rows),
                      "tool_calls": sum(t["tool_status"] is not None for r in rows for t in r["per_turn"]),
                      "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
                      "model_tokens": sum(r["model_tokens"] for r in rows),
                      "env_tokens": sum(r["env_tokens"] for r in rows),
                      "final_turn_not_eos": sum(not r["per_turn"][-1]["eos"] for r in rows)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
