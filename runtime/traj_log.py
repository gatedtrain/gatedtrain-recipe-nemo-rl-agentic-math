"""Environment-agnostic consumption log of area "agentic RL" (MM-27: a wrapper around the environment's step; no NeMo
RL source is edited). Standard library only, plus the Ray actor `LoggedSlidingPuzzleEnv` when NeMo RL is importable.

NeMo RL hands an environment only `role`/`content` of each message (rollouts.py:915-918) and keeps no per-trajectory
identity across turns, so the environment stamps one into the metadata it returns:
- In NeMo RL's synchronous multi-turn rollout (rollouts.py:1084-1243) every trajectory of a GRPO step is active in
  the first environment call of that step, in batch order (the step's `repeated_batch`), and later calls carry only
  the still-active trajectories, in order (`calculate_rewards` sorts results back, :996-1000).
- A sample whose incoming metadata has no stamp is in its first turn: it gets `{"traj_call": <this call>,
  "traj_pos": <its position>}` under the metadata key `_ptc`. The stamp rides in the returned metadata, which NeMo RL
  keeps for continuing samples (:1236-1243). A continuing sample whose environment returned metadata `None` (NeMo RL
  then keeps the previous metadata, :1241; sliding_puzzle does this on a parse failure, sliding_puzzle.py:311-314)
  gets its incoming (stamped) metadata back: the same object NeMo RL would have kept.
- Each call appends one fsynced line per sample to `env_calls.jsonl`: `{env_id, call, position, traj_call, traj_pos,
  turn (assistant messages so far), row_id, reward, terminated, end_reason, tool_status, obs_chars,
  response_sha256 (SHA-256 of the concatenated assistant contents so far), utc}`; a terminated sample also appends
  one line to `trajectories.jsonl`. `consumption.py` joins these with NeMo RL's `train_data_step<N>.jsonl`.

Nothing here depends on the environment: `math_code_env.py` and the sliding-puzzle subclass below use the same
`TrajectoryLogger`; tests/test_consumption.py drives both through one consumption builder.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

STAMP = "_ptc"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sha256_text(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


def assistant_text(conv: list[dict]) -> str:
    return "".join(str(m["content"]) for m in conv if m["role"] == "assistant")


def n_assistant(conv: list[dict]) -> int:
    return sum(1 for m in conv if m["role"] == "assistant")


class TrajectoryLogger:
    def __init__(self, log_dir: str | None, env_id: str):
        self.env_id = env_id
        self.calls = 0
        self.env_calls = os.path.join(log_dir, "env_calls.jsonl") if log_dir else None
        self.trajectories = os.path.join(log_dir, "trajectories.jsonl") if log_dir else None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def record(self, message_log_batch: list[list[dict]], metadata_in: list, metadata_out: list, rewards: list[float],
               terminateds: list[bool], observations: list[str], end_reasons: list | None = None,
               tool_statuses: list | None = None) -> list:
        """Stamp the outgoing metadata and log the call. Returns the metadata list to hand back to NeMo RL."""
        self.calls += 1
        call = self.calls
        n = len(message_log_batch)
        end_reasons = end_reasons or [None] * n
        tool_statuses = tool_statuses or [None] * n
        utc = _utc()
        lines, ends, md_ret = [], [], []
        for i in range(n):
            md_in = metadata_in[i] if metadata_in[i] is not None else {}
            stamp = md_in.get(STAMP) or {"traj_call": call, "traj_pos": i}
            term = bool(terminateds[i])
            md_out = metadata_out[i]
            if md_out is None and not term:
                md_out = md_in                       # NeMo RL keeps the previous metadata (rollouts.py:1241)
            if md_out is not None:
                md_out = {**md_out, STAMP: stamp}
            md_ret.append(md_out)
            conv = message_log_batch[i]
            line = {"env_id": self.env_id, "call": call, "position": i, "traj_call": stamp["traj_call"],
                    "traj_pos": stamp["traj_pos"], "turn": n_assistant(conv), "row_id": md_in.get("row_id"),
                    "reward": float(rewards[i]), "terminated": term,
                    "end_reason": (end_reasons[i] or "terminated") if term else None,
                    "tool_status": tool_statuses[i], "obs_chars": len(observations[i]),
                    "response_sha256": sha256_text(assistant_text(conv)), "utc": utc}
            lines.append(line)
            if term:
                ends.append({k: line[k] for k in ("env_id", "traj_call", "traj_pos", "turn", "row_id", "end_reason",
                                                  "response_sha256", "call", "utc")})
        if self.env_calls:
            _append(self.env_calls, lines)
            if ends:
                _append(self.trajectories, ends)
        return md_ret


def _append(path: str, rows: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def sliding_end_reason(reward: float) -> str:
    """sliding_puzzle terminates on a solve (reward 1.0, sliding_puzzle.py:192-198) or at max_moves (:288-300)."""
    return "solved" if reward >= 1.0 else "max_moves"


try:  # the Ray actor exists only inside the NeMo RL image
    import ray
    from nemo_rl.environments.games.sliding_puzzle import SlidingPuzzleEnv as _RemoteSlidingPuzzleEnv

    @ray.remote(max_restarts=0, max_task_retries=0)  # pragma: no cover
    class LoggedSlidingPuzzleEnv(_RemoteSlidingPuzzleEnv.__ray_actor_class__):
        """NeMo RL's SlidingPuzzleEnv (unchanged game logic) with the trajectory log (cfg `log_dir`)."""

        def __init__(self, cfg=None):
            cfg = dict(cfg or {})
            self.tlog = TrajectoryLogger(cfg.pop("log_dir", None), "sliding_puzzle")
            super().__init__(cfg)

        def step(self, message_log_batch, metadata):
            ret = super().step(message_log_batch, metadata)
            rewards = ret.rewards.tolist()
            terms = [bool(t) for t in ret.terminateds.tolist()]
            md = self.tlog.record(message_log_batch, metadata, list(ret.metadata), rewards, terms,
                                  [o["content"] for o in ret.observations],
                                  [sliding_end_reason(r) if t else None for r, t in zip(rewards, terms)])
            return ret._replace(metadata=md)
except ImportError:  # host: no ray / nemo_rl
    LoggedSlidingPuzzleEnv = None
