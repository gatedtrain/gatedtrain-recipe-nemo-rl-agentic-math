#!/usr/bin/env python
"""Cut the consumption-join fixtures in tests/fixtures/p1-a2/ from P1 run p1-a2's real logs (standard library only).

Source (read only): out/area10-agentic/p1-a2/train/{env_calls.jsonl, nemo/logs/exp_001/train_data_step{1,2,3}.jsonl}
of the first GPU probe (2026-11-15; 3 steps x 16 prompts x 8 generations). Per step two prompt groups and two members
each are kept (B = 4), chosen to cover: a 2-turn trajectory with an executed tool call (step 1, position 65), replies
truncated at 512 tokens without an answer, a truncated reply that still holds a correct \\boxed answer, replies whose
`<code>` block was not cut at `</code>` (the P1 stop-string defect, README 0.10), and plain boxed answers.

What changes against the source: env lines keep every field except `position` (renumbered within the first call of
the step) and `traj_pos` (renumbered); call numbers, row ids, rewards, hashes and timestamps are kept. Training lines
keep `content`, `rewards`, `input_lengths`, `token_ids`, `token_loss_mask` and `sample_loss_mask` byte-for-byte
(including the right padding of the step) and get `idx` renumbered; the unused per-token float arrays
(`advantages`, `generation_logprobs`, `prev_logprobs`) are dropped to keep the fixture small. `SOURCE.json` records
the SHA-256 of every source file and the selection.

Usage (host): python -I tests/cut_fixtures.py --src <out>/area10-agentic/p1-a2/train --dst tests/fixtures/p1-a2
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

G = 8                                          # generations per prompt in the source run
# step -> [(group index in the step, [original positions kept])]
SELECTION = {
    1: [(0, [0, 1]), (8, [65, 66])],
    2: [(5, [43, 46]), (8, [64, 65])],
    3: [(2, [16, 22]), (8, [64, 70])],
}
DROP = ("advantages", "generation_logprobs", "prev_logprobs")


def sha256_file(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    a = ap.parse_args(argv)
    env = [json.loads(x) for x in open(a.src / "env_calls.jsonl", encoding="utf-8")]
    first_calls = sorted({ln["traj_call"] for ln in env})
    logs = a.src / "nemo" / "logs" / "exp_001"
    out_env, rows = [], []
    (a.dst / "nemo" / "logs" / "exp_001").mkdir(parents=True, exist_ok=True)
    for k, sel in SELECTION.items():
        call = first_calls[k - 1]
        src_lines = [json.loads(x) for x in open(logs / f"train_data_step{k}.jsonl", encoding="utf-8")]
        new_pos = {}
        for g, members in sel:
            assert all(p // G == g for p in members), (k, g, members)
            for p in members:
                new_pos[p] = len(new_pos)
        for ln in env:
            if ln["traj_call"] == call and ln["traj_pos"] in new_pos:
                ln = dict(ln)
                ln["traj_pos"] = new_pos[ln["traj_pos"]]
                if ln["call"] == call:
                    ln["position"] = ln["traj_pos"]
                out_env.append(ln)
        out = []
        for p, i in sorted(new_pos.items(), key=lambda kv: kv[1]):
            rec = {key: v for key, v in src_lines[p].items() if key not in DROP}
            assert rec["idx"] == p
            rec["idx"] = i
            out.append(json.dumps(rec) + "\n")
        (a.dst / "nemo" / "logs" / "exp_001" / f"train_data_step{k}.jsonl").write_text("".join(out), encoding="utf-8")
        for g, members in sel:
            ids = {ln["row_id"] for ln in env if ln["traj_call"] == call and ln["traj_pos"] // G == g}
            assert len(ids) == 1, ids
            rows.append(ids.pop())
    out_env.sort(key=lambda ln: (ln["call"], ln["position"]))
    (a.dst / "env_calls.jsonl").write_text("".join(json.dumps(ln, sort_keys=True) + "\n" for ln in out_env),
                                           encoding="utf-8")
    (a.dst / "train_rows.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")
    src = {"env_calls.jsonl": sha256_file(a.src / "env_calls.jsonl")}
    src.update({f"nemo/logs/exp_001/train_data_step{k}.jsonl": sha256_file(logs / f"train_data_step{k}.jsonl")
                for k in SELECTION})
    (a.dst / "SOURCE.json").write_text(json.dumps({
        "run": "p1-a2", "source_dir": "out/area10-agentic/p1-a2/train", "source_sha256": src,
        "selection": {str(k): v for k, v in SELECTION.items()}, "prompts": 2, "generations": 2,
        "dropped_fields": list(DROP), "cutter": "tests/cut_fixtures.py"}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"env_lines": len(out_env), "rows": rows}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
