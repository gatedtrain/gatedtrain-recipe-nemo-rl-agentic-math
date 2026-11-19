"""Wire protocol and client of area "agentic RL"'s code executor (standard library only).

Imported by both sides: the executor sidecar (`executor_server.py`, in the executor container, which mounts only this
`executor/` directory and the spool) and its callers (`math_code_env.py` in the NeMo RL trainer's environment actor,
`gen_agentic.py` in the evaluation's generation container). Copied from area 6's `executor/cad_protocol.py` wire
format (not imported across areas): one request and one response per connection over a Unix socket, each an 8-byte
big-endian length followed by UTF-8 JSON.

Requests:
  {"op": "ping"}                                  -> {"ok": true, "limits": {...}, "versions": {...}, ...}
  {"op": "run", "jobs": [{"id": str, "cells": [str, ...]}, ...]}
                                                  -> {"results": [{"id", "status", "output", "output_chars",
                                                                   "wall_s", ...}, ...]}  (same order)
  any fault of the executor itself                -> {"error": "executor_error: ..." | "protocol_error: ..."}

A job is one tool call of one trajectory under stateless replay: `cells` holds the trajectory's earlier replayable
code cells followed by the new cell; the executor runs them all in one fresh process (earlier cells' output
discarded) and returns only the new cell's output. The executor never receives problems, answers or any other
reference data (MM-34a, area 6 review 11).

Job `status` (all are observations for the policy, never infrastructure errors): `ok`, `error` (the new cell raised),
`syntax_error`, `timeout` (wall clock or CPU limit), `crash` (the process died or returned an unreadable result).
"""

from __future__ import annotations

import json
import socket
import struct

MAX_MESSAGE = 64 << 20
STATUSES = ("ok", "error", "syntax_error", "timeout", "crash")


def send_msg(sock: socket.socket, obj) -> None:
    data = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sock.sendall(struct.pack(">Q", len(data)) + data)


def recv_msg(sock: socket.socket):
    (n,) = struct.unpack(">Q", _recv_exact(sock, 8))
    if n > MAX_MESSAGE:
        raise ValueError(f"message of {n} bytes exceeds {MAX_MESSAGE}")
    return json.loads(_recv_exact(sock, n).decode("utf-8"))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("executor closed the connection")
        buf.extend(chunk)
    return bytes(buf)


def _request(socket_path: str, req: dict, timeout: float):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(socket_path)
        send_msg(s, req)
        return recv_msg(s)


class ExecutorUnavailable(RuntimeError):
    """Transport or executor fault: an infrastructure error, never a policy outcome (MM-4)."""


def ping(socket_path: str, timeout: float = 10.0) -> dict:
    try:
        resp = _request(socket_path, {"op": "ping"}, timeout)
    except (OSError, ValueError) as e:
        raise ExecutorUnavailable(f"executor ping failed: {type(e).__name__}: {e}") from e
    if not isinstance(resp, dict) or not resp.get("ok"):
        raise ExecutorUnavailable(f"executor ping answered {resp!r}")
    return resp


def call_executor(socket_path: str, jobs: list[dict], timeout: float) -> list[dict]:
    """Run one batch of jobs; returns one result per job, in order. Raises ExecutorUnavailable on any transport or
    protocol fault."""
    if not jobs:
        return []
    try:
        resp = _request(socket_path, {"op": "run", "jobs": jobs}, timeout)
    except (OSError, ValueError) as e:
        raise ExecutorUnavailable(f"executor call failed: {type(e).__name__}: {e}") from e
    if not isinstance(resp, dict) or resp.get("error") or "results" not in resp:
        raise ExecutorUnavailable(f"executor error: {resp.get('error') if isinstance(resp, dict) else resp!r}")
    results = resp["results"]
    if len(results) != len(jobs) or [r.get("id") for r in results] != [j["id"] for j in jobs]:
        raise ExecutorUnavailable("executor returned results that do not match the request")
    for r in results:
        if r.get("status") not in STATUSES:
            raise ExecutorUnavailable(f"executor returned unknown status {r.get('status')!r}")
    return results
