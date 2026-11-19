#!/usr/bin/env python
"""Code executor of area "agentic RL": the sidecar that runs model-written Python (MM-34a, as area 6's B18 executor).

Standard library only. Runs in its own container (`run-executor.sh` prints the docker commands): network none, uid
1000, all capabilities dropped, no-new-privileges, read-only root, tmpfs /tmp, CPU / memory / pids / open-file
limits, and mounts of exactly this `executor/` directory (read-only) and one spool directory (read-write, holds the
Unix socket); never the area root (items and training rows with their answers), never a run's /out. The trainer's
environment actor (`math_code_env.py`) and the evaluation generator (`gen_agentic.py`) are its only clients; they
mount the same spool at `/spool`.

Per job (one tool call of one trajectory, stateless replay of its code cells, see `exec_protocol.py`): a fresh
`<python> -s -P sandbox_runner.py` process in its own session, cwd a fresh empty directory under `--tmp-root`
(removed afterwards), stdin the job JSON, an environment built here (PATH, HOME = the work directory,
PYTHONHASHSEED=0, one BLAS/OpenMP thread; nothing inherited). The runner sets its own rlimits before any cell runs
(sandbox_runner.py). This process enforces the wall clock: on expiry it kills the job's whole process group
(status `timeout`); after every job it kills the group again (leftovers). A job killed by SIGXCPU/SIGKILL at the CPU
limit is `timeout`; any other death or an unreadable result is `crash`. Up to `--workers` jobs run at once (threads
waiting on child processes; no fork from a threaded process: `subprocess` uses posix_spawn/vfork without a
preexec function).

Trust boundary (M2). Model code runs with this server's uid and can leave every Python-level restriction of the
runner (ctypes reaches libc), so the boundary does not depend on them:
- the job inherits only its stdin/stdout/stderr, each one end of a socket pair created for that job (`close_fds`;
  never the listening socket or a client connection). Unlike a pipe, a socket cannot be reopened through
  `/proc/<pid>/fd`, so no other job can write into this job's result or read its input;
- `--serve` and `--run-file` make this process non-dumpable (`PR_SET_DUMPABLE` 0): a process of the same uid without
  CAP_SYS_PTRACE cannot open its `/proc/<pid>/fd` entries, read or write its memory, ptrace it or take its file
  descriptors (`pidfd_getfd`); the runner does the same to itself before reading its job;
- they also make this process a child subreaper (`PR_SET_CHILD_SUBREAPER`): a process a job forks, even in a new
  session, stays a descendant of this one; after every batch all remaining children are killed and reaped before the
  answer is sent, so no model code is alive when the client next connects. In the sidecar this process is the
  container's PID 1 and outlives every job;
- after every batch `--serve` checks that the socket path is still its own socket (device and inode): a job that
  replaced it gets the batch answered with `executor_error` and the server stops, so a client never reaches a
  socket a job bound (MM-4: a loud failure, never a forged result).
What a job can still do with the shared uid: signal (kill or stop) another job of the same batch or this server,
which turns that job into `timeout` / `crash` or ends the sidecar (a failure, never a forged result).

`<python>` is the interpreter running this server (`sys.executable`): in the sidecar the NeMo RL image's
`/opt/nemo_rl_venv/bin/python` (Python 3.13 with numpy and sympy available to model code, README section 0); in the
host tests the area-4 venv.

Modes:
  executor_server.py --serve /spool/mathexec.sock [--workers 8]   # sidecar
  executor_server.py --run-file JOBS.json                         # one batch from a file, results to stdout (tests)
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exec_protocol  # noqa: E402

RUNNER = HERE / "sandbox_runner.py"
DEFAULT_LIMITS = {
    "wall_s": 10.0,          # whole replay (all cells of the job)
    "cpu_s": 10,             # RLIMIT_CPU
    "as_mb": 2048,           # RLIMIT_AS
    "fsize_mb": 1,           # RLIMIT_FSIZE
    "nofile": 64,            # RLIMIT_NOFILE
    "nproc_extra": 32,       # RLIMIT_NPROC = this uid's tasks at start + this margin
    "capture_chars": 16384,  # characters of the new cell's output kept by the runner
}
MAX_JOBS = 1024
MAX_CELL_CHARS = 64 << 10
MAX_RESULT_BYTES = 4 << 20   # a runner's result frame is at most about 6 * capture_chars bytes; the rest is dropped
PR_SET_DUMPABLE, PR_SET_CHILD_SUBREAPER = 4, 36


def sha256_file(p: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code_hashes() -> dict:
    return {name: sha256_file(HERE / name) for name in ("executor_server.py", "sandbox_runner.py", "exec_protocol.py")}


def _child_env(workdir: str) -> dict:
    return {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": workdir, "TMPDIR": workdir, "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8", "LANG": "C.UTF-8",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _prctl(option: int, value: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(option, ctypes.c_ulong(value), ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0)) != 0:
        e = ctypes.get_errno()
        raise OSError(e, f"prctl({option}, {value}): {os.strerror(e)}")


def harden_process() -> None:
    """Non-dumpable (jobs cannot reach this process's fds or memory) and child subreaper (jobs' orphans stay ours)."""
    _prctl(PR_SET_DUMPABLE, 0)
    _prctl(PR_SET_CHILD_SUBREAPER, 1)


def _children() -> list[int]:
    me, out = os.getpid(), []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as f:
                stat = f.read()
            if int(stat[stat.rfind(b")") + 2:].split()[1]) == me:
                out.append(int(name))
        except (OSError, ValueError, IndexError):
            continue
    return out


def reap_strays(rounds: int = 100) -> int:
    """Kill and reap every child of this process. Called only when no runner is running (after a batch): with
    `harden_process()` every remaining child is a process a job left behind. Returns how many were killed."""
    killed = 0
    for _ in range(rounds):
        kids = _children()
        if not kids:
            return killed
        for pid in kids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for pid in kids:
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        killed += len(kids)
    raise RuntimeError("processes left behind by jobs keep appearing")


def _exchange(p: subprocess.Popen, sin: socket.socket, sout: socket.socket, serr: socket.socket, payload: bytes,
              wall: float) -> tuple[bytes, bytes, bool]:
    """Write the job to the runner, read its stdout and stderr to EOF and wait for it, all within `wall` seconds.
    Returns (stdout, stderr, timed_out)."""
    deadline = time.monotonic() + wall
    try:
        sin.settimeout(wall)
        sin.sendall(payload)
    except OSError:
        pass                                    # the runner died or never read: its exit status tells
    finally:
        sin.close()
    bufs = {sout: bytearray(), serr: bytearray()}
    with selectors.DefaultSelector() as sel:
        for s in bufs:
            s.setblocking(False)
            sel.register(s, selectors.EVENT_READ)
        while sel.get_map():
            left = deadline - time.monotonic()
            if left <= 0:
                return bytes(bufs[sout]), bytes(bufs[serr]), True
            for key, _ in sel.select(left):
                try:
                    chunk = key.fileobj.recv(1 << 16)
                except BlockingIOError:
                    continue
                except OSError:
                    chunk = b""
                if not chunk:
                    sel.unregister(key.fileobj)
                    continue
                buf = bufs[key.fileobj]
                buf.extend(chunk[:max(0, MAX_RESULT_BYTES - len(buf))])
    try:
        p.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        return bytes(bufs[sout]), bytes(bufs[serr]), True
    return bytes(bufs[sout]), bytes(bufs[serr]), False


class Executor:
    def __init__(self, workers: int = 8, limits: dict | None = None, tmp_root: str | None = None,
                 python: str | None = None):
        self.workers = int(workers)
        self.limits = {**DEFAULT_LIMITS, **(limits or {})}
        self.tmp_root = tmp_root or tempfile.gettempdir()
        self.python = python or sys.executable
        self.pool = ThreadPoolExecutor(max_workers=self.workers)
        self.info = {"python": self.python, "python_version": sys.version.split()[0], "code_sha256": code_hashes(),
                     "limits": self.limits, "workers": self.workers}

    def run_job(self, job: dict) -> dict:
        cells = job["cells"]
        if not isinstance(cells, list) or not cells or not all(isinstance(c, str) for c in cells):
            raise ValueError(f"job {job.get('id')!r}: cells must be a non-empty list of strings")
        if sum(len(c) for c in cells) > MAX_CELL_CHARS * max(1, len(cells)):
            raise ValueError(f"job {job.get('id')!r}: cells too large")
        lim = self.limits
        wd = tempfile.mkdtemp(prefix="mathexec-", dir=self.tmp_root)
        payload = json.dumps({"cells": cells, "limits": lim}).encode("utf-8")
        t0 = time.monotonic()
        # stdin, stdout and stderr: one end each of a socket pair of this job (see "Trust boundary" above)
        pairs = [socket.socketpair() for _ in range(3)]
        try:
            try:
                p = subprocess.Popen([self.python, "-s", "-P", str(RUNNER)], stdin=pairs[0][1].fileno(),
                                     stdout=pairs[1][1].fileno(), stderr=pairs[2][1].fileno(), cwd=wd,
                                     env=_child_env(wd), start_new_session=True, close_fds=True)
            except OSError as e:
                raise RuntimeError(f"cannot start the sandbox process: {e}") from e
            finally:
                for _, child_end in pairs:
                    child_end.close()
            try:
                out, err, timed_out = _exchange(p, pairs[0][0], pairs[1][0], pairs[2][0], payload, float(lim["wall_s"]))
            finally:
                _kill_group(p.pid)
                if p.poll() is None:
                    p.kill()
                p.wait()
            wall = round(time.monotonic() - t0, 3)
            base = {"id": job["id"], "wall_s": wall, "returncode": p.returncode}
            if timed_out:
                return {**base, "status": "timeout", "output": "", "output_chars": 0, "detail": "wall"}
            rc = p.returncode
            if rc == 0:
                try:
                    res = json.loads(out.decode("utf-8"))
                    if res.get("status") in ("ok", "error", "syntax_error") and isinstance(res.get("output"), str):
                        return {**base, "status": res["status"], "output": res["output"],
                                "output_chars": int(res.get("output_chars", len(res["output"])))}
                except (ValueError, UnicodeDecodeError, AttributeError, TypeError):
                    pass
                return {**base, "status": "crash", "output": "", "output_chars": 0, "detail": "unreadable result"}
            if rc in (-signal.SIGXCPU, -signal.SIGKILL):
                return {**base, "status": "timeout", "output": "", "output_chars": 0,
                        "detail": f"cpu (signal {-rc})"}
            tail = err.decode("utf-8", "replace")[-300:] if err else ""
            return {**base, "status": "crash", "output": "", "output_chars": 0,
                    "detail": f"exit {rc}" if rc > 0 else f"signal {-rc}", "stderr_tail": tail}
        finally:
            for parent_end, _ in pairs:
                parent_end.close()
            shutil.rmtree(wd, ignore_errors=True)

    def run_jobs(self, jobs: list[dict]) -> list[dict]:
        if len(jobs) > MAX_JOBS:
            raise ValueError(f"{len(jobs)} jobs exceed {MAX_JOBS}")
        ids = [str(j["id"]) for j in jobs]
        if len(set(ids)) != len(ids):
            raise ValueError("job ids must be unique")
        return list(self.pool.map(self.run_job, [{**j, "id": str(j["id"])} for j in jobs]))


WARMUP_CELL = "import numpy, sympy\nsympy.Integer(1) + 1"


def warmup(ex: Executor) -> dict:
    """Run one trusted job before listening, so the first training step does not pay the cold import of numpy and
    sympy from disk (57 s measured with a cold page cache, 0.2 s warm) inside its 10 s wall limit."""
    slow = Executor(workers=1, tmp_root=ex.tmp_root, python=ex.python, limits={**ex.limits, "wall_s": 600.0})
    try:
        return slow.run_job({"id": "warmup", "cells": [WARMUP_CELL]})
    finally:
        slow.pool.shutdown()


def _path_identity(path: str):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_dev, st.st_ino


def serve(sock_path: str, ex: Executor, do_warmup: bool = True) -> int:
    """The sidecar's main loop; run it in a process of its own (it hardens the process and, after every batch, kills
    all of the process's children)."""
    harden_process()
    if do_warmup:
        w = warmup(ex)
        print(json.dumps({"event": "warmup", "status": w["status"], "wall_s": w["wall_s"]}), flush=True)
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old = os.umask(0o077)
    srv.bind(sock_path)
    os.umask(old)
    srv.listen(16)
    ident = _path_identity(sock_path)
    print(json.dumps({"event": "listening", "socket": sock_path, **ex.info}), flush=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    srv.settimeout(1.0)
    n_batches = 0
    while not stop.is_set():
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        except OSError as e:
            if e.errno == errno.EINTR:
                continue
            raise
        with conn:
            conn.settimeout(600)
            try:
                req = exec_protocol.recv_msg(conn)
                if req.get("op") == "ping":
                    exec_protocol.send_msg(conn, {"ok": True, **ex.info})
                    continue
                if req.get("op") != "run":
                    exec_protocol.send_msg(conn, {"error": f"protocol_error: unknown op {req.get('op')!r}"})
                    continue
                t0 = time.monotonic()
                try:
                    res = ex.run_jobs(req["jobs"])
                finally:
                    strays = reap_strays()
                if _path_identity(sock_path) != ident:
                    stop.set()
                    print(json.dumps({"event": "socket_replaced", "socket": sock_path}), flush=True)
                    raise RuntimeError(f"the socket {sock_path} was replaced while jobs ran; the executor stops")
                n_batches += 1
                counts: dict[str, int] = {}
                for r in res:
                    counts[r["status"]] = counts.get(r["status"], 0) + 1
                print(json.dumps({"event": "batch", "n": len(res), "seconds": round(time.monotonic() - t0, 3),
                                  "statuses": counts, "batches": n_batches, "strays_killed": strays}), flush=True)
                exec_protocol.send_msg(conn, {"results": res})
            except RuntimeError as e:
                exec_protocol.send_msg(conn, {"error": f"executor_error: {e}"})
            except (ConnectionError, ValueError, KeyError, TypeError) as e:
                try:
                    exec_protocol.send_msg(conn, {"error": f"protocol_error: {type(e).__name__}: {e}"})
                except OSError:
                    pass
    srv.close()
    if _path_identity(sock_path) != ident:
        return 3                                # not ours any more: leave it, exit with an error
    try:
        os.unlink(sock_path)
    except OSError:
        pass
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--serve", metavar="SOCKET")
    g.add_argument("--run-file", type=Path, metavar="JOBS_JSON")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tmp-root", default=None)
    p.add_argument("--wall-s", type=float, default=DEFAULT_LIMITS["wall_s"])
    p.add_argument("--no-warmup", action="store_true")
    a = p.parse_args(argv)
    ex = Executor(workers=a.workers, tmp_root=a.tmp_root, limits={"wall_s": a.wall_s})
    if a.serve:
        return serve(a.serve, ex, do_warmup=not a.no_warmup)
    harden_process()
    jobs = json.loads(a.run_file.read_text(encoding="utf-8"))
    try:
        results = ex.run_jobs(jobs)
    finally:
        reap_strays()
    print(json.dumps({"info": ex.info, "results": results}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
