"""One job of area "agentic RL"'s code executor: runs in a fresh child process started by executor_server.py.

Never imported. Started as `<python> -s -P sandbox_runner.py` with an environment built by the server (no PYTHONPATH,
PYTHONHASHSEED=0, one BLAS thread), cwd a fresh empty directory under the executor's /tmp (tmpfs in the container),
its own session, stdin / stdout / stderr each one end of a socket pair of this job (executor_server.py, "Trust
boundary"). First it makes itself non-dumpable (`PR_SET_DUMPABLE` 0: other jobs of the same uid cannot open its
`/proc/<pid>/fd` entries, read or write its memory or ptrace it). Reads `{"cells": [...], "limits": {...}}` from
stdin, then, before any cell runs:
- stdin becomes /dev/null; file descriptors 1 and 2 go to /dev/null (a cell that writes to the raw descriptors
  writes nowhere); the result goes to a private duplicate of the original stdout;
- resource limits with soft = hard (a cell cannot raise them): RLIMIT_CPU, RLIMIT_AS, RLIMIT_FSIZE, RLIMIT_NOFILE,
  RLIMIT_CORE 0, RLIMIT_NPROC (this uid's current task count plus a margin, so fork bombs fail);
- `random.seed(0)`, so a replay sees the same random stream as the original run;
- a Python audit hook (defence in depth only; native code can bypass it, the container is the boundary) that refuses
  sockets, subprocesses, exec/spawn/fork, signals and opening files for writing outside the work directory.

Stateless replay: all cells but the last run first with their output discarded and their exceptions ignored (they
were already observed when they were new); then the last cell runs with `print` output captured (at most
`capture_chars` characters kept, the total counted), notebook style: when its last statement is an expression whose
value is not None, `repr(value)` is printed. Result `{"status", "output", "output_chars"}`, status `ok`, `error`
(the new cell raised; the output ends with the exception line) or `syntax_error`. Timeouts and crashes are
classified by the server from the exit status.
"""

import ast
import builtins
import ctypes
import json
import os
import random
import resource
import sys


class _Capture:
    """Minimal text stream: keeps the first `limit` characters, counts all."""

    def __init__(self, limit: int):
        self.limit = limit
        self.parts: list[str] = []
        self.kept = 0
        self.total = 0

    def write(self, s):
        if not isinstance(s, str):
            s = str(s)
        self.total += len(s)
        room = self.limit - self.kept
        if room > 0:
            piece = s[:room]
            self.parts.append(piece)
            self.kept += len(piece)
        return len(s)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        return None

    def isatty(self):
        return False

    def fileno(self):
        raise OSError("captured stream has no file descriptor")

    @property
    def encoding(self):
        return "utf-8"

    def text(self) -> str:
        return "".join(self.parts)


def _set_nondumpable() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0)) != 0:  # PR_SET_DUMPABLE
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_DUMPABLE, 0) failed")


def _uid_tasks() -> int:
    """Tasks (threads) of this real uid, as RLIMIT_NPROC counts them."""
    uid = os.getuid()
    n = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/status", encoding="ascii", errors="replace") as f:
                ruid = threads = None
                for line in f:
                    if line.startswith("Uid:"):
                        ruid = int(line.split()[1])
                    elif line.startswith("Threads:"):
                        threads = int(line.split()[1])
                if ruid == uid:
                    n += threads or 1
        except (OSError, ValueError, IndexError):
            continue
    return n


def _set_limits(limits: dict) -> None:
    def hard(res, value):
        resource.setrlimit(res, (value, value))

    hard(resource.RLIMIT_CPU, int(limits["cpu_s"]))
    hard(resource.RLIMIT_AS, int(limits["as_mb"]) << 20)
    hard(resource.RLIMIT_FSIZE, int(limits["fsize_mb"]) << 20)
    hard(resource.RLIMIT_NOFILE, int(limits["nofile"]))
    hard(resource.RLIMIT_CORE, 0)
    hard(resource.RLIMIT_NPROC, _uid_tasks() + int(limits["nproc_extra"]))


_DENY = ("socket.", "subprocess.", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
         "os.kill", "os.killpg", "signal.pthread_kill", "pty.spawn", "os.setuid", "os.setgid", "os.chroot")


def _install_audit_hook(workdir: str) -> None:
    wd = os.path.realpath(workdir)

    def hook(event, args):
        if event.startswith(_DENY):
            raise PermissionError(f"{event} is not allowed in the sandbox")
        if event == "open" and args:
            path, mode, flags = (list(args) + [None, None, None])[:3]
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                      (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
            if writing and isinstance(path, (str, bytes)):
                p = os.fsdecode(path)
                real = os.path.realpath(p if os.path.isabs(p) else os.path.join(wd, p))
                if real != wd and not real.startswith(wd + os.sep):
                    raise PermissionError(f"writing {p} is not allowed in the sandbox")

    sys.addaudithook(hook)


def _error_line(e: BaseException) -> str:
    line = None
    tb = e.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "<cell>":
            line = tb.tb_lineno
        tb = tb.tb_next
    where = f" (line {line})" if line else ""
    msg = str(e)
    return f"{type(e).__name__}{where}: {msg}" if msg else f"{type(e).__name__}{where}"


def _run_new_cell(code: str, g: dict, cap: _Capture) -> str:
    try:
        tree = ast.parse(code, "<cell>", "exec")
    except (SyntaxError, ValueError) as e:      # ValueError: source contains null bytes
        cap.write(f"{type(e).__name__}: {e}\n")
        return "syntax_error"
    try:
        body = tree.body
        if body and isinstance(body[-1], ast.Expr):
            head = ast.Module(body=body[:-1], type_ignores=[])
            exec(compile(head, "<cell>", "exec"), g)
            value = eval(compile(ast.Expression(body[-1].value), "<cell>", "eval"), g)
            if value is not None:
                cap.write(repr(value) + "\n")
        else:
            exec(compile(tree, "<cell>", "exec"), g)
    except BaseException as e:  # noqa: BLE001 - every failure of model code is an observation
        cap.write(_error_line(e) + "\n")
        return "error"
    return "ok"


def main() -> None:
    _set_nondumpable()
    job = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    cells, limits = job["cells"], job["limits"]
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    res_fd = os.dup(1)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    _set_limits(limits)
    random.seed(0)
    cap = _Capture(int(limits["capture_chars"]))
    sink = _Capture(0)
    g = {"__name__": "__main__", "__builtins__": builtins}
    _install_audit_hook(os.getcwd())
    status = "ok"
    try:
        for code in cells[:-1]:
            sys.stdout = sys.stderr = sink
            try:
                exec(compile(code, "<cell>", "exec"), g)
            except BaseException:  # noqa: BLE001
                pass
        sys.stdout = sys.stderr = cap
        status = _run_new_cell(cells[-1], g, cap)
    except BaseException as e:  # noqa: BLE001 - e.g. MemoryError while compiling
        status = "error"
        cap.write(_error_line(e) + "\n")
    data = json.dumps({"status": status, "output": cap.text(), "output_chars": cap.total}).encode("utf-8")
    view = memoryview(data)
    while view:
        n = os.write(res_fd, view)
        view = view[n:]
    os._exit(0)


if __name__ == "__main__":
    main()
