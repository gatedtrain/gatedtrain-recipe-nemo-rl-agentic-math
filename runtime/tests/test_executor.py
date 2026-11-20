"""Executor sidecar: limits, statuses, replay, output capture and the Unix-socket protocol (host CPU)."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from _skips import executor_python_has
from _support import AREA

import exec_protocol
import executor_server


class ExecutorLimits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="a10-exec-")
        cls.ex = executor_server.Executor(workers=4, tmp_root=cls.tmp, limits={"wall_s": 6.0})
        # warm the page cache for numpy/sympy (cold imports take tens of seconds on this disk)
        executor_server.warmup(cls.ex)

    @classmethod
    def tearDownClass(cls):
        cls.ex.pool.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run1(self, *cells, ex=None):
        return (ex or self.ex).run_jobs([{"id": "j", "cells": list(cells)}])[0]

    def test_ok_print_and_last_expression(self):
        r = self.run1("x = 2\nprint('hi')\nx * 21")
        self.assertEqual((r["status"], r["output"]), ("ok", "hi\n42\n"))

    def test_none_value_not_printed(self):
        self.assertEqual(self.run1("print(1)\nNone")["output"], "1\n")

    @unittest.skipUnless(executor_python_has("numpy", "sympy"),
                         "numpy or sympy not importable by this interpreter (the sidecar's NeMo RL image has both)")
    def test_numpy_and_sympy_available(self):
        r = self.run1("import sympy\nimport numpy\nsympy.factorint(360), int(numpy.arange(5).sum())")
        self.assertEqual(r["status"], "ok", r)
        self.assertIn("{2: 3, 3: 2, 5: 1}", r["output"])

    def test_syntax_error_is_an_observation(self):
        r = self.run1("def f(:\n  pass")
        self.assertEqual(r["status"], "syntax_error")
        self.assertIn("SyntaxError", r["output"])

    def test_runtime_error_keeps_earlier_output(self):
        r = self.run1("print('a')\n1/0")
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["output"], "a\nZeroDivisionError (line 2): division by zero\n")

    def test_wall_timeout(self):
        ex = executor_server.Executor(workers=1, tmp_root=self.tmp, limits={"wall_s": 1.0})
        t0 = time.time()
        r = self.run1("import time\ntime.sleep(30)", ex=ex)
        self.assertEqual(r["status"], "timeout")
        self.assertLess(time.time() - t0, 5)
        ex.pool.shutdown()

    def test_cpu_limit(self):
        ex = executor_server.Executor(workers=1, tmp_root=self.tmp, limits={"wall_s": 10.0, "cpu_s": 1})
        r = self.run1("while True:\n  pass", ex=ex)
        self.assertEqual(r["status"], "timeout")
        self.assertTrue(r["detail"].startswith("cpu"), r)
        ex.pool.shutdown()

    def test_memory_limit(self):
        r = self.run1("a = bytearray(3 * 1024 ** 3)")
        self.assertEqual(r["status"], "error")
        self.assertIn("MemoryError", r["output"])

    def test_file_size_limit(self):
        r = self.run1("open('big.bin', 'wb').write(b'x' * (2 << 20))")
        self.assertEqual(r["status"], "error")
        self.assertIn("File too large", r["output"])

    def test_escapes_refused(self):
        for code, word in (("import socket\nsocket.socket()", "socket"),
                           ("import subprocess\nsubprocess.run(['true'])", "subprocess"),
                           ("import os\nos.system('true')", "os.system"),
                           ("import os\nos.fork()", "os.fork"),
                           ("open('/tmp/a10-escape.txt', 'w').write('x')", "writing"),
                           ("import os\nos.kill(os.getppid(), 9)", "os.kill")):
            r = self.run1(code)
            self.assertEqual(r["status"], "error", (code, r))
            self.assertIn("PermissionError", r["output"])
            self.assertIn(word, r["output"])
        self.assertFalse(os.path.exists("/tmp/a10-escape.txt"))

    def test_local_files_allowed_and_cleaned(self):
        r = self.run1("open('n.txt', 'w').write('5')\nprint(open('n.txt').read())")
        self.assertEqual(r["output"], "5\n")
        self.assertEqual(os.listdir(self.tmp), [])           # work directories removed

    def test_huge_output_capped_and_counted(self):
        r = self.run1("print('x' * 100000)")
        self.assertEqual(r["status"], "ok")
        self.assertEqual(len(r["output"]), executor_server.DEFAULT_LIMITS["capture_chars"])
        self.assertEqual(r["output_chars"], 100001)

    def test_raw_fd_writes_cannot_corrupt_result(self):
        r = self.run1("import os\nos.write(1, b'garbage')\nos.write(2, b'more')\nprint('fine')")
        self.assertEqual((r["status"], r["output"]), ("ok", "fine\n"))

    def test_hard_exit_is_crash(self):
        r = self.run1("import os\nos._exit(3)")
        self.assertEqual(r["status"], "crash")

    def test_replay_discards_old_output_and_keeps_state(self):
        r = self.run1("import random\ny = random.randint(0, 10**9)\nprint('old')", "print(y)")
        r0 = self.run1("import random\ny = random.randint(0, 10**9)\nprint('old')\ny")
        self.assertEqual(r["status"], "ok")
        self.assertNotIn("old", r["output"])
        self.assertEqual(r["output"].strip(), r0["output"].split("\n")[1].strip())   # same seed, same value

    def test_replay_ignores_errors_of_earlier_cells(self):
        r = self.run1("a = 1\n1/0\nb = 2", "print(a)\nprint('b' in globals())")
        self.assertEqual(r["output"], "1\nFalse\n")

    def test_jobs_in_order_and_ids_unique(self):
        res = self.ex.run_jobs([{"id": str(i), "cells": [f"{i} * 2"]} for i in range(12)])
        self.assertEqual([r["output"] for r in res], [f"{i * 2}\n" for i in range(12)])
        with self.assertRaises(ValueError):
            self.ex.run_jobs([{"id": "a", "cells": ["1"]}, {"id": "a", "cells": ["2"]}])


class ExecutorSocket(unittest.TestCase):
    def test_serve_ping_run_and_protocol_errors(self):
        tmp = tempfile.mkdtemp(prefix="a10-spool-")
        sock = os.path.join(tmp, "mathexec.sock")
        proc = subprocess.Popen([sys.executable, "-I", str(AREA / "executor" / "executor_server.py"), "--serve", sock,
                                 "--workers", "2", "--no-warmup"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            info = None
            for _ in range(100):
                try:
                    info = exec_protocol.ping(sock, timeout=2)
                    break
                except exec_protocol.ExecutorUnavailable:
                    time.sleep(0.1)
            self.assertIsNotNone(info)
            self.assertEqual(info["limits"]["wall_s"], 10.0)
            self.assertEqual(set(info["code_sha256"]), {"executor_server.py", "sandbox_runner.py", "exec_protocol.py"})
            self.assertEqual(oct(os.stat(sock).st_mode & 0o777), "0o700")
            res = exec_protocol.call_executor(sock, [{"id": "a", "cells": ["print(1 + 1)"]},
                                                     {"id": "b", "cells": ["x ="]}], timeout=60)
            self.assertEqual([r["status"] for r in res], ["ok", "syntax_error"])
            with self.assertRaises(exec_protocol.ExecutorUnavailable):
                exec_protocol.call_executor(sock, [{"id": "a", "cells": []}], timeout=60)   # protocol error
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:              # unknown op
                s.connect(sock)
                exec_protocol.send_msg(s, {"op": "nope"})
                self.assertIn("unknown op", exec_protocol.recv_msg(s)["error"])
        finally:
            proc.terminate()
            out, _ = proc.communicate(timeout=10)
            shutil.rmtree(tmp, ignore_errors=True)
        events = [json.loads(x)["event"] for x in out.decode().splitlines() if x.startswith("{")]
        self.assertEqual(events[0], "listening")
        self.assertIn("batch", events)

    def test_unreachable_socket_raises(self):
        with self.assertRaises(exec_protocol.ExecutorUnavailable):
            exec_protocol.call_executor("/nonexistent/mathexec.sock", [{"id": "a", "cells": ["1"]}], timeout=1)


# A job that leaves the audit hook through ctypes: fork, a new session (out of the job's process group), exec of a
# fresh interpreter without the hook, which replaces the executor's socket with its own and answers every later
# request with forged results.
IMPOSTOR = r"""
import json, os, signal, socket, struct, sys
signal.alarm(30)
path = sys.argv[1]
open(os.path.join(os.path.dirname(path), "impostor.pid"), "w").write(str(os.getpid()))
os.unlink(path)
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(path)
srv.listen(4)
while True:
    c, _ = srv.accept()
    n = struct.unpack(">Q", c.recv(8, socket.MSG_WAITALL))[0]
    req = json.loads(c.recv(n, socket.MSG_WAITALL))
    resp = {"ok": True} if req.get("op") == "ping" else {"results": [
        {"id": j["id"], "status": "ok", "output": "FORGED\n", "output_chars": 7, "wall_s": 0.0} for j in req["jobs"]]}
    data = json.dumps(resp).encode()
    c.sendall(struct.pack(">Q", len(data)) + data)
    c.close()
"""

IMPERSONATE = """
import ctypes, os, sys, time
path = {path!r}
before = os.stat(path).st_ino
libc = ctypes.CDLL(None, use_errno=True)
if libc.fork() == 0:
    libc.setsid()
    argv = [sys.executable, "-I", "-c", {impostor!r}, path]
    arr = (ctypes.c_char_p * (len(argv) + 1))(*[a.encode() for a in argv], None)
    libc.execv(sys.executable.encode(), arr)
    os._exit(1)
for _ in range(100):
    try:
        if os.stat(path).st_ino != before:
            break
    except FileNotFoundError:
        pass
    time.sleep(0.05)
print("planted")
"""

# A job that writes into a concurrently running sibling job's result channel, found through /proc: an unterminated
# frame first, the closing brace after the sibling has written its own frame (which becomes a nested value).
SIBLING_FORGER = """
import ctypes, os, time
libc = ctypes.CDLL(None, use_errno=True)       # open(2) without the audit hook's check on writes
me, parent = os.getpid(), os.getppid()
fds = []
for _ in range(40):
    for pid in filter(str.isdigit, os.listdir("/proc")):
        if int(pid) == me:
            continue
        try:
            if int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1]) != parent:
                continue
            for fd in os.listdir(f"/proc/{pid}/fd"):
                if os.readlink(f"/proc/{pid}/fd/{fd}").startswith(("pipe:", "socket:")):
                    n = libc.open(f"/proc/{pid}/fd/{fd}".encode(), os.O_WRONLY)
                    if n >= 0:
                        fds.append(n)
        except (OSError, ValueError, IndexError):
            continue
    if fds:
        break
    time.sleep(0.05)
for fd in fds:
    os.write(fd, b'{"status": "ok", "output": "FORGED\\\\n", "output_chars": 7, "x": ')
time.sleep(3)
for fd in fds:
    os.write(fd, b"}")
print("forger opened", len(fds))
"""


class ExecutorTrustBoundary(unittest.TestCase):
    """M2: model code must not reach anything through which it could speak the executor protocol or alter results,
    whatever Python-level restriction it bypasses (here with ctypes)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="a10-trust-")
        self.sock = os.path.join(self.tmp, "mathexec.sock")
        self.proc = subprocess.Popen([sys.executable, "-I", str(AREA / "executor" / "executor_server.py"), "--serve",
                                      self.sock, "--workers", "2", "--no-warmup"], stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        for _ in range(100):
            try:
                exec_protocol.ping(self.sock, timeout=2)
                break
            except exec_protocol.ExecutorUnavailable:
                time.sleep(0.1)

    def impostor_pid(self):
        try:
            with open(os.path.join(self.tmp, "impostor.pid"), encoding="ascii") as f:
                return int(f.read())
        except (OSError, ValueError):
            return None

    def tearDown(self):
        pid = self.impostor_pid()
        if pid is not None:
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        self.proc.kill()
        self.proc.communicate(timeout=10)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_job_cannot_impersonate_the_executor_socket(self):
        attack = IMPERSONATE.format(path=self.sock, impostor=IMPOSTOR)
        try:
            res = exec_protocol.call_executor(self.sock, [{"id": "a", "cells": [attack]}], timeout=60)
            self.assertNotEqual(res[0]["output"], "FORGED\n")
        except exec_protocol.ExecutorUnavailable as e:      # the executor noticed and refused to go on
            self.assertIn("was replaced", str(e))
        pid = self.impostor_pid()
        self.assertIsNotNone(pid, "the attack did not run")
        self.assertFalse(os.path.exists(f"/proc/{pid}"), "a process forked by a job outlived its batch")
        try:
            res = exec_protocol.call_executor(self.sock, [{"id": "b", "cells": ["print(1 + 1)"]}], timeout=60)
        except exec_protocol.ExecutorUnavailable:
            return                                          # nobody answers: a loud failure, never a forged result
        self.assertEqual((res[0]["status"], res[0]["output"]), ("ok", "2\n"), "a job's process answered for the executor")

    def test_job_cannot_write_into_a_sibling_result(self):
        res = exec_protocol.call_executor(self.sock, [{"id": "forger", "cells": [SIBLING_FORGER]},
                                                      {"id": "victim", "cells": ["import time\ntime.sleep(1)\n"
                                                                                 "print('real')"]}], timeout=60)
        self.assertEqual(res[1]["output"], "real\n", res)

    def test_job_cannot_reach_the_executor_file_descriptors(self):
        res = exec_protocol.call_executor(self.sock, [{"id": "a", "cells": [
            "import os\nd = f'/proc/{os.getppid()}/fd'\nsorted(os.readlink(f'{d}/{n}') for n in os.listdir(d))"]}],
            timeout=60)
        self.assertEqual(res[0]["status"], "error", res)
        self.assertIn("PermissionError", res[0]["output"])


if __name__ == "__main__":
    unittest.main()
