#!/bin/bash
# Prints (never runs) the docker commands of the recipe's executor sidecar (runtime/README.md section 0.5, "Executor
# sidecar"). Run the printed commands yourself, in order.
#
#   run-executor.sh serve <spool_dir> <label>   sidecar for one trainer or one evaluation-generation container:
#                                              Unix socket <spool_dir>/mathexec.sock, container gatedtrain-mathexec-<label>
#                                              (unique per launch; the printed stop names it)
#
# The sidecar runs model-written Python. It mounts ONLY <runtime>/executor (executor_server.py, sandbox_runner.py,
# exec_protocol.py; read-only) and the spool (read-write): never the runtime root (items-test/ and data/ hold the
# answers), never models/, never a run's /out (MM-34a: generated code must not be able to read reference data). The
# paired trainer or generator mounts the same spool at /spool and the runtime root read-only; it never runs generated
# code itself.
# Flags as the CAD recipe's executor ("B18"): --network none, uid 1000, no capabilities, no-new-privileges, read-only
# root, tmpfs /tmp (noexec), CPU / memory / swap / pids / open-file limits, no core dumps.
# The runtime directory is this script's parent directory; it is mounted at its own host path.
# Image: gatedtrain's NeMo RL image for uid 1000 (`gatedtrain-a6-nemo-rl:1`, built from gatedtrain's containers/nemo-rl-uid1000/ and pinned
# by its local image ID, gatedtrain.families.nemo_rl NEMO_RL_UID1000_IMAGE); its /opt/nemo_rl_venv/bin/python is Python
# 3.13 with numpy and sympy for the model's code. GATEDTRAIN_MATHEXEC_IMAGE / GATEDTRAIN_MATHEXEC_PYTHON override both (for example
# a standard-library-only executor image with a host CPython tree mounted read-only; add -v for that tree yourself).
# The commands use the Docker context `gatedtrain`, as gatedtrain's launcher prints them.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
AREA=$(dirname "$HERE")
EXE=$AREA/executor
IMAGE=${GATEDTRAIN_MATHEXEC_IMAGE:-sha256:7a4078e70b28cc6de622fa3cf25c7596c52337fa3f1779488b402c5fa1c74e69}
PY=${GATEDTRAIN_MATHEXEC_PYTHON:-/opt/nemo_rl_venv/bin/python}
B18=(--network none --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges --read-only
  --tmpfs /tmp:rw,nosuid,nodev,noexec,size=4g --cpus 8 --memory 8g --memory-swap 8g --pids-limit 256
  --ulimit nofile=1024:1024 --ulimit core=0
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1)
case "${1:-}" in
  serve)
    SPOOL=${2:?spool dir (new, empty, owned by uid 1000)}; LABEL=${3:?label (e.g. mm-a10-agentic-p1-a)}
    [[ "$LABEL" =~ ^[a-z0-9][a-z0-9_.-]*$ ]] || { echo "label must match [a-z0-9][a-z0-9_.-]*" >&2; exit 2; }
    [[ "$SPOOL" = /* ]] || { echo "spool dir must be absolute" >&2; exit 2; }
    case "$SPOOL/" in "$AREA"/*) echo "the spool must not be inside the area" >&2; exit 2 ;; esac
    NAME=gatedtrain-mathexec-$LABEL
    printf '%q ' docker --context gatedtrain run -d --name "$NAME" "${B18[@]}" -v "$EXE:$EXE:ro" -v "$SPOOL:/spool:rw" \
      "$IMAGE" "$PY" -I "$EXE/executor_server.py" --serve /spool/mathexec.sock --workers 8; echo
    printf '%q ' docker --context gatedtrain logs "$NAME"; echo "   # 'listening' after the warm-up import"
    printf '%q ' docker --context gatedtrain stop "$NAME"; echo "   # after the paired container has exited"
    printf '%q ' docker --context gatedtrain rm "$NAME"; echo ;;
  *) echo "usage: $0 serve <spool_dir> <label>" >&2; exit 2 ;;
esac
