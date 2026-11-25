#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
# T4 A3 launcher (check or generation) for one model on one GPU.
# usage: t4_gen_run.sh <a3 dir> <stack root> <python> <gpu index> <model dir> <name> <t4_gen args...>
# Writes logs/<name>.log (start line with pid/pgid, the program output, an rc= line). Stop: kill -- -<PGID>.
set -u
if [ $# -lt 7 ]; then echo "usage: t4_gen_run.sh a3dir stack python gpu model name args..." >&2; exit 2; fi
D=$1; R=$2; PY=$3; GPU=$4; M=$5; NAME=$6; shift 6
cd "$D" || { echo "$(date -u +%FT%TZ) t4_gen_run $NAME: cannot cd $D" >> "$HOME/t4_launch_fail.log"; exit 2; }
mkdir -p logs out
export PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
LOG=logs/$NAME.log
trap 'echo "$(date -u +%FT%TZ) $NAME killed by signal" >> "$LOG"; exit 143' TERM INT HUP
echo "$(date -u +%FT%TZ) $NAME start gpu $GPU pid $$ pgid $(ps -o pgid= $$ | tr -d ' ') args $*" >> "$LOG"
LOCKDIR=${T4_LOCKDIR:-$HOME/gpu_queue}
mkdir -p "$LOCKDIR"
exec 9> "$LOCKDIR/gpu$GPU.lock"
if ! flock -n 9; then echo "$(date -u +%FT%TZ) REFUSED: lock $LOCKDIR/gpu$GPU.lock held" >> "$LOG"; exit 4; fi
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
if [ -z "$UUID" ]; then echo "$(date -u +%FT%TZ) REFUSED: no uuid for GPU $GPU" >> "$LOG"; exit 5; fi
BUSY=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep -c "$UUID")
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
if [ "$BUSY" -ne 0 ] || [ "${USED:-99999}" -gt 1024 ]; then
  echo "$(date -u +%FT%TZ) REFUSED: GPU $GPU busy (procs $BUSY, used ${USED} MiB)" >> "$LOG"; exit 5
fi
md5sum t4_gen.py t4_q.py t4_lab.py >> "$LOG"
$PY -u t4_gen.py "$M" "$@" --stack "$R" >> "$LOG" 2>&1
RC=$?
echo "$(date -u +%FT%TZ) $NAME rc=$RC" >> "$LOG"
exit $RC
