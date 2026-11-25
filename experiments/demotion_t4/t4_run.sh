#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
# T4 run (A1 diagnostic or A2 full) of one model on one GPU; every arm of the model runs in this one process.
# usage: t4_run.sh <root> <python> <gpu index> <model dir> <tag> <name> <arms,comma> [--diag8 | --items i,j,...]
# Needs smoke/PASS_<tag> whose content equals the current FREEZE.md5 line (smoke passed on this exact code) and
# smoke/t4_<tag>.smoke.json written for the same model directory.
# Writes out/t4_<name>.json (resumable: rerun the exact same command) and logs/run_<name>.log.
# T4_STACK (optional): stack root for the simulation modules if it differs from <root> (pro5000: a md5-identical
# copy whose path s0_cov.py hard-codes). Stop: kill -- -<PGID> (never pkill -f).
set -u
if [ $# -lt 7 ] || [ -z "$7" ]; then echo "usage: t4_run.sh root python gpu model tag name arms [--diag8|--items ...]" >&2; exit 2; fi
R=$1; PY=$2; GPU=$3; M=$4; TAG=$5; NAME=$6; RUNS=$7; shift 7
for a in "$@"; do
  case "$a" in --runs*|--stack*|--smoke*|--max_len*|--chunk*) echo "t4_run.sh: $a is fixed by the launcher" >&2; exit 2;; esac
done
cd "$R" || { echo "$(date -u +%FT%TZ) t4_run $NAME: cannot cd $R" >> "$HOME/t4_launch_fail.log"; exit 2; }
mkdir -p logs out
export PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
LOG=logs/run_$NAME.log
trap 'echo "$(date -u +%FT%TZ) run $NAME killed by signal" >> "$LOG"; exit 143' TERM INT HUP
echo "$(date -u +%FT%TZ) run $NAME start tag $TAG gpu $GPU pid $$ pgid $(ps -o pgid= $$ | tr -d ' ') arms $RUNS $*" >> "$LOG"
F0=$(md5sum FREEZE.md5)
if ! md5sum -c --quiet FREEZE.md5 >> "$LOG" 2>&1; then echo "$(date -u +%FT%TZ) REFUSED: FREEZE.md5 mismatch" >> "$LOG"; exit 3; fi
# the item selection and arms are fixed per output name, so a resume can never mix selections
ARGS="model=$M tag=$TAG arms=$RUNS extra=$*"
if [ -f "out/t4_$NAME.args" ] && [ "$(cat "out/t4_$NAME.args")" != "$ARGS" ]; then
  echo "$(date -u +%FT%TZ) REFUSED: out/t4_$NAME.args differs from this command" >> "$LOG"; exit 7
fi
LOCKDIR=${T4_LOCKDIR:-$HOME/gpu_queue}
mkdir -p "$LOCKDIR"
exec 9> "$LOCKDIR/gpu$GPU.lock"
if ! flock -n 9; then echo "$(date -u +%FT%TZ) REFUSED: lock $LOCKDIR/gpu$GPU.lock held" >> "$LOG"; exit 4; fi
if [ ! -f "smoke/PASS_$TAG" ] || [ "$(cat "smoke/PASS_$TAG")" != "$F0" ]; then
  echo "$(date -u +%FT%TZ) REFUSED: no smoke pass for $TAG on the current FREEZE" >> "$LOG"; exit 6
fi
if ! grep -qF "\"model\": \"$M\"" "smoke/t4_$TAG.smoke.json"; then
  echo "$(date -u +%FT%TZ) REFUSED: smoke for $TAG was not run on $M" >> "$LOG"; exit 6
fi
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
if [ -z "$UUID" ]; then echo "$(date -u +%FT%TZ) REFUSED: no uuid for GPU $GPU" >> "$LOG"; exit 5; fi
BUSY=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep -c "$UUID")
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
if [ "$BUSY" -ne 0 ] || [ "${USED:-99999}" -gt 1024 ]; then
  echo "$(date -u +%FT%TZ) REFUSED: GPU $GPU busy (procs $BUSY, used ${USED} MiB)" >> "$LOG"; exit 5
fi
echo "$ARGS" > "out/t4_$NAME.args"
$PY -u t4/t4_lab.py "$M" "nab1/out/traces_$TAG.jsonl" "out/t4_$NAME.json" --stack "${T4_STACK:-$R}" --runs "$RUNS" "$@" >> "$LOG" 2>&1
RC=$?
echo "$(date -u +%FT%TZ) run $NAME rc=$RC" >> "$LOG"
exit $RC
