#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# T4 GPU smoke: controls on the longest trace of one model, on one GPU.
# usage: t4_smoke.sh <root> <python> <gpu index> <model dir> <tag>
# Writes smoke/t4_<tag>.smoke.json and, when every control passed, smoke/PASS_<tag> (FREEZE.md5 line).
# T4_STACK (optional): stack root for the simulation modules if it differs from <root> (pro5000: a md5-identical
# copy whose path s0_cov.py hard-codes). Stop: kill -- -<PGID> (never pkill -f).
set -u
R=$1; PY=$2; GPU=$3; M=$4; TAG=$5
cd "$R" || exit 2
mkdir -p logs smoke
export PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
LOG=logs/smoke_$TAG.log
echo "$(date -u +%FT%TZ) smoke $TAG start gpu $GPU pid $$ pgid $(ps -o pgid= $$ | tr -d ' ')" >> "$LOG"
if ! md5sum -c --quiet FREEZE.md5 >> "$LOG" 2>&1; then echo "$(date -u +%FT%TZ) REFUSED: FREEZE.md5 mismatch" >> "$LOG"; exit 3; fi
F0=$(md5sum FREEZE.md5)
LOCKDIR=${T4_LOCKDIR:-$HOME/gpu_queue}
mkdir -p "$LOCKDIR"
exec 9> "$LOCKDIR/gpu$GPU.lock"
if ! flock -n 9; then echo "$(date -u +%FT%TZ) REFUSED: lock $LOCKDIR/gpu$GPU.lock held" >> "$LOG"; exit 4; fi
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
BUSY=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep -c "$UUID")
if [ "$BUSY" -ne 0 ]; then echo "$(date -u +%FT%TZ) REFUSED: compute process on GPU $GPU" >> "$LOG"; exit 5; fi
OUT=smoke/t4_$TAG.smoke.json
rm -f "$OUT" "smoke/PASS_$TAG"
$PY t4/t4_lab.py "$M" "nab1/out/traces_$TAG.jsonl" "$OUT" --stack "${T4_STACK:-$R}" --smoke >> "$LOG" 2>&1
RC=$?
echo "$(date -u +%FT%TZ) smoke $TAG rc=$RC" >> "$LOG"
if [ "$RC" -eq 0 ] && [ "$(md5sum FREEZE.md5)" = "$F0" ] && md5sum -c --quiet FREEZE.md5 >> "$LOG" 2>&1; then
  echo "$F0" > "smoke/PASS_$TAG.tmp" && sync && mv "smoke/PASS_$TAG.tmp" "smoke/PASS_$TAG"
  echo "$(date -u +%FT%TZ) SMOKE PASS $TAG" >> "$LOG"
fi
exit $RC
