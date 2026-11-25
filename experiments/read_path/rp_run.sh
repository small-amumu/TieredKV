#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
# RP run (PLAN_RP.md): wait for the GPU lock and an idle card, then the preflight (B 8, L 32768, thp), then the
# full sweep. Every step's rc goes to <run_dir>/run.log; the md5 of the scripts is logged before the first step.
# The python children inherit the GPU lock fd on purpose: the lock stays held while a child still uses the card.
# usage: setsid nohup bash rp_run.sh <run_dir> <gpu index> > /dev/null 2>&1 < /dev/null &
set -u
FAIL=$HOME/rp_run_fail.log
R=${1:-}
GPU=${2:-}
if [ -z "$R" ] || [ -z "$GPU" ]; then echo "$(date -u +%FT%TZ) usage: rp_run.sh <run_dir> <gpu>" >> "$FAIL"; exit 2; fi
case $GPU in ''|*[!0-9]*) echo "$(date -u +%FT%TZ) gpu index must be a number: $GPU" >> "$FAIL"; exit 2;; esac
X=$HOME/work/rp
PY=$HOME/work/sekv_cmp/venv/bin/python
LOG=$R/run.log
if ! mkdir -p "$R"; then echo "$(date -u +%FT%TZ) cannot create $R" >> "$FAIL"; exit 2; fi
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
trap 'say "KILLED (signal)"; exit 130' TERM INT HUP
if ! exec 8>> "$R/.run.lock"; then echo "$(date -u +%FT%TZ) cannot open $R/.run.lock" >> "$FAIL"; exit 2; fi
if ! flock -n 8; then say "REFUSED: another rp_run.sh holds $R/.run.lock"; exit 3; fi
for f in "$X/rp_probe.py" "$X/pcie_meter.py" "$X/rp_run.sh" "$PY"; do
    if [ ! -e "$f" ]; then say "STOP: missing $f"; exit 2; fi
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU
say "START pid $$ pgid $(ps -o pgid= $$ | tr -d ' ') gpu $GPU; waiting for $HOME/gpu_queue/gpu$GPU.lock"
mkdir -p "$HOME/gpu_queue"
if ! exec 9>> "$HOME/gpu_queue/gpu$GPU.lock"; then say "STOP: cannot open the GPU lock"; exit 4; fi
# non-blocking attempts every 60 s, so the signal trap can run and the wait is logged
kk=0
until flock -n 9; do
    if [ $((kk % 30)) -eq 0 ]; then say "waiting for the GPU lock ($kk min)"; fi
    kk=$((kk + 1))
    sleep 60
done
say "lock held; waiting until GPU $GPU has no compute process (nvidia-smi failures count as busy)"
k=0
while true; do
    n=$(timeout 30 nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    rc=$?          # 124 = nvidia-smi hung (timeout): counts as busy
    if [ $rc -eq 0 ] && [ -z "$n" ]; then break; fi
    k=$((k + 1))
    if [ $((k % 30)) -eq 1 ]; then say "busy (rc=$rc): $(echo "$n" | tr '\n' ' ' | cut -c1-200)"; fi
    sleep 60
done
m=$(md5sum "$X/rp_probe.py" "$X/pcie_meter.py" "$X/rp_run.sh")
rc=$?
if [ $rc -ne 0 ]; then say "STOP: md5sum failed rc=$rc"; exit 2; fi
say "card idle; md5 $(echo "$m" | awk '{print $2 ":" $1}' | tr '\n' ' ')"
"$PY" -B "$X/rp_probe.py" "$R/preflight.jsonl" --preflight > "$R/preflight.out" 2>&1
rc=$?
say "preflight rc=$rc"
if [ $rc -ne 0 ]; then say "STOP: preflight failed"; exit 1; fi
say "full sweep start"
"$PY" -B "$X/rp_probe.py" "$R/rp_full.jsonl" > "$R/rp_full.out" 2>&1
rc=$?
say "full rc=$rc"
say "END"
