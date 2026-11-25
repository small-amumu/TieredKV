#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
# T3 run (PLAN_RP.md section 4). phase cpu: CPU throughput sweep, no GPU context (CUDA_VISIBLE_DEVICES empty),
# nice 10. phase gpu: numeric check + GPU<->CPU round trips, after the GPU lock and an idle card (as rp_run.sh).
# Every rc goes to <run_dir>/run_<phase>.log.
# usage: setsid nohup bash rp_t3_run.sh <run_dir> <cpu|gpu> <gpu index> > /dev/null 2>&1 < /dev/null &
set -u
FAIL=$HOME/rp_run_fail.log
R=${1:-}
PH=${2:-}
GPU=${3:-}
if [ -z "$R" ] || [ -z "$PH" ] || [ -z "$GPU" ]; then
    echo "$(date -u +%FT%TZ) usage: rp_t3_run.sh <run_dir> <cpu|gpu> <gpu>" >> "$FAIL"; exit 2
fi
case $PH in cpu|gpu) ;; *) echo "$(date -u +%FT%TZ) bad phase $PH" >> "$FAIL"; exit 2;; esac
case $GPU in ''|*[!0-9]*) echo "$(date -u +%FT%TZ) gpu index must be a number: $GPU" >> "$FAIL"; exit 2;; esac
X=$HOME/work/rp
PY=$HOME/work/sekv_cmp/venv/bin/python
if ! mkdir -p "$R"; then echo "$(date -u +%FT%TZ) cannot create $R" >> "$FAIL"; exit 2; fi
LOG=$R/run_$PH.log
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
trap 'say "KILLED (signal)"; exit 130' TERM INT
if ! exec 8>> "$R/.run_$PH.lock"; then echo "$(date -u +%FT%TZ) cannot open $R/.run_$PH.lock" >> "$FAIL"; exit 2; fi
if ! flock -n 8; then say "REFUSED: another rp_t3_run.sh $PH holds the run lock"; exit 3; fi
for f in "$X/rp_cpu.py" "$X/rp_cpu_ext.cpp" "$X/rp_t3_run.sh" "$PY"; do
    if [ ! -e "$f" ]; then say "STOP: missing $f"; exit 2; fi
done
say "START phase $PH pid $$ pgid $(ps -o pgid= $$ | tr -d ' ')"
# wait until the GPU has no compute process (nvidia-smi failure or timeout counts as busy)
wait_idle() {
    local k=0 n rc
    while true; do
        n=$(timeout 30 nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
        rc=$?
        if [ $rc -eq 0 ] && [ -z "$n" ]; then return 0; fi
        k=$((k + 1))
        if [ $((k % 30)) -eq 1 ]; then say "busy (rc=$rc): $(echo "$n" | tr '\n' ' ' | cut -c1-200)"; fi
        sleep 60
    done
}
if [ "$PH" = "gpu" ]; then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES=$GPU
    mkdir -p "$HOME/gpu_queue"
    if ! exec 9>> "$HOME/gpu_queue/gpu$GPU.lock"; then say "STOP: cannot open the GPU lock"; exit 4; fi
    kk=0
    until flock -n 9; do
        if [ $((kk % 30)) -eq 0 ]; then say "waiting for the GPU lock ($kk min)"; fi
        kk=$((kk + 1))
        sleep 60
    done
    say "lock held; waiting until GPU $GPU has no compute process"
    wait_idle
    MODES=numeric,rtt
    NICE=0
else
    export CUDA_VISIBLE_DEVICES=
    MODES=cpu
    NICE=10
fi
# host-wide lock: the CPU timing sweep and the GPU phase (its responder spins on a CPU) never overlap
if ! exec 7>> "$HOME/rp_cpu_timing.lock"; then say "STOP: cannot open the CPU timing lock"; exit 4; fi
kc=0
until flock -n 7; do
    if [ $((kc % 30)) -eq 0 ]; then say "waiting for the CPU timing lock ($kc min)"; fi
    kc=$((kc + 1))
    sleep 60
done
say "CPU timing lock held"
if [ "$PH" = "gpu" ]; then
    say "re-checking that GPU $GPU is idle after the CPU timing lock"
    wait_idle
fi
m=$(md5sum "$X/rp_cpu.py" "$X/rp_cpu_ext.cpp" "$X/rp_t3_run.sh")
rc=$?
if [ $rc -ne 0 ]; then say "STOP: md5sum failed rc=$rc"; exit 2; fi
say "md5 $(echo "$m" | awk '{print $2 ":" $1}' | tr '\n' ' '); loadavg $(cat /proc/loadavg)"
nice -n $NICE "$PY" -B "$X/rp_cpu.py" "$R/t3_$PH.jsonl" --modes $MODES > "$R/t3_$PH.out" 2>&1
rc=$?
say "t3 $PH rc=$rc"
say "END"
