#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
# STAR night chain (2026-11-24).
#   1. wait until every given pid (the OPX accuracy chain and its acc script) has exited;
#   2. OPX timing phase, then host-memory phase: opx_formal_star.sh tim 1, then mem 1,
#      unchanged (PLAN_OPX.md section 6); each OPX arm itself waits for an idle host;
#   3. RP read-path microbenchmark on GPU 1 (rp_run.sh, unchanged), on all CPUs of
#      GPU 1's NUMA node;
#   4. T3 cpu phase, then T3 gpu phase on GPU 1, both pinned to one logical CPU per
#      physical core of that node (rp_t3_run.sh, unchanged).
# The RP / T3 runners (written for lab05) derive every path from $HOME (scripts, python, GPU
# and CPU locks, build caches), so they run with HOME=$H (~/work, the only directory we use on
# this shared account). $H/work/sekv_cmp/venv is a symlink to $H/venv; $H/venv/bin goes
# first on PATH (ninja for the C++ extension build). The OPX script keeps the real HOME: it
# sets H=$HOME/work itself, as in its calib / acc phases. Steps 3 and 4 each start only when
# no GPU of the host has a compute process. Every step is logged with its rc to
# $H/star_night_chain.log; a failed step does not stop the later ones. Children run with the
# chain-lock fd closed, so an orphaned child cannot keep the chain lock (only the chain's own
# short sleeps can, for at most 2 min after it dies). During each timing step a sampler
# writes host CPU load every 30 s to $H/star_night_load.log.
# usage: setsid nohup bash star_night_chain.sh <pid> [pid ...] > /dev/null 2>&1 < /dev/null &
# stop:  kill -TERM -- -<PGID of the chain> (the START line gives it)
set -u
H=/home/user/work
X=$H/work/rp
LOG=$H/star_night_chain.log
LOADLOG=$H/star_night_load.log
GPU=1
SAMP=
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
stop_sampler() {
    if [ -n "$SAMP" ]; then kill "$SAMP" 2>/dev/null; wait "$SAMP" 2>/dev/null; SAMP=; fi
}
trap 'stop_sampler; say "KILLED (signal)"; exit 130' TERM INT
if [ $# -lt 1 ]; then say "usage: star_night_chain.sh <pid> [pid ...]"; exit 2; fi
for p in "$@"; do
    case $p in ''|*[!0-9]*) say "bad pid '$p'"; exit 2;; esac
done
if ! exec 8>> "$H/.star_night_chain.lock"; then say "STOP: cannot open the chain lock"; exit 2; fi
if ! flock -n 8; then say "REFUSED: another star_night_chain.sh holds the chain lock"; exit 3; fi
for f in "$H/opx/opx_formal_star.sh" "$X/rp_run.sh" "$X/rp_probe.py" "$X/pcie_meter.py" \
         "$X/rp_t3_run.sh" "$X/rp_cpu.py" "$X/rp_cpu_ext.cpp" \
         "$H/work/sekv_cmp/venv/bin/python" "$H/venv/bin/ninja"; do
    if [ ! -e "$f" ]; then say "STOP: missing $f"; exit 2; fi
done
# NUMA node of GPU 1 (PCI bus order, as CUDA_DEVICE_ORDER=PCI_BUS_ID in the runners), its CPUs;
# up to 10 tries, since one nvidia-smi failure should not lose the night
NODE=
for try in 1 2 3 4 5 6 7 8 9 10; do
    BUS=$(timeout 30 nvidia-smi -i $GPU --query-gpu=pci.bus_id --format=csv,noheader \
          2>/dev/null | tr 'A-F' 'a-f' | tr -d ' ')
    NODE=$(cat "/sys/bus/pci/devices/${BUS:4}/numa_node" 2>/dev/null)
    case $NODE in ''|*[!0-9]*) NODE=; say "NUMA lookup try $try failed (bus '$BUS')"; sleep 60;;
                  *) break;; esac
done
if [ -z "$NODE" ]; then say "STOP: no NUMA node for GPU $GPU"; exit 2; fi
NALL=$(cat "/sys/devices/system/node/node$NODE/cpulist")
NPHYS=$(lscpu -p=CPU,CORE,NODE | awk -F, -v n="$NODE" \
        '!/^#/ && $3 == n && !s[$2]++ {printf "%s%s", (k++ ? "," : ""), $1}')
if [ -z "$NALL" ] || [ -z "$NPHYS" ]; then say "STOP: empty cpu list for node $NODE"; exit 2; fi
say "START pid $$ pgid $(ps -o pgid= $$ | tr -d ' '); GPU $GPU bus $BUS node $NODE" \
    "cpus $NALL; physical $NPHYS; waiting for pids $*"
wait_host_idle() {
    local k=0 n rc
    while true; do
        n=$(timeout 30 nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
        rc=$?          # nvidia-smi failure or hang counts as busy
        if [ $rc -eq 0 ] && [ -z "$n" ]; then return 0; fi
        if [ $((k % 30)) -eq 0 ]; then
            say "host busy (rc=$rc): $(echo "$n" | tr '\n' ' ' | cut -c1-200)"
        fi
        k=$((k + 1))
        sleep 60
    done
}
# CPU-load evidence during each timing step (OPX tim / mem, RP, T3): other users' CPU jobs on
# this shared host are invisible to the GPU idle check. Every 30 s: the cumulative CPU counters
# (first line of /proc/stat, to difference later), loadavg, and the 5 busiest processes of
# top's second frame (current CPU, not lifetime). No nvidia-smi here: OPX and rp_probe record
# the GPU processes themselves, and their NVML sampling should not be disturbed.
sampler() {
    while kill -0 $$ 2>/dev/null; do      # $$ = the chain: stop if it dies without its trap
        echo "$(date -u +%FT%TZ) $1 | $(head -n 1 /proc/stat) | $(cat /proc/loadavg) |" \
             "$(top -b -n 2 -d 1 -w 200 2>/dev/null | awk '/^ *PID /{f++; next}
                f == 2 && n < 5 {printf "%s %s %s %s; ", $1, $2, $9, $12; n++}')"
        sleep 30
    done >> "$LOADLOG" 2>&1
}
start_sampler() { sampler "$1" 8>&- & SAMP=$!; say "load sampler pid $SAMP for $1"; }
alive=1
w=0
while [ $alive -eq 1 ]; do
    alive=0
    for p in "$@"; do
        if kill -0 "$p" 2>/dev/null; then alive=1; fi
    done
    if [ $alive -eq 1 ]; then
        if [ $((w % 15)) -eq 0 ]; then say "still waiting for pids $* ($((w * 2)) min)"; fi
        w=$((w + 1))
        sleep 120
    fi
done
say "waited pids gone; acc done.log tail:" \
    "$(tail -n 2 "$H/opx/formal/acc/done.log" 2>&1 | tr '\n' ' ')"
OPXLOCK=/tmp/opx_formal_star.lock      # the host lock opx_formal_star.sh takes with flock -n
for ph in tim mem; do
    # an orphaned child of the previous phase may still hold the OPX host lock: wait for it
    k=0
    until flock -n "$OPXLOCK" true 8>&-; do
        if [ $((k % 30)) -eq 0 ]; then
            say "OPX host lock busy: $(fuser "$OPXLOCK" 2>&1 | tr '\n' ' ' | cut -c1-120)"
        fi
        k=$((k + 1))
        sleep 60
    done
    say "OPX $ph start"
    start_sampler "OPX $ph"
    ( cd "$H/opx" && bash opx_formal_star.sh "$ph" $GPU ) 8>&-
    rc=$?
    stop_sampler
    say "OPX $ph rc=$rc; done.log tail: $(tail -n 1 "$H/opx/formal/$ph/done.log" 2>&1)"
    if [ $rc -ne 0 ]; then
        say "OPX launch.log tail: $(tail -n 2 "$H/opx/formal/launch.log" 2>&1 | tr '\n' ' ')"
    fi
done
wait_host_idle
say "RP start (GPU $GPU, cpus $NALL)"
start_sampler RP
HOME=$H PATH=$H/venv/bin:$PATH taskset -c "$NALL" \
    bash "$X/rp_run.sh" "$X/run_star" $GPU 8>&-
rc=$?
stop_sampler
say "RP rc=$rc; run.log tail: $(tail -n 1 "$X/run_star/run.log" 2>&1)"
# T3: both phases on one logical CPU per physical core of the node, so threads never share a
# core; OpenMP binding is left at its default, as on lab05 (torch and the extension each load
# their own libgomp, and two runtimes binding from place 0 would stack on the same cores)
wait_host_idle
say "T3 cpu start (physical cores $NPHYS)"
start_sampler "T3 cpu"
HOME=$H PATH=$H/venv/bin:$PATH taskset -c "$NPHYS" \
    bash "$X/rp_t3_run.sh" "$X/run_t3_star" cpu $GPU 8>&-
rc=$?
stop_sampler
say "T3 cpu rc=$rc"
wait_host_idle
say "T3 gpu start (GPU $GPU, physical cores $NPHYS)"
start_sampler "T3 gpu"
HOME=$H PATH=$H/venv/bin:$PATH taskset -c "$NPHYS" \
    bash "$X/rp_t3_run.sh" "$X/run_t3_star" gpu $GPU 8>&-
rc=$?
stop_sampler
say "T3 gpu rc=$rc"
say "END"
