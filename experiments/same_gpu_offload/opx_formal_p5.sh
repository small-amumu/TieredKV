#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
# OPX formal runs on pro5000 (PLAN_OPX.md); GPU0 / GPU2 only (the other two cards belong to other users).
# usage: setsid nohup bash opx_formal_p5.sh <phase> <gpu> [widths] > /dev/null 2>&1 < /dev/null &
#   ours <gpu> <widths>  our engine, accuracy mode, ox set (5 tasks x 50, 130K); one process and one file per width
#   fp <gpu>             full-attention upper bound: tb_bench_v39 arm fp on the ox set (same items)
#   calibfp <gpu>        tb_bench_v39 arm fp at 32K on the lc set vt / fwe / qa_1, items 0-9 (calibration)
# One lock per card. A process holds the host-wide load lock for its first 300 s, so two cards never read 8B weights from
# the spinning disk at the same time (issue 162). Before every run: wait until the card has no compute process
# (nvidia-smi failure = busy). Complete outputs are skipped; an unfinished one is moved to .partialN and rerun.
# Refusals go to opx_formal/launch.log, rc to opx_formal/done.log, compute processes on the card to <name>.gpuwatch.log.
set -u
PH=${1:-}; G=${2:-}; W=${3:-}
K=$HOME/kvt_bench; B=$K/hc_mac/rp1; SK=$K/sekv_cmp; RU=$K/hc_mac/s0_shared/ruler
PY=$HOME/vllm_main/bin/python; M=$HOME/models/Meta-Llama-3.1-8B-Instruct
O=$B/opx_formal
mkdir -p "$O" "$B/tmp"
LL=$O/launch.log
say() { echo "$(date -u +%FT%TZ) phase=$PH gpu=$G w=$W: $*" >> "$LL"; }
case "$PH" in ours|fp|calibfp) ;; *) say "bad phase"; exit 2;; esac
case "$G" in 0|2) ;; *) say "pro5000: only GPU0 / GPU2 are ours: refused"; exit 2;; esac
[ "$PH" != ours ] || [ -n "$W" ] || { say "ours needs widths"; exit 2; }
case "$PH" in fp) NEXP=250;; calibfp) NEXP=30;; *) NEXP=0;; esac   # fp items expected in a finished tb_bench run
exec 9> "/tmp/opx_formal_p5_gpu$G.lock"
flock -n 9 || { say "GPU$G already has an OPX launcher: refused"; exit 1; }
SELF=$(readlink -f "$0")
U=$(nvidia-smi -i "$G" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d " ")
[ -n "$U" ] || { say "no uuid for GPU$G: refused"; exit 2; }
free_gb=$(df -BG --output=avail "$O" | tail -1 | tr -dc 0-9)
[ "${free_gb:-0}" -ge 20 ] || { say "only ${free_gb} GB free: refused"; exit 2; }
LOG=$O/done.log
busy() {
  local a
  a=$(timeout 30 nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null) || { echo FAIL; return; }
  echo "$a" | tr -d " " | grep "^$U,"
}
wait_free() {
  local b w=0
  while :; do
    b=$(busy)
    [ -z "$b" ] && break
    [ $((w % 60)) = 0 ] && echo "$(date -u +%FT%TZ) $1 waiting for GPU$G: $(echo $b | tr '\n' ' ')" >> "$LOG"
    w=$((w + 1)); sleep 60
  done
  [ $w -gt 0 ] && echo "$(date -u +%FT%TZ) $1 GPU$G free, starting" >> "$LOG"
  return 0
}
complete() {  # complete <name>: 0 if its output is finished. ours: end record. tb_bench rewrites its summary .json after
  # EVERY item (tb_bench_v39 line 864 is inside the item loop), so its completeness is the per-item record count
  local n
  if [ "$PH" = ours ]; then grep -q '"end": true' "$O/$1.jsonl" 2>/dev/null; return; fi
  # distinct (task, i) among fp records without an error: tb_bench re-runs errored items on resume and appends a new
  # line, so the raw line count over-counts (round-4 review)
  n=$("$PY" -I -c 'import json, sys
s = set()
for ln in open(sys.argv[1], "rb").read().split(b"\n"):
    try:
        r = json.loads(ln.decode("utf-8"))
    except Exception:
        continue
    if r.get("arm") == "fp" and not r.get("error"):
        s.add((r.get("task"), r.get("i")))
print(len(s))' "$O/$1.jsonl" 2>/dev/null); n=${n:-0}
  [ "$n" -ge "$NEXP" ]
}
prep() {  # prep <name>: 0 = run it, 1 = already complete
  local k=1 f moved=""
  complete "$1" && { echo "$(date -u +%FT%TZ) $1 complete, skipped" >> "$LOG"; return 1; }
  wait_free "$1"                                   # wait first: an orphaned run may still be writing this output
  complete "$1" && { echo "$(date -u +%FT%TZ) $1 complete, skipped" >> "$LOG"; return 1; }
  if [ "$PH" != ours ]; then                       # tb_bench resumes from its own <out>l: keep it, keep logs
    [ -e "$O/$1.jsonl" ] && echo "$(date -u +%FT%TZ) $1 resuming from its existing jsonl (tb_bench resume)" >> "$LOG"
    for f in out err; do [ -e "$O/$1.$f" ] && mv "$O/$1.$f" "$O/$1.$f.$(date -u +%Y%m%dT%H%M%S)"; done
    return 0
  fi
  while [ -e "$O/$1.jsonl.partial$k" ] || [ -e "$O/$1.log.partial$k" ]; do k=$((k + 1)); done
  for f in jsonl log; do [ -e "$O/$1.$f" ] && { mv "$O/$1.$f" "$O/$1.$f.partial$k"; moved="$moved $f"; }; done
  [ -n "$moved" ] && echo "$(date -u +%FT%TZ) $1 leftovers ($moved ) moved to .partial$k, rerunning" >> "$LOG"
  return 0
}
FAILS=""
ITEMERR=""
FIN=0
WATCH=""
trap 'kill $WATCH 2>/dev/null; [ "$FIN" = 1 ] || echo "$(date -u +%FT%TZ) $PH gpu$G ABORTED" >> "$LOG"' EXIT
export CUDA_VISIBLE_DEVICES=$U HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PKV5_ROWS
run() {  # run <name> <command...>: load lock for the first 300 s, gpuwatch, rc, completeness
  local name=$1 rc pid; shift
  prep "$name" || return 0
  ( exec 9>&-; while :; do echo "$(date -u +%T) $(timeout 30 nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
      --format=csv,noheader | grep "$U" | tr '\n' ';')"; sleep 2; done ) >> "$O/$name.gpuwatch.log" 2>&1 &
  WATCH=$!
  exec 7> /tmp/opx_p5_load.lock
  flock 7                                          # waits while the other card's process is loading
  echo "$(date -u +%FT%TZ) start $name gpu$G $U md5 $(md5sum "$SELF" $B/engine7/opx_rp_run.py $B/engine7/opx_engine_v8.py \
    $B/engine7/opx_graph.py $B/engine7/opx_common.py $B/engine7/pcie_meter.py $B/kalign/kalign_opx.py \
    $SK/code/tb_bench_v39.py | awk '{printf "%s:%s ", $2, substr($1,1,8)}')" >> "$LOG"
  ( exec 9>&- 7>&-; "$@" ) &
  pid=$!
  sleep 300
  flock -u 7; exec 7>&-
  wait $pid
  rc=$?
  kill $WATCH 2>/dev/null; WATCH=""
  echo "$(date -u +%FT%TZ) $name rc=$rc" >> "$LOG"
  { [ "$rc" = 0 ] && complete "$name"; } || FAILS="$FAILS $name"
  n=$(grep '"error": ' "$O/$name.jsonl" 2>/dev/null | grep -vc '"skipped_length"'); n=${n:-0}
  [ "$n" -gt 0 ] && { echo "$(date -u +%FT%TZ) $name has $n item records with errors" >> "$LOG"; ITEMERR="$ITEMERR $name:$n"; }
  return 0
}
case "$PH" in
  ours)
    D=""; for t in vt cwe fwe qa_1 qa_2; do D="$D,$RU/ox/llama31_8b/130000/$t/validation.jsonl"; done
    for w in ${W//,/ }; do
      run "ours_h$w" bash -c "cd '$B/engine7' && KVT_ROOT='$B/root' KALIGN_DIR='$B/kalign' \
        KALIGN_CAL='$K/data/diag_long_text.txt' OPX_KERNEL=opx TRITON_CACHE_DIR='$B/cache/triton' \
        CUDA_CACHE_PATH='$B/cache/nv' XDG_CACHE_HOME='$B/cache' HF_HOME='$B/cache/hf' TMPDIR='$B/tmp' \
        exec '$PY' -B -I opx_rp_run.py '$M' '$O/ours_h$w.jsonl' --mode ruler --data '${D#,}' --items 0:50 \
        --widths $w --length 130000 --cap 132048 --layout fixed --graph --slim --exact_pin > '$O/ours_h$w.log' 2>&1"
    done
    ;;
  fp|calibfp)
    if [ "$PH" = fp ]; then RD=$RU/ox; TS=""; for t in vt cwe fwe qa_1 qa_2; do TS="$TS,ruler:$t@130000"; done; N=50
    else RD=$RU/lc; TS=",ruler:vt@32768,ruler:fwe@32768,ruler:qa_1@32768"; N=10; fi
    run "$PH" bash -c "cd '$SK/code' && KVT_ROOT='$K/hc_mac/tbtree_v3' TB_Q_MODULE=ours128_q RULER_DIR='$RD' \
      RULER_TAG=llama31_8b LB16_PYLIB='$SK/pylib' LB16_DATA='$SK/nodata' exec '$PY' -B -I tb_bench_v39.py '$M' \
      '$O/$PH.json' --arms fp --tasks '${TS#,}' --n $N --max_len 8192 > '$O/$PH.out' 2> '$O/$PH.err'"
    ;;
esac
FIN=1
if [ -z "$FAILS" ]; then echo "$(date -u +%FT%TZ) $PH gpu$G ALL DONE ok; item errors:${ITEMERR:- none}" >> "$LOG"
else echo "$(date -u +%FT%TZ) $PH gpu$G FINISHED WITH FAILURES:$FAILS; item errors:${ITEMERR:- none}" >> "$LOG"; fi
