#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# NAB1 chain (REG_NAB1.md) on STAR: one GPU per call, models in order; per model trace generation, then replay.
# usage (logs/ must exist): setsid nohup bash nab1_chain.sh <gpu> <model> [<model> ...] \
#   > logs/launch_gpu<gpu>.out 2>&1 < /dev/null &
# models: qwen3_0p6b qwen3_1p7b qwen3_4b. Stop: kill -- -<PGID> (the EXIT trap logs DONE; SIGKILL logs nothing).
set -u
GPU=${1:?gpu}; shift
W=$HOME/work/nab/nab1; PY=$HOME/work/venv/bin/python; M=$HOME/work/models
DATA=$HOME/work/nab/ch4x/reason/math500/test.jsonl
RUNS=kivi128,kivi128k64_x2s128_j50_h54,h2o
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # 4B fp32 replay at 8800 peaks ~20 GiB (A3 probe)
LOG=$W/logs/nab1_chain_gpu$GPU.log
mkdir -p $W/out $W/logs
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
exec 9>>"$W/logs/gpu$GPU.lock"; flock -n 9 || { log "gpu$GPU chain lock held, exit"; exit 3; }
fail=1
trap 'log "DONE nab1 chain gpu$GPU fail=$fail exit=$?"' EXIT
trap 'fail=1; exit 143' TERM INT HUP
cd $W || exit 1
log "START gpu$GPU models=[$*] md5 $(md5sum nab1_gen.py nab1_lab.py nab1_evict.py $DATA \
  | awk '{n=split($2,p,"/"); printf "%s:%s ", p[n], substr($1,1,8)}')"
fail=0
for tag in "$@"; do
  case $tag in
    qwen3_0p6b) md=$M/Qwen3-0.6B; bs=8;;
    qwen3_1p7b) md=$M/Qwen3-1.7B; bs=8;;
    qwen3_4b)   md=$M/Qwen3-4B;   bs=8;;
    *) log "unknown model $tag"; fail=1; continue;;
  esac
  CUDA_VISIBLE_DEVICES=$GPU $PY -I nab1_gen.py $md out/traces_$tag.jsonl --data $DATA --start 0 --n 64 --batch $bs \
    --max_new 8192 > logs/gen_$tag.log 2>&1
  rc=$?; log "NAB1 gen $tag rc=$rc"; [ $rc -ne 0 ] && { fail=1; continue; }
  CUDA_VISIBLE_DEVICES=$GPU $PY -I nab1_lab.py $md out/traces_$tag.jsonl out/lab_$tag.json --stack $HOME/work/nab \
    --runs $RUNS --max_len 8800 --chunk 256 > logs/lab_$tag.log 2>&1
  rc=$?; log "NAB1 lab $tag rc=$rc"; [ $rc -ne 0 ] && fail=1
done
exit $fail
