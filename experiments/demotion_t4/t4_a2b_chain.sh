#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# A2b on pro5000, one GPU, sequential per model: smoke on the held-out traces, then the replay with the measured
# uniform points h44 / h53 (136 B) next to h54 and score h32 (both repeated from the held-out run: the merge check).
# usage: t4_a2b_chain.sh <gpu index> <tag> [<tag> ...]   tags: qwen3_0p6b qwen3_1p7b qwen3_4b
# The simulation modules load from T4_STACK=/home/user/kvt_bench (md5-identical copy; s0_cov.py hard-codes its path).
# Resumable: rerun the same command. Stops at the first failure. Stop: kill -- -<PGID> (never pkill -f).
set -u
GPU=${1:?gpu}; shift
R=$HOME/kvt_bench/t4
PY="$HOME/kvt_venv/bin/python -I -B"
MD=$HOME/models
ARMS=h54,h44,h53,sc_G1024_h32_75
export T4_STACK=$HOME/kvt_bench
declare -A MODEL=( [qwen3_0p6b]=Qwen3-0.6B [qwen3_1p7b]=Qwen3-1.7B [qwen3_4b]=Qwen3-4B )
cd "$R" || exit 2
mkdir -p logs
CL=logs/a2b_chain_gpu$GPU.log
exec 8> "logs/a2b_chain_gpu$GPU.lock"
if ! flock -n 8; then echo "$(date -u +%FT%TZ) REFUSED: another a2b chain holds gpu $GPU" >> "$CL"; exit 4; fi
echo "$(date -u +%FT%TZ) a2b_chain start gpu $GPU pid $$ pgid $(ps -o pgid= $$ | tr -d ' ') tags $*" >> "$CL"
trap 'echo "$(date -u +%FT%TZ) a2b_chain gpu$GPU killed by signal" >> "$CL"; exit 143' TERM INT HUP
wait_free() {  # the previous step's process may still hold the GPU for a moment after it exits; wait at most 10 min
  local uuid busy i
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
  for i in $(seq 1 40); do
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")
    [ "$busy" -eq 0 ] && return 0
    sleep 15
  done
  return 1
}
RC=0
for TAG in "$@"; do
  if [ -z "${MODEL[$TAG]:-}" ]; then echo "$(date -u +%FT%TZ) unknown tag $TAG" >> "$CL"; RC=2; break; fi
  M=$MD/${MODEL[$TAG]}
  HT=${TAG}_ho
  echo "$(date -u +%FT%TZ) step smoke $HT" >> "$CL"
  ./t4_smoke.sh "$R" "$PY" "$GPU" "$M" "$HT"
  RC=$?
  echo "$(date -u +%FT%TZ) step smoke $HT rc=$RC" >> "$CL"
  [ "$RC" -eq 0 ] || break
  wait_free || echo "$(date -u +%FT%TZ) GPU $GPU still busy after 10 min" >> "$CL"
  echo "$(date -u +%FT%TZ) step replay a2b_$TAG" >> "$CL"
  ./t4_run.sh "$R" "$PY" "$GPU" "$M" "$HT" "a2b_$TAG" "$ARMS"
  RC=$?
  echo "$(date -u +%FT%TZ) step replay a2b_$TAG rc=$RC" >> "$CL"
  [ "$RC" -eq 0 ] || break
done
echo "$(date -u +%FT%TZ) a2b_chain gpu$GPU rc=$RC" >> "$CL"
exit $RC
