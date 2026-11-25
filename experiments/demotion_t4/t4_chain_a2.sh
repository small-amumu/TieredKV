#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
# lab05 GPU 0, sequential: A2 full run (all traces, all 15 arms) on the models given as arguments, in order.
# usage: t4_chain_a2.sh <tag> [<tag> ...]   tags: qwen3_0p6b qwen3_1p7b qwen3_4b
# Resumable: rerun the same command; each model output resumes item by item. Stops at the first failing step.
# Stop: kill -- -<PGID> (never pkill -f).
set -u
R=$HOME/work/t4
PY="python3 -B"
MD=$HOME/work/models
ARMS=h54,h43,h33,h32,h32c,all_del,sc_G512_del75,sc_G1024_del75,sc_G2048_del75,sc_G1024_h32_75,sc_G1024_key75,sc_G1024_2lvl,rnd_G1024_del75,old_G1024_del75,rnd_G1024_h32_75
declare -A MODEL=( [qwen3_0p6b]=Qwen3-0.6B [qwen3_1p7b]=Qwen3-1.7B [qwen3_4b]=Qwen3-4B )
export PYTHONPATH=$HOME/work/t4pylib
cd "$R" || exit 2
CL=logs/chain_a2.log
echo "$(date -u +%FT%TZ) chain_a2 start pid $$ pgid $(ps -o pgid= $$ | tr -d ' ') tags $*" >> "$CL"
RC=0
for TAG in "$@"; do
  if [ -z "${MODEL[$TAG]:-}" ]; then echo "$(date -u +%FT%TZ) unknown tag $TAG" >> "$CL"; RC=2; break; fi
  echo "$(date -u +%FT%TZ) step a2_$TAG" >> "$CL"
  ./t4_run.sh "$R" "$PY" 0 "$MD/${MODEL[$TAG]}" "$TAG" "a2_$TAG" "$ARMS"
  RC=$?
  echo "$(date -u +%FT%TZ) step a2_$TAG rc=$RC" >> "$CL"
  [ "$RC" -eq 0 ] || break
done
echo "$(date -u +%FT%TZ) chain_a2 rc=$RC" >> "$CL"
exit $RC
