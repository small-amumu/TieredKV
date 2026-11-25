#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
# lab05 GPU 0, sequential: A1 diagnostic (8 length-octile items) on 0.6B and 1.7B, 4B smoke, A1 on 4B.
# Stops at the first failing step. Stop: kill -- -<PGID> (never pkill -f).
set -u
R=$HOME/work/t4
PY="python3 -B"
MD=$HOME/work/models
ARMS=h54,h33,h32,sc_G1024_del75,rnd_G1024_del75,sc_G1024_h32_75
export PYTHONPATH=$HOME/work/t4pylib
cd "$R" || exit 2
CL=logs/chain_a1.log
echo "$(date -u +%FT%TZ) chain_a1 start pid $$ pgid $(ps -o pgid= $$ | tr -d ' ')" >> "$CL"
step() { echo "$(date -u +%FT%TZ) step $*" >> "$CL"; "$@"; rc=$?; echo "$(date -u +%FT%TZ) step rc=$rc" >> "$CL"; return $rc; }
step ./t4_run.sh "$R" "$PY" 0 "$MD/Qwen3-0.6B" qwen3_0p6b a1_qwen3_0p6b "$ARMS" --diag8 &&
step ./t4_run.sh "$R" "$PY" 0 "$MD/Qwen3-1.7B" qwen3_1p7b a1_qwen3_1p7b "$ARMS" --diag8 &&
step ./t4_smoke.sh "$R" "$PY" 0 "$MD/Qwen3-4B" qwen3_4b &&
step ./t4_run.sh "$R" "$PY" 0 "$MD/Qwen3-4B" qwen3_4b a1_qwen3_4b "$ARMS" --diag8
RC=$?
echo "$(date -u +%FT%TZ) chain_a1 rc=$RC" >> "$CL"
exit $RC
