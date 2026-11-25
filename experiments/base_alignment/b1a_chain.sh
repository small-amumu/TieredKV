#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
# Round B1a chain (2026-11-14): our stack base vs official-aligned Block-GTQ v3, REG_B1a.md.
# Two lanes on pro5000 (two pro5000 cards): lane A on GPU2, lane B on GPU3 (GPU0 / GPU1 taken by another user at freeze time), each via code/run_pkv3.sh into
# its own tag directory (results/pkv/b1a_A, results/pkv/b1a_B: run_pkv3 keeps one run.log and one redo list per tag).
# Refuses to start if an output directory exists.
# usage: setsid -f bash -c "nohup bash ~/kvt_bench/hc_mac/rootcause/b1a/b1a_chain.sh > /var/tmp/b1a_launch.out 2>&1 < /dev/null"
cd ~/kvt_bench/hc_mac || exit 1
exec 8> /tmp/hcmac_b1a_chain.lock
flock -n 8 || { echo "B1a chain already running"; exit 1; }
for d in results/pkv/b1a_A results/pkv/b1a_B; do
  [ -e "$d" ] && { echo "$d exists: refusing to start"; exit 1; }
done
RUNS=stk128,kivi128,gtqk20v20w128,gtqk22v20w128,gtqk24v20w128,gtqk28v20w128
LOG=results/pkv/b1a_chain.log
echo "$(date -u +%FT%TZ) start md5 $(md5sum rootcause/gtq3_lab.py rootcause/gtq_q_v3.py rootcause/sel10_q.py \
  rootcause/sel4_q.py rootcause/b1a/b1a_analyze.py rootcause/b1a/b1a_chain.sh rootcause/b1a/b1a_lane*.txt \
  rootcause/b1a/REG_B1a.md code/run_pkv3.sh | awk '{printf "%s:%s ", $2, substr($1,1,8)}')" >> $LOG
(GPUS="2" bash code/run_pkv3.sh rootcause/b1a/b1a_laneA.txt b1a_A ../rootcause/gtq3_lab.py --runs $RUNS
 echo "$(date -u +%FT%TZ) lane A done" >> $LOG) &
A=$!
sleep 20
(GPUS="3" bash code/run_pkv3.sh rootcause/b1a/b1a_laneB.txt b1a_B ../rootcause/gtq3_lab.py --runs $RUNS
 echo "$(date -u +%FT%TZ) lane B done" >> $LOG) &
B=$!
wait "$A" "$B"
~/vllm_main/bin/python -I rootcause/b1a/b1a_analyze.py results/pkv rootcause/b1a > results/pkv/b1a_analysis.txt 2>&1
rc=$?
echo "$(date -u +%FT%TZ) B1a CHAIN DONE analyze rc=$rc" >> $LOG
