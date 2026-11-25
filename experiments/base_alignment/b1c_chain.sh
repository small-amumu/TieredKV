#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# Round B1c chain (2026-11-14): our stack base vs RotateKV (official code path), REG_B1c.md.
# Two lanes on pro5000 (two pro5000 cards): lane A on GPU0, lane B on GPU1 (GPU2 / GPU3 taken by another user at freeze time), each via code/run_pkv3.sh into
# its own tag directory (results/pkv/b1c_A, results/pkv/b1c_B: run_pkv3 keeps one run.log and one redo list per tag).
# Refuses to start if an output directory exists.
# usage: setsid -f bash -c "nohup bash ~/kvt_bench/hc_mac/rootcause/b1c/b1c_chain.sh > /var/tmp/b1c_launch.out 2>&1 < /dev/null"
cd ~/kvt_bench/hc_mac || exit 1
exec 8> /tmp/hcmac_b1c_chain.lock
flock -n 8 || { echo "B1c chain already running"; exit 1; }
for d in results/pkv/b1c_A results/pkv/b1c_B; do
  [ -e "$d" ] && { echo "$d exists: refusing to start"; exit 1; }
done
RUNS=stk128,kivi128,rkv2,rkv2w128,rkv2w256,rkv3,gtqk22v20w128
LOG=results/pkv/b1c_chain.log
echo "$(date -u +%FT%TZ) start md5 $(md5sum rootcause/rkv_lab.py rootcause/rkv_q.py rootcause/gtq_q_v3.py rootcause/sel10_q.py \
  rootcause/sel4_q.py rootcause/b1c/b1c_analyze.py rootcause/b1c/b1c_chain.sh rootcause/b1c/b1c_lane*.txt \
  rootcause/b1c/REG_B1c.md code/run_pkv3.sh | awk '{printf "%s:%s ", $2, substr($1,1,8)}')" >> $LOG
(GPUS="0" bash code/run_pkv3.sh rootcause/b1c/b1c_laneA.txt b1c_A ../rootcause/rkv_lab.py --runs $RUNS
 echo "$(date -u +%FT%TZ) lane A done" >> $LOG) &
A=$!
sleep 20
(GPUS="1" bash code/run_pkv3.sh rootcause/b1c/b1c_laneB.txt b1c_B ../rootcause/rkv_lab.py --runs $RUNS
 echo "$(date -u +%FT%TZ) lane B done" >> $LOG) &
B=$!
wait "$A" "$B"
~/vllm_main/bin/python -I rootcause/b1c/b1c_analyze.py results/pkv rootcause/b1c > results/pkv/b1c_analysis.txt 2>&1
rc=$?
echo "$(date -u +%FT%TZ) B1c CHAIN DONE analyze rc=$rc" >> $LOG
