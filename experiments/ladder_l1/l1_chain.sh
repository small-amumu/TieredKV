#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
# Round L1 chain (2026-11-14): algorithm ladder on the SpeCache / KIVI-128 base, REG_L1_ladder.md.
# Two lanes on pro5000 (at most two cards): lane A = main panel on GPU1, lane B = held-out then new panel on
# GPU3. Each lane runs code/run_pkv3.sh (one model at a time on its card, failed models rerun once, DONE3 line).
# Refuses to start if an output directory exists; holds a lock so it cannot run twice.
# usage (from anywhere): setsid nohup bash ~/kvt_bench/hc_mac/rootcause/l1/l1_chain.sh > /dev/null 2>&1 < /dev/null &
cd ~/kvt_bench/hc_mac || exit 1
exec 8> /tmp/hcmac_l1_chain.lock
flock -n 8 || { echo "L1 chain already running"; exit 1; }
for d in results/pkv/l1_main results/pkv/l1_held results/pkv/l1_new; do
  [ -e "$d" ] && { echo "$d exists: refusing to start"; exit 1; }
done
RUNS=kivi128,kivi128k64_xqs,kivi128k120_x2s120_h88,kivi128k80_x2s160_h88,kivi128k80_x2s160_j50_h88
RUNS=$RUNS,kivi128k64_x2s128_j50_h54,kivi128k64_xor,kivi128k80_x2s160_j30_h88,kivi128k64_x2s128_hff_id
LOG=results/pkv/l1_chain.log
echo "$(date -u +%FT%TZ) start md5 $(md5sum rootcause/sel14_lab.py rootcause/sel14_q.py rootcause/sel13_q.py \
  rootcause/sel4_q.py rootcause/l1/l1_analyze.py rootcause/l1/l1_chain.sh rootcause/l1/l1_*.txt rootcause/l1/REG_L1_ladder.md code/run_pkv3.sh | awk '{printf "%s:%s ", $2, substr($1,1,8)}')" >> $LOG
(GPUS="1" bash code/run_pkv3.sh rootcause/l1/l1_main.txt l1_main ../rootcause/sel14_lab.py --runs $RUNS
 echo "$(date -u +%FT%TZ) lane A done" >> $LOG) &
A=$!
(GPUS="3" bash code/run_pkv3.sh rootcause/l1/l1_held.txt l1_held ../rootcause/sel14_lab.py --runs $RUNS
 GPUS="3" bash code/run_pkv3.sh rootcause/l1/l1_new.txt l1_new ../rootcause/sel14_lab.py --runs $RUNS
 echo "$(date -u +%FT%TZ) lane B done" >> $LOG) &
B=$!
wait "$A" "$B"
~/vllm_main/bin/python -I rootcause/l1/l1_analyze.py results/pkv rootcause/l1 > results/pkv/l1_analysis.txt 2>&1
rc=$?
echo "$(date -u +%FT%TZ) L1 CHAIN DONE analyze rc=$rc" >> $LOG
