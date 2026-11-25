#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
# R3 drift check D0 (pro5000): X-0140's own harness (tb_bench_v37 ef85729a, the m128r37.sh environment) reruns the fp arm only on
# items 0-1 of vt / fwe / qa_1 @130000; the verdict compares pred and score with the archived X-0140 fp records of those items.
# Usage: bash r3_fpdrift.sh <gpu 0-3>
set -u
B=/home/user/kvt_bench/hc_mac/rp1
EG=$B/engine7
O=$EG/fpdrift
T=/home/user/kvt_bench/hc_mac/tbtree_v3
S=/home/user/kvt_bench/hc_mac/s0_shared/ruler
G=${1:-}
case "$G" in 0|1|2|3) ;; *) echo "GPU arg must be 0-3, got: '$G'" >&2; exit 2;; esac
CODE="$(md5sum $T/hc_mac/rootcause/{tb_bench_v37,cap_codec,cap_attn,ours_q,ours128_q,pkv5_q}.py | cut -c1-8 | tr '\n' ' ')"
[ "$CODE" = "ef85729a 903b8342 29025867 92ef53ff 7aa91292 817000f6 " ] || { echo "harness md5 mismatch: $CODE" >&2; exit 3; }
( cd $S/lc/llama31_8b/130000 && md5sum -c $EG/DATA_MD5SUMS --quiet ) || { echo "data md5 gate failed" >&2; exit 3; }
mkdir -p $O $B/tmp
[ -e $O/fp.jsonl ] || [ -e $O/rc.log ] && { echo "fpdrift output exists: refusing (tb_bench would resume)" >&2; exit 5; }
exec 9>$B/tmp/gpu$G.lock
flock -n 9 || { echo "GPU$G lock held by another driver: refusing" >&2; exit 4; }
if nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader | grep -q .; then echo "GPU$G busy (process)" >&2; exit 4; fi
MU=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)
case "$MU" in ''|*[!0-9]*) echo "GPU$G memory reading not numeric: '$MU'" >&2; exit 4;; esac
if [ "$MU" -ge 500 ]; then echo "GPU$G busy ($MU MiB used)" >&2; exit 4; fi
echo "$(date -u +%FT%TZ) R3_FPDRIFT_START gpu=$G code $CODE driver_md5=$(md5sum $EG/r3_fpdrift.sh | cut -c1-32)" >> $O/rc.log
cd $T/hc_mac/rootcause || exit 1
env KVT_ROOT=$T TB_Q_MODULE=ours128_q OURS_GROW_CHUNK=1048576 TB_TF=1 TB_GOLD=1 PKV5_ROWS=64 PYTHONHASHSEED=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=$G HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 RULER_DIR=$S/lc RULER_TAG=llama31_8b PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  timeout -k 60 7200 /home/user/vllm_main/bin/python -B -I tb_bench_v37.py /home/user/models/Meta-Llama-3.1-8B-Instruct $O/fp.json \
  --arms fp --tasks "ruler:vt@130000,ruler:fwe@130000,ruler:qa_1@130000" --n 2 > $O/fp.out 2>&1
rc=$?
echo "$(date -u +%FT%TZ) r3_fpdrift rc=$rc fp_records=$(grep -c '"arm": "fp"' $O/fp.jsonl 2>/dev/null || echo 0)" >> $O/rc.log
echo "$(date -u +%FT%TZ) R3_FPDRIFT_END" >> $O/rc.log
