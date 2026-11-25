#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
# R1 (pro5000): the real engine E1 vs the accuracy reference and fp on the Table A protocol (X-0139 = tb_bench v3.7b TB_TF=1
# TB_GOLD=1), LLaMA-2-7B-32K (MHA, linear RoPE x8), RULER 6 tasks @32768, items 0-9 (X-0139's items); one process per task.
# Two task groups, one card each (pro5000 rule: at most two cards at once, any index, idle first):
#   group a = niah_single_3 niah_multikey_2 niah_multikey_3 (X-0139 g0a), group b = niah_multivalue niah_multiquery vt (g2a).
# Usage: bash r1_driver.sh <gpu 0-3> <a|b> [formal|smoke]     smoke: item 10, first task of the group only, output in smoke/.
set -u
B=/home/user/kvt_bench/hc_mac/rp1
EG=$B/engine6
PY=/home/user/vllm_main/bin/python
M=/home/user/models/LLaMA-2-7B-32K
D=/home/user/kvt_bench/hc_mac/s0_shared/ruler/e1/llama2_7b_32k/32768
G=${1:-}; GRP=${2:-}; MODE=${3:-formal}
case "$G" in 0|1|2|3) ;; *) echo "GPU arg must be 0-3, got: '$G'" >&2; exit 2;; esac
case "$GRP" in a) TASKS="niah_single_3 niah_multikey_2 niah_multikey_3";; b) TASKS="niah_multivalue niah_multiquery vt";;
  *) echo "group must be a or b, got: '$GRP'" >&2; exit 2;; esac
case "$MODE" in formal|smoke) ;; *) echo "mode must be formal or smoke, got: $MODE" >&2; exit 2;; esac
if [ "$MODE" = smoke ]; then O=$EG/smoke; ITEMS=10:11; TASKS=${TASKS%% *}; TAG=R1_SMOKE_$GRP
else O=$EG; ITEMS=0:10; TAG=R1_$GRP; fi
cd $EG || exit 2
U=$(nvidia-smi -i $G --query-gpu=uuid --format=csv,noheader) || exit 2
[ -n "$U" ] || { echo "GPU$G uuid empty" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=$U
export KVT_ROOT=$B/root KALIGN_DIR=$B/kalign KALIGN_CAL=/home/user/kvt_bench/data/diag_long_text.txt
export TRITON_CACHE_DIR=$B/cache/triton CUDA_CACHE_PATH=$B/cache/nv XDG_CACHE_HOME=$B/cache HF_HOME=$B/cache/hf \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_EXTENSIONS_DIR=$B/cache/torch_ext TORCHINDUCTOR_CACHE_DIR=$B/cache/inductor \
       TMPDIR=$B/tmp PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONHASHSEED=0
unset PKV5_ROWS
md5sum -c $EG/MD5SUMS --quiet || { echo "md5 gate failed" >&2; exit 3; }
( cd $D && md5sum -c $EG/DATA_MD5SUMS --quiet ) || { echo "data md5 gate failed" >&2; exit 3; }
mkdir -p $O/out $O/logs $B/tmp
RC=$O/rc_$GRP.log
for t in $TASKS; do
  if [ -e $O/out/r1_$t.jsonl ] || [ -e $O/logs/r1_$t.log ]; then echo "output of $t exists: refusing (no appends)" >&2; exit 5; fi
done
[ -e $RC ] && { echo "$RC exists: refusing" >&2; exit 5; }
exec 9>$B/tmp/gpu$G.lock                  # one driver per card (held until this driver exits)
flock -n 9 || { echo "GPU$G lock held by another driver: refusing" >&2; exit 4; }
if nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader | grep -q .; then echo "GPU$G busy (process)" >&2; exit 4; fi
MU=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)
case "$MU" in ''|*[!0-9]*) echo "GPU$G memory reading not numeric: '$MU'" >&2; exit 4;; esac
if [ "$MU" -ge 500 ]; then echo "GPU$G busy ($MU MiB used)" >&2; exit 4; fi
echo "$(date -u +%FT%TZ) ${TAG}_START gpu=$G uuid=$U driver_md5=$(md5sum $EG/r1_driver.sh | cut -c1-32)" >> $RC
$PY -B -I $EG/tests/chunk_mask_test.py > $O/logs/chunk_test_$GRP.log 2>&1
crc=$?
CST=$(grep -o 'CHUNK_TEST [A-Z]*' $O/logs/chunk_test_$GRP.log | tail -1)
echo "$(date -u +%FT%TZ) chunk_test_$GRP rc=$crc $CST" >> $RC
if [ $crc -ne 0 ] || [ "$CST" != "CHUNK_TEST PASS" ]; then echo "$(date -u +%FT%TZ) ${TAG}_ABORT chunk test" >> $RC; exit 6; fi
for t in $TASKS; do                       # one process per task file
  t0=$(date +%s)
  timeout -k 60 21600 $PY -B -I $EG/rp2_tf.py $M $O/out/r1_$t.jsonl --data $D/$t/validation.jsonl --items $ITEMS \
    > $O/logs/r1_$t.log 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) r1_$t rc=$rc secs=$(( $(date +%s) - t0 )) $(grep -o 'RP2_TF_[A-Z]*.*' $O/logs/r1_$t.log | tail -1)" >> $RC
done
echo "$(date -u +%FT%TZ) ${TAG}_END" >> $RC
