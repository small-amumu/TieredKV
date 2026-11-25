#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
# R3 (pro5000): the real engine E1, free greedy generation as tb_bench v3.7 (rp_run.py --mode ruler), Llama-3.1-8B-Instruct,
# RULER vt / fwe / qa_1 @130000 (X-0140's lc set), items 0-19, widths h33 and h54; one process per task.
# Usage: bash r3_driver.sh <gpu 0-3> <task[,task...]> [formal|smoke]     smoke: cwe item 0 (cwe is outside R3: no spare items
# exist in vt / fwe / qa_1), width h54, output in smoke/ (never used for a verdict); the task argument is still validated.
set -u
B=/home/user/kvt_bench/hc_mac/rp1
EG=$B/engine7
PY=/home/user/vllm_main/bin/python
M=/home/user/models/Meta-Llama-3.1-8B-Instruct
D=/home/user/kvt_bench/hc_mac/s0_shared/ruler/lc/llama31_8b/130000
G=${1:-}; TL=${2:-}; MODE=${3:-formal}
case "$G" in 0|1|2|3) ;; *) echo "GPU arg must be 0-3, got: '$G'" >&2; exit 2;; esac
TASKS=$(echo "$TL" | tr ',' ' ')
for t in $TASKS; do case "$t" in vt|fwe|qa_1) ;; *) echo "task must be vt, fwe or qa_1, got: '$t'" >&2; exit 2;; esac; done
[ -n "$TASKS" ] || { echo "no task given" >&2; exit 2; }
[ "$(echo $TASKS | tr ' ' '\n' | sort | uniq -d)" = "" ] || { echo "duplicate task in '$TL'" >&2; exit 2; }
case "$MODE" in formal|smoke) ;; *) echo "mode must be formal or smoke, got: $MODE" >&2; exit 2;; esac
GT=$(echo "$TL" | tr ',' '_')
if [ "$MODE" = smoke ]; then O=$EG/smoke; ITEMS=0:1; TASKS=cwe; WIDTHS=54; TAG=R3_SMOKE_$GT
else O=$EG; ITEMS=0:20; WIDTHS=33,54; TAG=R3_$GT; fi
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
RC=$O/rc_$GT.log
for t in $TASKS; do
  if [ -e $O/out/r3_$t.jsonl ] || [ -e $O/logs/r3_$t.log ]; then echo "output of $t exists: refusing (rp_run would resume)" >&2; exit 5; fi
done
[ -e $RC ] && { echo "$RC exists: refusing" >&2; exit 5; }
exec 9>$B/tmp/gpu$G.lock                  # one driver per card (shared with R1's driver; held until this driver exits)
flock -n 9 || { echo "GPU$G lock held by another driver: refusing" >&2; exit 4; }
if nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader | grep -q .; then echo "GPU$G busy (process)" >&2; exit 4; fi
MU=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits)
case "$MU" in ''|*[!0-9]*) echo "GPU$G memory reading not numeric: '$MU'" >&2; exit 4;; esac
if [ "$MU" -ge 500 ]; then echo "GPU$G busy ($MU MiB used)" >&2; exit 4; fi
echo "$(date -u +%FT%TZ) ${TAG}_START gpu=$G uuid=$U driver_md5=$(md5sum $EG/r3_driver.sh | cut -c1-32)" >> $RC
$PY -B -I $EG/tests/chunk_mask_test.py > $O/logs/chunk_test_$GT.log 2>&1
crc=$?
CST=$(grep -o 'CHUNK_TEST [A-Z]*' $O/logs/chunk_test_$GT.log | tail -1)
echo "$(date -u +%FT%TZ) chunk_test_$GT rc=$crc $CST" >> $RC
if [ $crc -ne 0 ] || [ "$CST" != "CHUNK_TEST PASS" ]; then echo "$(date -u +%FT%TZ) ${TAG}_ABORT chunk test" >> $RC; exit 6; fi
for t in $TASKS; do                       # one process per task file, under a per-task lock (no two drivers on one task)
  exec 8>$B/tmp/r3_task_$t.$MODE.lock
  flock -n 8 || { echo "$(date -u +%FT%TZ) r3_$t rc=97 task lock held by another driver" >> $RC; continue; }
  if [ -e $O/out/r3_$t.jsonl ] || [ -e $O/logs/r3_$t.log ]; then echo "$(date -u +%FT%TZ) r3_$t rc=98 output exists" >> $RC; exec 8>&-; continue; fi
  t0=$(date +%s)
  timeout -k 60 28800 $PY -B -I $EG/rp_run.py $M $O/out/r3_$t.jsonl --mode ruler --data $D/$t/validation.jsonl --items $ITEMS \
    --widths $WIDTHS --length 130000 > $O/logs/r3_$t.log 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) r3_$t rc=$rc secs=$(( $(date +%s) - t0 ))" >> $RC
  exec 8>&-
done
echo "$(date -u +%FT%TZ) ${TAG}_END" >> $RC
