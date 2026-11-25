#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# hd5090 (host-native), one GPU, sequential per model tag:
#   1. held-out traces: nab1_gen.py (the A2 trace generator, bf16 HF generate, md5-checked) on MATH-500 items 64..191;
#   2. T4 smoke on those traces (controls on this host, same frozen code as A2);
#   3. teacher-forced replay of the held-out traces with 7 arms (t4_run.sh, the A2 launcher).
# usage: t4_ho_chain.sh <gpu index> <tag> [<tag> ...]   tags: qwen3_0p6b qwen3_1p7b qwen3_4b
# Resumable: rerun the same command (generation resumes per batch, replay per item). Stops at the first failure.
# A resumed generation batch is regenerated with the items it still lacks (batch_items records the actual set).
# Stop: kill -- -<PGID> (never pkill -f).
set -u
GPU=${1:?gpu}; shift
R=$HOME/t4a3/t4
PY="$HOME/envs/t4/bin/python -I -B"
MD=$HOME/t4a3/models
GEN_MD5=fa845d07082a142b134683e49653bb36             # nab1_gen.py that wrote the A2 traces
ARMS=h54,h43,h33,h32,sc_G1024_h32_75,rnd_G1024_h32_75,sc_G1024_2lvl
declare -A MODEL=( [qwen3_0p6b]=Qwen3-0.6B [qwen3_1p7b]=Qwen3-1.7B [qwen3_4b]=Qwen3-4B )
cd "$R" || exit 2
mkdir -p logs nab1/out "$HOME/gpu_queue"
CL=logs/ho_chain_gpu$GPU.log
exec 8> "logs/ho_chain_gpu$GPU.lock"
if ! flock -n 8; then echo "$(date -u +%FT%TZ) REFUSED: another ho_chain holds gpu $GPU" >> "$CL"; exit 4; fi
echo "$(date -u +%FT%TZ) ho_chain start gpu $GPU pid $$ pgid $(ps -o pgid= $$ | tr -d ' ') tags $*" >> "$CL"
trap 'echo "$(date -u +%FT%TZ) ho_chain gpu$GPU killed by signal" >> "$CL"; exit 143' TERM INT HUP
if [ "$(md5sum nab1_gen.py | cut -c1-32)" != "$GEN_MD5" ]; then
  echo "$(date -u +%FT%TZ) REFUSED: nab1_gen.py md5 differs from $GEN_MD5" >> "$CL"; exit 3
fi
gpu_free() {
  local uuid busy used
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU")
  [ -n "$uuid" ] || return 1
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  [ "$busy" -eq 0 ] && [ "${used:-99999}" -le 1024 ]
}
clean_traces() {  # keep only parseable records (a kill can leave a torn last line); print the number of distinct items
  $PY - "$1" <<'PYEOF'
import json, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    print(0); sys.exit(0)
good, seen = [], set()
for line in open(p, encoding="utf-8").read().splitlines():
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if r["i"] not in seen:
        seen.add(r["i"]); good.append(line)
tmp = p + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    f.write("".join(x + "\n" for x in good))
os.replace(tmp, p)
print(len(seen))
PYEOF
}
RC=0
for TAG in "$@"; do
  if [ -z "${MODEL[$TAG]:-}" ]; then echo "$(date -u +%FT%TZ) unknown tag $TAG" >> "$CL"; RC=2; break; fi
  M=$MD/${MODEL[$TAG]}
  HT=${TAG}_ho
  TR=nab1/out/traces_$HT.jsonl
  N=$(clean_traces "$TR")
  if [ "${N:-0}" -lt 128 ]; then
    if ! gpu_free; then echo "$(date -u +%FT%TZ) REFUSED: GPU $GPU busy before generation" >> "$CL"; RC=5; break; fi
    echo "$(date -u +%FT%TZ) step gen $HT (have $N)" >> "$CL"
    flock -n "$HOME/gpu_queue/gpu$GPU.lock" env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
      $PY -u nab1_gen.py "$M" "$TR" --data data/math500_test.jsonl --start 64 --n 128 --batch 8 --max_new 8192 \
      >> "logs/gen_$HT.log" 2>&1
    RC=$?
    N=$(clean_traces "$TR")
    echo "$(date -u +%FT%TZ) step gen $HT rc=$RC items=$N" >> "$CL"
    if [ "$RC" -ne 0 ]; then break; fi
    if [ "${N:-0}" -ne 128 ]; then RC=9; break; fi
  fi
  echo "$(date -u +%FT%TZ) step smoke $HT" >> "$CL"
  ./t4_smoke.sh "$R" "$PY" "$GPU" "$M" "$HT"
  RC=$?
  echo "$(date -u +%FT%TZ) step smoke $HT rc=$RC" >> "$CL"
  [ "$RC" -eq 0 ] || break
  echo "$(date -u +%FT%TZ) step replay a3ho_$TAG" >> "$CL"
  ./t4_run.sh "$R" "$PY" "$GPU" "$M" "$HT" "a3ho_$TAG" "$ARMS"
  RC=$?
  echo "$(date -u +%FT%TZ) step replay a3ho_$TAG rc=$RC" >> "$CL"
  [ "$RC" -eq 0 ] || break
done
echo "$(date -u +%FT%TZ) ho_chain gpu$GPU rc=$RC" >> "$CL"
exit $RC
