#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# T1 chain (REG_T1.md v5) on STAR GPU0 only: models in order, then the scorer. Requires a passing smoke (smoke/PASS).
# usage (from ~/work/nab/t1):
#   md5sum -c --quiet FREEZE.md5 && \
#   ( setsid nohup bash t1_chain.sh 0 qwen3_0p6b qwen3_1p7b qwen3_4b >> logs/launch_gpu0.out 2>&1 < /dev/null & )
# Stop: kill -- -<PGID> (the EXIT trap logs DONE; SIGKILL logs nothing). Lock fds 8/9 are inherited by the python
# child: after a SIGKILL check `fuser ~/work/gpu_queue/gpu0.lock` before relaunching.
set -u
GPU=${1:?gpu}; shift
[ "$GPU" = "0" ] || { echo "$(date -u +%FT%TZ) REFUSED: T1 runs on GPU0 only (REG_T1 section 1)"; exit 2; }
[ $# -gt 0 ] || { echo "$(date -u +%FT%TZ) REFUSED: no models given"; exit 2; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
W=$HOME/work/nab/t1; N1=$HOME/work/nab/nab1; PY=$HOME/work/venv/bin/python; M=$HOME/work/models
Q=$HOME/work/gpu_queue                      # STAR convention shared by the other GPU jobs
mkdir -p $W/out $W/logs $Q
cd $W || exit 1
LOG=$W/logs/t1_chain_gpu$GPU.log
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
exec 8>>"$Q/gpu$GPU.lock"; flock -n 8 || { echo "$(date -u +%FT%TZ) REFUSED: gpu$GPU queue lock held"; exit 3; }
exec 9>>"$W/logs/gpu$GPU.lock"; flock -n 9 || { echo "$(date -u +%FT%TZ) REFUSED: t1 chain lock held"; exit 3; }
if [ -f "$LOG" ]; then mkdir -p logs/old; mv "$LOG" "logs/old/t1_chain_gpu$GPU.$(date -u +%Y%m%dT%H%M%SZ).log"; fi
frozen_ok(){
  md5sum -c --quiet FREEZE.md5 || { log "REFUSED FREEZE.md5 mismatch $1"; return 1; }
  md5sum -c --quiet smoke/PASS || { log "REFUSED smoke/PASS missing or stale (re-frozen since the smoke) $1"; return 1; }
}
frozen_ok "at start" || exit 2
GATE=$(cat smoke/gate_mib 2>/dev/null)
case "$GATE" in ''|*[!0-9]*) log "REFUSED smoke/gate_mib missing or not a number"; exit 2;; esac
[ "$GATE" -ge 1536 ] || { log "REFUSED gate $GATE below 1536 MiB"; exit 2; }
fail=1
trap 'log "DONE t1 chain gpu$GPU fail=$fail exit=$?"' EXIT
trap 'fail=1; exit 143' TERM INT HUP
log "START gpu$GPU gate ${GATE} MiB models=[$*] md5 $(md5sum t1_q.py t1_lab.py t1_analyze.py t1_chain.sh t1_smoke.sh \
  t1_selftest.py t1_aggtest.py | awk '{n=split($2,p,"/"); printf "%s:%s ", p[n], substr($1,1,8)}')"
fail=0
labs=""
for tag in "$@"; do
  case $tag in
    qwen3_0p6b) md=$M/Qwen3-0.6B;;
    qwen3_1p7b) md=$M/Qwen3-1.7B;;
    qwen3_4b)   md=$M/Qwen3-4B;;
    *) log "unknown model $tag: stop"; fail=1; break;;
  esac
  frozen_ok "before $tag" || { fail=1; break; }
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU 2>/dev/null | tr -d ' ')
  case "$free" in ''|*[!0-9]*) log "gpu$GPU free memory unreadable ('$free') before $tag: stop"; fail=1; break;; esac
  if [ "$free" -lt "$GATE" ]; then log "gpu$GPU free ${free} MiB < gate ${GATE} MiB before $tag: stop"; fail=1; break; fi
  log "T1 lab $tag start (gpu$GPU free ${free} MiB)"
  CUDA_VISIBLE_DEVICES=$GPU $PY -I t1_lab.py $md $N1/out/traces_$tag.jsonl out/t1_$tag.json --stack $HOME/work/nab \
    --nab1_lab $N1/out/lab_$tag.json --max_len 8800 --chunk 256 >> logs/lab_$tag.log 2>&1
  rc=$?; log "T1 lab $tag rc=$rc"; [ $rc -ne 0 ] && { fail=1; break; }
  labs="$labs out/t1_$tag.json"
done
if [ $fail -eq 0 ]; then
  $PY -I t1_analyze.py out $labs >> logs/analyze.log 2>&1
  rc=$?; log "T1 analyze rc=$rc"; [ $rc -ne 0 ] && fail=1
fi
exit $fail
