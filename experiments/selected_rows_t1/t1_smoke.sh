#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# T1 smoke (REG_T1.md v5 section 6 step 4) on STAR GPU0: Q0a on 0.6B item 0, then Q0a and the peak on the 4B longest
# trace. On success writes smoke/gate_mib (4B peak reserved MiB + 1536) and smoke/PASS (md5 of FREEZE.md5 as it was
# when the smoke started; FREEZE.md5 lists every T1 file, the stack modules and the inputs). Run in the foreground:
#   cd ~/work/nab/t1 && md5sum -c --quiet FREEZE.md5 && bash t1_smoke.sh >> logs/smoke.log 2>&1
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
W=$HOME/work/nab/t1; N1=$HOME/work/nab/nab1; PY=$HOME/work/venv/bin/python; M=$HOME/work/models
Q=$HOME/work/gpu_queue                      # STAR convention shared by the other GPU jobs
SMOKE_MIN_FREE_MIB=20480
cd $W || exit 1
mkdir -p smoke logs $Q
echo "$(date -u +%FT%TZ) smoke start"
exec 8>>"$Q/gpu0.lock"; flock -n 8 || { echo "gpu0 queue lock held"; exit 3; }
rm -f smoke/gate_mib smoke/gate_mib.tmp smoke/PASS smoke/PASS.tmp smoke/*.smoke.json smoke/*.smoke.json.tmp
md5sum -c --quiet FREEZE.md5 || { echo "FREEZE.md5 mismatch"; exit 2; }
F0=$(md5sum FREEZE.md5) || exit 1
free_ok() {
  local free
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
  case "$free" in ''|*[!0-9]*) echo "gpu0 free memory unreadable ('$free')"; return 1;; esac
  [ "$free" -ge "$SMOKE_MIN_FREE_MIB" ] || { echo "gpu0 free ${free} MiB < ${SMOKE_MIN_FREE_MIB} MiB"; return 1; }
}
free_ok || exit 4
CUDA_VISIBLE_DEVICES=0 $PY -I t1_lab.py $M/Qwen3-0.6B $N1/out/traces_qwen3_0p6b.jsonl smoke/t1_qwen3_0p6b.smoke.json \
  --stack $HOME/work/nab --nab1_lab $N1/out/lab_qwen3_0p6b.json --max_len 8800 --chunk 256 --smoke --items 0 || exit 1
LONG=$($PY -I -c "
import json
t=[json.loads(l) for l in open('$N1/out/traces_qwen3_4b.jsonl') if l.strip()]
print(max(t, key=lambda r: len(r['prompt_ids'])+len(r['gen_ids']))['i'])") || exit 1
free_ok || exit 4
CUDA_VISIBLE_DEVICES=0 $PY -I t1_lab.py $M/Qwen3-4B $N1/out/traces_qwen3_4b.jsonl smoke/t1_qwen3_4b.smoke.json \
  --stack $HOME/work/nab --nab1_lab $N1/out/lab_qwen3_4b.json --max_len 8800 --chunk 256 --smoke --items $LONG || exit 1
$PY -I -c "
import json, math, os
for f in ('smoke/t1_qwen3_0p6b.smoke.json', 'smoke/t1_qwen3_4b.smoke.json'):
    it = json.load(open(f))['items'][0]
    assert it['q0a_equal'] is True, f
it = json.load(open('smoke/t1_qwen3_4b.smoke.json'))['items'][0]
g = int(math.ceil(it['peak_reserved_mib'] + 1536))
with open('smoke/gate_mib.tmp', 'w') as fh:
    fh.write(str(g) + chr(10))
    fh.flush()
    os.fsync(fh.fileno())
os.replace('smoke/gate_mib.tmp', 'smoke/gate_mib')
print('4B item', it['i'], 'L', it['L'], 'peak alloc', round(it['peak_alloc_mib']), 'reserved',
      round(it['peak_reserved_mib']), 'gate', g)" || exit 1
[ "$(md5sum FREEZE.md5)" = "$F0" ] || { echo "FREEZE.md5 changed during the smoke"; exit 2; }
md5sum -c --quiet FREEZE.md5 || { echo "frozen files changed during the smoke"; exit 2; }
echo "$F0" > smoke/PASS.tmp && sync smoke/PASS.tmp && mv smoke/PASS.tmp smoke/PASS || exit 1
echo "$(date -u +%FT%TZ) SMOKE PASS"
