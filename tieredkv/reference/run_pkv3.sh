#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
# pkv runner v3: like run_pkv2.sh but the GPU set comes from $GPUS (default "1 3": GPU0 is shared with another user's
# large jobs that grab memory after our free-memory check), and every model whose run fails is rerun once at the end
# (old logs moved to <tag>/fail2/). Ends with a DONE3 line in the run.log.
# Usage: GPUS="1 3" run_pkv3.sh <listfile> <tag> <script> [extra args...]
cd ~/kvt_bench/hc_mac
L=$1; TAG=$2; SCR=$3; shift 3
R=results/pkv/$TAG; mkdir -p $R; GP=${GPUS:-"1 3"}

runq() {
  local LF=$1; shift
  while read -r m name dt need; do g=""
    while [ -z "$g" ]; do for c in $GP; do exec 9> /tmp/hcmac_gpu$c.lock
        if flock -n 9; then f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $c | tr -d " ")
          if [ $((f / 1024)) -ge $need ]; then g=$c; break; fi; exec 9>&-; fi; done; [ -z "$g" ] && sleep 30; done
    echo "$(date -u +%FT%TZ) $name start gpu $g" >> $R/run.log
    CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 PYTHONHASHSEED=0 ~/vllm_main/bin/python -I code/$SCR $m $R/$name.json --dtype $dt "$@" > $R/$name.out 2> $R/$name.err
    rc=$?; echo "$(date -u +%FT%TZ) $name rc=$rc" >> $R/run.log; exec 9>&-
  done < $LF
}

runq $L "$@"
F=code/${TAG}_redo3.txt; : > $F
for name in $(awk '{print $2}' $L); do
  st=$(grep " $name rc=" $R/run.log | tail -n 1 | sed 's/.*rc=//')
  if [ "$st" != 0 ]; then grep " $name " $L >> $F; mkdir -p $R/fail2; mv $R/$name.* $R/fail2/ 2>/dev/null; fi
done
[ -s $F ] && runq $F "$@"
echo "$(date -u +%FT%TZ) DONE3" >> $R/run.log
