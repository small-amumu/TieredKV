#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Zijie Hu
# OPX formal runs on STAR (PLAN_OPX.md). usage: setsid nohup bash opx_formal_star.sh <phase> <gpu> > /dev/null 2>&1 < /dev/null &
#   acc    competitors, accuracy mode, ox set (5 tasks x 50 items, 130K), 7 arms one after another on GPU <gpu>
#   calib  official full-attention paths at 32K (lc set vt / fwe / qa_1, items 0-9): ShadowKV full, RetroInfer Full_Flash_Attn
#   tim    timing subset (ox items 0-3 of each task), every arm incl. ours, after warm-up runs (ShadowKV 2048, RetroInfer
#          0.018, ours all four widths);
#          every arm waits until no compute process of anyone is on ANY GPU of the host
#   mem    host memory: skv2048, ret018, ours h54 on lc vt item 0, full prompt and --trunc 16000, --mem_probe
# One lock for the whole host (phases never overlap). Before every arm: wait until the card (tim: the host) has no compute
# process; nvidia-smi failures count as busy. An arm whose output has an end record is skipped; an unfinished output is
# moved to <name>.partialN (with its log and traces) and the arm reruns. Refusals go to formal/launch.log, the rc of
# every run to <phase>/done.log, the compute processes every 2 s to <phase>/gpuwatch.log.
set -u
PH=${1:-}; G=${2:-}
H=$HOME/work; S=$H/sota; X=$H/opx; B=$X/rp1
mkdir -p "$X/formal"
LL=$X/formal/launch.log
say() { echo "$(date -u +%FT%TZ) phase=$PH gpu=$G: $*" >> "$LL"; }
case "$PH" in acc|calib|tim|mem) ;; *) say "bad phase"; exit 2;; esac
case "$G" in 0|1) ;; *) say "bad gpu"; exit 2;; esac
exec 9> /tmp/opx_formal_star.lock
flock -n 9 || { say "another OPX phase holds the host lock: refused"; exit 1; }
SELF=$(readlink -f "$0")
O=$X/formal/$PH
mkdir -p "$O" "$B/tmp"
LOG=$O/done.log
U=$(nvidia-smi -i "$G" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d " ")
[ -n "$U" ] || { say "no uuid for GPU$G: refused"; exit 2; }
free_gb=$(df -BG --output=avail "$O" | tail -1 | tr -dc 0-9)
[ "${free_gb:-0}" -ge 30 ] || { say "only ${free_gb} GB free on $O: refused"; exit 2; }
busy() {  # prints the compute processes that block the next arm; nvidia-smi failure prints FAIL
  local a
  a=$(timeout 30 nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null) || { echo FAIL; return; }
  a=$(echo "$a" | tr -d " " | grep -v "^$")
  if [ "$PH" = tim ]; then echo "$a"; else echo "$a" | grep "^$U," ; fi
}
wait_free() {  # wait_free <arm>: block until busy() is empty, logging when waiting starts and ends
  local b w=0
  while :; do
    b=$(busy)
    [ -z "$b" ] && break
    # first wait and then one heartbeat per hour, so a long wait is distinguishable from a hung launcher
    [ $((w % 60)) = 0 ] && echo "$(date -u +%FT%TZ) $1 waiting: $(echo $b | tr '\n' ' ')" >> "$LOG"
    w=$((w + 1)); sleep 60
  done
  [ $w -gt 0 ] && echo "$(date -u +%FT%TZ) $1 card free, starting" >> "$LOG"
  return 0
}
( exec 9>&-; while :; do s=$(timeout 30 nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader \
    2>/dev/null) || s=FAIL; echo "$(date -u +%T) $(echo "$s" | tr '\n' ';')"; sleep 2; done ) >> "$O/gpuwatch.log" 2>&1 &
WATCH=$!
FAILS=""
FIN=0
trap 'kill $WATCH 2>/dev/null; [ "$FIN" = 1 ] || echo "$(date -u +%FT%TZ) ABORTED" >> "$LOG"' EXIT
echo "$(date -u +%FT%TZ) start phase=$PH gpu$G $U md5 $(md5sum $X/opx_skv.py $X/opx_retro.py $X/opx_common.py \
  $X/pcie_meter.py $B/engine7/opx_rp_run.py $B/engine7/opx_engine_v8.py $B/engine7/opx_graph.py $B/engine7/opx_common.py \
  $B/engine7/pcie_meter.py $B/kalign/kalign_opx.py "$SELF" | awk '{printf "%s:%s ", $2, substr($1,1,8)}')" >> "$LOG"
export CUDA_VISIBLE_DEVICES=$U HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PKV5_ROWS
OX=$S/data/ox130k; LC=$S/data/ruler130k; L32=$S/data/ruler32k
M=$H/models/Meta-Llama-3.1-8B-Instruct; ML=$S/mlink/Llama-3.1-8B-Instruct
TASKS=vt,cwe,fwe,qa_1,qa_2
cd "$X" || exit 2
ITEMERR=""
is_complete() { grep -q '"end": true' "$O/$1.jsonl" 2>/dev/null; }
prep() {  # prep <name>: 0 = run it, 1 = already complete (end record present)
  local k=1 x moved=""
  is_complete "$1" && { echo "$(date -u +%FT%TZ) $1 complete, skipped" >> "$LOG"; return 1; }
  wait_free "$1"                                   # wait first: an orphaned run may still be writing this output
  is_complete "$1" && { echo "$(date -u +%FT%TZ) $1 complete, skipped" >> "$LOG"; return 1; }
  while [ -e "$O/$1.jsonl.partial$k" ] || [ -e "$O/$1.log.partial$k" ] || [ -e "$O/$1.traces.partial$k" ]; do
    k=$((k + 1)); done
  for x in jsonl log traces; do                    # leftovers of an unfinished or failed-at-load run
    [ -e "$O/$1.$x" ] && { mv "$O/$1.$x" "$O/$1.$x.partial$k"; moved="$moved $x"; }
  done
  [ -n "$moved" ] && echo "$(date -u +%FT%TZ) $1 leftovers ($moved ) moved to .partial$k, rerunning" >> "$LOG"
  return 0
}
done_rc() {  # done_rc <name> <rc>: failed = rc != 0 or no end record; items with an error key are counted and listed
  local n
  echo "$(date -u +%FT%TZ) $1 rc=$2" >> "$LOG"
  if [ "$2" != 0 ] || ! is_complete "$1"; then FAILS="$FAILS $1"; fi
  n=$(grep '"error": ' "$O/$1.jsonl" 2>/dev/null | grep -vc '"skipped_length"'); n=${n:-0}
  [ "$n" -gt 0 ] && { echo "$(date -u +%FT%TZ) $1 has $n item records with errors" >> "$LOG"; ITEMERR="$ITEMERR $1:$n"; }
  return 0
}
skv() {  # skv <name> <data> <len> <tasks> <n> [flags]
  local name=$1 d=$2 len=$3 tasks=$4 n=$5 rc; shift 5
  prep "$name" || return 0
  PATH=$S/venv_skv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=8.9 \
  $S/venv_skv/bin/python -u opx_skv.py $M "$d" "$O/$name.jsonl" --len $len --tasks $tasks --n $n "$@" > "$O/$name.log" 2>&1
  rc=$?; done_rc "$name" $rc; }
retro() {  # retro <name> <data> <len> <tasks> <n> [flags]
  local name=$1 d=$2 len=$3 tasks=$4 n=$5 rc; shift 5
  prep "$name" || return 0
  $S/venv_retro/bin/python -u opx_retro.py $ML "$d" "$O/$name.jsonl" --len $len --tasks $tasks --n $n "$@" \
    > "$O/$name.log" 2>&1
  rc=$?; done_rc "$name" $rc; }
ours() {  # ours <name> <data files, comma list> <items> <widths> <length> [flags]
  local name=$1 d=$2 items=$3 w=$4 len=$5 rc; shift 5
  prep "$name" || return 0
  ( cd "$B/engine7" && KVT_ROOT=$B/root KALIGN_DIR=$B/kalign KALIGN_CAL=$B/diag_long_text.txt OPX_KERNEL=opx \
    TRITON_CACHE_DIR=$B/cache/triton CUDA_CACHE_PATH=$B/cache/nv XDG_CACHE_HOME=$B/cache HF_HOME=$B/cache/hf \
    TMPDIR=$B/tmp $H/venv/bin/python -B -I opx_rp_run.py $M "$O/$name.jsonl" --mode ruler --data "$d" \
      --items "$items" --widths "$w" --length "$len" --cap $((len + 2048)) --layout fixed --graph --slim --exact_pin "$@" \
      > "$O/$name.log" 2>&1 )
  rc=$?; done_rc "$name" $rc; }
oxfiles() { local s=""; for t in vt cwe fwe qa_1 qa_2; do s="$s,$OX/$t/validation.jsonl"; done; echo "${s#,}"; }
case "$PH" in
  acc)
    skv skv1024  $OX 130000 $TASKS 50 --sparse_budget 1024 --fix_v_stride
    skv skv2048  $OX 130000 $TASKS 50 --sparse_budget 2048 --fix_v_stride
    skv skv4096  $OX 130000 $TASKS 50 --sparse_budget 4096 --fix_v_stride
    skv skv2048_official $OX 130000 $TASKS 50 --sparse_budget 2048
    retro ret009 $OX 130000 $TASKS 50 --retrieval_budget 0.009
    retro ret018 $OX 130000 $TASKS 50 --retrieval_budget 0.018
    retro ret036 $OX 130000 $TASKS 50 --retrieval_budget 0.036
    ;;
  calib)
    # ShadowKV full keeps the whole KV on the GPU; within one process the 2nd item already OOMs at 32K (2026-11-23
    # 14:47Z, as in X-0140 Part C): one fresh process per item, then the item records are merged into skv_full.jsonl
    if ! is_complete skv_full; then
      wait_free skv_full
      [ -e "$O/skv_full.jsonl" ] && mv "$O/skv_full.jsonl" "$O/skv_full.jsonl.partial$(date -u +%s)"
      parts=""; allok=1
      for t in vt fwe qa_1; do
        for i in 0 1 2 3 4 5 6 7 8 9; do
          skv "skv_full_${t}_$i" $L32 32768 $t 1 --offset $i --method full     # complete parts are skipped
          parts="$parts $O/skv_full_${t}_$i.jsonl"
          is_complete "skv_full_${t}_$i" || allok=0
        done
      done
      { echo '{"meta": true, "merged_from": "skv_full_<task>_<i>.jsonl, one process per item"}'
        for f in $parts; do grep -v '"meta": true' "$f" 2>/dev/null; done
        [ $allok = 1 ] && echo '{"meta": true, "end": true}'; } > "$O/skv_full.jsonl"
      echo "$(date -u +%FT%TZ) skv_full merged from $(echo $parts | wc -w) per-item files, all complete=$allok" >> "$LOG"
      [ $allok = 1 ] || FAILS="$FAILS skv_full"
    fi
    retro retro_full $L32 32768 vt,fwe,qa_1 10 --attn Full_Flash_Attn
    ;;
  tim)
    # warm-up on lc vt item 0 (not in the ox set): Triton / flashinfer / CUDA caches of this card (issue 163);
    # every width of ours, since kernels specialise per width; outputs kept, never read
    skv warm_skv $LC 130000 vt 1 --sparse_budget 2048 --fix_v_stride --timing
    retro warm_retro $LC 130000 vt 1 --retrieval_budget 0.018 --timing
    ours warm_ours $LC/vt/validation.jsonl 0:1 32,33,43,54 130000 --timing
    for b in 1024 2048 4096; do skv tim_skv$b $OX 130000 $TASKS 4 --sparse_budget $b --fix_v_stride --timing; done
    skv tim_skv2048_official $OX 130000 $TASKS 4 --sparse_budget 2048 --timing
    for r in 0.009 0.018 0.036; do retro tim_ret${r#0.} $OX 130000 $TASKS 4 --retrieval_budget $r --timing; done
    ours tim_ours "$(oxfiles)" 0:4 32,33,43,54 130000 --timing
    ;;
  mem)
    for tr in 0 16000; do
      if [ $tr = 0 ]; then len=130000; tag=full; else len=16384; tag=16k; fi
      skv mem_skv2048_$tag $LC $len vt 1 --sparse_budget 2048 --fix_v_stride --trunc $tr --mem_probe
      retro mem_ret018_$tag $LC $len vt 1 --retrieval_budget 0.018 --trunc $tr --mem_probe
      ours mem_ours_h54_$tag $LC/vt/validation.jsonl 0:1 54 $len --trunc $tr --mem_probe
    done
    ;;
esac
FIN=1
if [ -z "$FAILS" ]; then echo "$(date -u +%FT%TZ) ALL DONE ok; item errors:${ITEMERR:- none}" >> "$LOG"
else echo "$(date -u +%FT%TZ) FINISHED WITH FAILURES:$FAILS; item errors:${ITEMERR:- none}" >> "$LOG"; fi
