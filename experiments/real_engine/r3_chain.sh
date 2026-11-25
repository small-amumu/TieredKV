#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The TieredKV Authors
# R3 chain (pro5000): wait for R1's driver on this card to exit (exact PID), then run R3's steps on the card. The first card
# to come free claims the role "first" (atomic mkdir); the other card gets "second".
#   first : D0 (r3_fpdrift.sh), the smoke (cwe item 0, h54), the mechanical smoke check, then the formal tasks vt and fwe.
#   second: wait (at most 10 h) for the smoke verdict file, then the formal task qa_1 if the smoke passed.
# A driver refused only because the card is busy or locked (exit 4) is retried every 60 s, at most 30 times.
# The first chain always leaves a smoke verdict file when it stops (trap), so the second chain cannot wait forever.
# Usage: bash r3_chain.sh <gpu> <R1 driver pid on that card>
set -u
EG=/home/user/kvt_bench/hc_mac/rp1/engine7
G=${1:-}; WP=${2:-}
case "$G" in 0|1|2|3) ;; *) echo "GPU arg must be 0-3" >&2; exit 2;; esac
case "$WP" in ''|*[!0-9]*) echo "wait pid must be numeric" >&2; exit 2;; esac
L=$EG/chain_gpu$G.log
echo "$(date -u +%FT%TZ) CHAIN start gpu=$G wait_pid=$WP md5=$(md5sum $EG/r3_chain.sh | cut -c1-32)" >> $L
while kill -0 $WP 2>/dev/null; do sleep 60; done
if mkdir $EG/.chain_first_claim 2>/dev/null; then ROLE=first; else ROLE=second; fi
echo "$(date -u +%FT%TZ) pid $WP gone; role $ROLE" >> $L
cd $EG || exit 2
retry() {                                  # run a driver; retry only on exit 4 (card busy / locked)
  local n=0 rc
  while :; do
    "$@"; rc=$?
    [ $rc -ne 4 ] && return $rc
    n=$((n + 1)); [ $n -ge 30 ] && return $rc
    echo "$(date -u +%FT%TZ) $* refused (busy/locked), retry $n" >> $L; sleep 60
  done
}
sleep 30                                   # let the card's memory drain before the idle checks
if [ "$ROLE" = first ]; then
  mkdir -p smoke
  trap '[ -e smoke/SMOKE_OK ] || [ -e smoke/SMOKE_FAIL ] || echo "{\"ok\": false, \"why\": [\"first chain stopped before the smoke check\"]}" > smoke/SMOKE_FAIL' EXIT
  retry bash r3_fpdrift.sh $G; echo "$(date -u +%FT%TZ) fpdrift rc=$?" >> $L
  sleep 20
  retry bash r3_driver.sh $G vt smoke; echo "$(date -u +%FT%TZ) smoke driver rc=$?" >> $L
  /usr/bin/python3 -I r3_smoke_check.py $EG >> $L 2>&1; sc=$?
  echo "$(date -u +%FT%TZ) smoke check rc=$sc" >> $L
  [ $sc -eq 0 ] || { echo "$(date -u +%FT%TZ) CHAIN first stop: smoke failed" >> $L; exit 7; }
  sleep 20
  retry bash r3_driver.sh $G vt,fwe; echo "$(date -u +%FT%TZ) formal vt,fwe driver rc=$?" >> $L
else
  t0=$(date +%s)
  while [ ! -e smoke/SMOKE_OK ] && [ ! -e smoke/SMOKE_FAIL ]; do
    [ $(( $(date +%s) - t0 )) -gt 36000 ] && { echo "$(date -u +%FT%TZ) CHAIN second stop: no smoke verdict in 10 h" >> $L; exit 8; }
    sleep 60
  done
  [ -e smoke/SMOKE_OK ] || { echo "$(date -u +%FT%TZ) CHAIN second stop: smoke failed" >> $L; exit 7; }
  retry bash r3_driver.sh $G qa_1; echo "$(date -u +%FT%TZ) formal qa_1 driver rc=$?" >> $L
fi
echo "$(date -u +%FT%TZ) CHAIN $ROLE END" >> $L
