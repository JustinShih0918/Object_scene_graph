#!/usr/bin/env bash
# The whole cross-anchor protocol, end to end, unattended.
#
#   nohup bash scripts/run_crossanchor_pipeline.sh > outputs/pipeline.log 2>&1 &
#
#   1. build both priors for all eight scenes   (ASCENT servers UP, serial)
#   2. audit the maps                            -- A HARD GATE
#   3. stop the model servers, free the card
#   4. pass 2 on the five multi-floor scenes     (servers DOWN, parallel)
#
# Stage 2 is a gate, not a report. If a scene fails it, the run STOPS with the
# audit on stdout and pass 2 is not started: the failures it catches -- a storey
# with no scene-graph layer, an authored target outside the mapped region, a
# stair ramp cut short -- are all things that make pass 2 unwinnable before it
# begins, and a 0% arm then looks like a bad algorithm rather than a bad map.
# Re-placing an off-map target is a judgement call (which layout, which anchor)
# and is deliberately left to a person.
#
# DualMap's benchmark is NOT here. Its evaluation needs
# `outputs/dualmap_official_bench/seed12`, which is not on this machine; its
# MAPPING pass is included above and needs nothing.
set -u
cd "$(dirname "$0")/.."
umask 002

OBS=${OBS:-outputs/maps_p1500_ascent}
OSG=${OSG:-outputs/maps_p1500_osg}
AUDIT=${AUDIT:-outputs/audit/map_coverage.json}
export OBS OSG
log() { echo "[$(date +%F' '%H:%M:%S)] == $*"; }

log "stage 1: priors"
bash scripts/run_prior_maps.sh || { log "STAGE 1 FAILED"; exit 1; }

log "stage 2: audit (the gate)"
mkdir -p "$(dirname "$AUDIT")"
python scripts/audit_map_coverage.py \
    --maps "$OBS" --graphs "$OSG" \
    --layout-root /habitat-data-collector/outputs/dualmap_multifloor \
    --layout-root outputs/collector_layouts --layout-indices 1 \
    --exempt-storey 00808-y9hTuugGdiq:0 \
    --exempt-storey 00878-XB4GS9ShBRE:0 \
    --out "$AUDIT" --label "${STEPS:-1500}-union"
status=$?
if [ "$status" -eq 2 ]; then
  log "AUDIT CHECKED NOTHING (exit 2) -- a missing snapshot, or a broken"
  log "invocation. Pass 2 NOT started. Report: $AUDIT"
  exit 2
fi

# PER SCENE, not all-or-nothing. The gate exists so a scene whose map cannot
# support the search is not run and then read as an algorithm failure -- and
# that argument is about a SCENE. Holding back four good ones because the fifth
# tripped a threshold would waste the night without making any number more
# honest. Each passing scene runs; each failing one is named and left for a
# person, because topping a scene up or re-placing a target is a judgement call.
MF5="00800-TEEsavR23oF 00808-y9hTuugGdiq 00821-eF36g7L6Z9M 00873-bxsVRursffK 00878-XB4GS9ShBRE"
# shellcheck disable=SC2086
PASSED=$(python scripts/audit_verdicts.py "$AUDIT" passed $MF5)
# shellcheck disable=SC2086
FAILED=$(python scripts/audit_verdicts.py "$AUDIT" failed $MF5)
if [ -n "$FAILED" ]; then
  log "scenes held back by the audit:"
  echo "$FAILED" | sed 's/^/      /'
fi
if [ -z "$PASSED" ]; then
  log "NO multi-floor scene passed the audit. Pass 2 NOT started."
  log "Report: $AUDIT"
  exit 1
fi
log "audit: pass 2 will run on -- $PASSED"

log "stage 3: stopping the model servers"
bash scripts/serve_perception.sh --stop
for _ in $(seq 1 20); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 6000 ] && break
  sleep 15
done
log "card at $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) after stop"

log "stage 4: pass 2 on the scenes the audit passed"
# shellcheck disable=SC2086
bash scripts/run_mf5_pass2.sh $PASSED || { log "PASS 2 FAILED"; exit 1; }

log "pipeline complete"
