#!/usr/bin/env bash
# ASCENT over DualMap's released benchmark, one CONDITION at a time.
#
#   bash scripts/run_ascent_dualmap.sh
#
# cross_anchor (53) -> static (79) -> in_anchor (54) = 186 trials. The order is
# deliberate: cross_anchor is the condition this project is about, so it lands
# first and is usable on its own if the run is cut short.
#
# PARALLEL, but only because the cause of the earlier corruption was removed.
# The five servers share ONE MobileSAM predictor whose `set_image` / `predict`
# pair is a critical section; behind a threaded Flask, two clients interleaved
# it and the decoder got image_embeddings=None -- four HTTP 500s inside six
# minutes at three clients, killing 3 of 5 scenes. `model_api/sam_out.py` now
# holds a module-level lock across that pair, which orders access without
# changing what any model computes.
#
# Probed after the lock, against the LIVE run so its own SAM calls were
# interleaved too: 1/2/3/4 concurrent clients -> 3/6/9/12 requests, ALL 200,
# median latency 0.04 -> 0.13 s, and zero 500s on SAM or GroundingDINO.
#
# 3, not 4: the servers hold 11.3 GB and grow to ~16 when exercised; at 1.7 GB
# a client that is 16 + 3x1.7 = 21.1 of 24.5 GB, where 4 would be 22.8 and an
# OOM overnight costs more than the extra width earns.
#
# Measured cost: 1.82 steps/s, so a 500-step trial is ~4.6 min and 186 is
# ~14 h. Each condition is resumable on its own -- `run_dualmap_arm.sh` skips
# any trial already present under $OUT.
set -u
cd "$(dirname "$0")/.."
log() { echo "[$(date +%H:%M:%S)] $*"; }
OUT=${OUT:-outputs/ascent_dualmap}
MAX_PARALLEL=${MAX_PARALLEL:-3}

log "starting perception servers (idempotent)"
bash scripts/serve_perception.sh >/dev/null 2>&1
for _ in $(seq 1 40); do
  up=0
  for p in 13182 13183 13184 13185 13186; do
    c=$(curl -s -o /dev/null -m 2 -w '%{http_code}' "http://localhost:$p/" 2>/dev/null)
    [ -n "$c" ] && [ "$c" != "000" ] && up=$((up+1))
  done
  [ "$up" -ge 5 ] && break; sleep 15
done
log "servers ${up}/5 | MAX_PARALLEL=${MAX_PARALLEL}"
[ "$up" -lt 5 ] && { log "servers did not start -- ASCENT cannot run"; exit 1; }

# Re-verify the servers before EACH condition, and restart them if they are
# gone. Without this a server that dies during one condition silently turns the
# remaining two into a few minutes of instant failures -- measured: static and
# in_anchor "finished" in 7 and 5 minutes having written zero episodes.
servers_ok() {
  local up=0 p c
  for p in 13182 13183 13184 13185 13186; do
    c=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://localhost:$p/" 2>/dev/null)
    [ -n "$c" ] && [ "$c" != "000" ] && up=$((up+1))
  done
  [ "$up" -ge 5 ]
}

for cond in cross_anchor static in_anchor; do
  if ! servers_ok; then
    log "    servers not answering before ${cond}; restarting"
    bash scripts/serve_perception.sh >/dev/null 2>&1
    for _ in $(seq 1 40); do servers_ok && break; sleep 15; done
    servers_ok || { log "    servers will not come up -- stopping"; exit 1; }
    log "    servers back"
  fi
  n=$(python3 -c "import json;print(len(json.load(open('data/splits/dualmap_${cond}.json'))['trials']))")
  log "=== condition ${cond} (${n} trials)"
  ARM=ascent PRESET_PREFIX=dualmap_protocol_ \
    SPLIT="data/splits/dualmap_${cond}.json" OUT="$OUT" \
    CONDITIONS="$cond" MAP_ROOT= MAX_PARALLEL="$MAX_PARALLEL" CHUNK=4 \
    bash scripts/run_dualmap_arm.sh
  # Second pass: `run_dualmap_arm.sh` skips whatever is already on disk, so this
  # retries exactly the trials that were lost to a transient failure and costs
  # nothing when there are none.
  if servers_ok; then
    ARM=ascent PRESET_PREFIX=dualmap_protocol_ \
      SPLIT="data/splits/dualmap_${cond}.json" OUT="$OUT" \
      CONDITIONS="$cond" MAP_ROOT= MAX_PARALLEL="$MAX_PARALLEL" CHUNK=4 \
      bash scripts/run_dualmap_arm.sh
  fi
  done_n=$(find "$OUT" -name episodes.jsonl -exec cat {} + 2>/dev/null | grep -c . || echo 0)
  log "    ${cond} finished; ${done_n} episodes on disk so far"
done

bash scripts/serve_perception.sh --stop >/dev/null 2>&1
log "=== ASCENT dualmap complete"
