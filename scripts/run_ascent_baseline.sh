#!/usr/bin/env bash
# The ASCENT baseline over the 26 cross-anchor episodes -- PARALLEL.
#
# The old header said two Habitat processes at once ends in a CUDA OOM. That was
# written when the five perception servers held ~16 GB; measured now they hold
# 11.3 and each Habitat client only 1.7, leaving 11.5 GB free. More to the
# point, the models are memory-resident but compute-IDLE: sampled during a live
# ASCENT episode the GPU ran 1-41% (mean ~19%) and the client used 2.4 of 32
# CPU cores, because the cost is CPU-bound simulation plus HTTP round-trips,
# not model inference. So the servers can feed several clients at once.
#
# 3, not 4: the servers grow when exercised (~16 GB per the docs) and
# 16 + 4x1.7 = 22.8 of 24.5 GB leaves no margin. `wait_for_vram` is the guard.
#
# Measured serial cost: 1.82 steps/s -> 9.2 min per 1000-step episode, ~4 h for
# 26. At 3-wide that should be ~1h20m.
set -u
cd "$(dirname "$0")/.."
set -a; [ -f docker/.env ] && . ./docker/.env; set +a
OUT=${OUT:-outputs/ascent_crossanchor}
# SERIAL. Not a resource limit -- a CORRECTNESS one. The perception servers
# hold ONE predictor object with per-image state: MobileSAM's `set_image`
# stashes `self.features`, and `predict_torch` reads it. Two clients interleave
# those calls and the second one's set_image lands between the first one's
# set_image and predict, so the decoder gets image_embeddings=None and the
# request 500s. Measured: zero 500s across the serial window 03:23-03:33, then
# four within six minutes of a third client starting, killing 3 of 5 scenes.
#
# Idle GPU (19% mean) says there IS spare capacity; it does NOT say the servers
# are safe to share. Parallelising ASCENT needs one server SET per client, not
# more clients per set.
MAX_PARALLEL=${MAX_PARALLEL:-1}
MIN_FREE_MIB=${MIN_FREE_MIB:-3500}
# Step budget. 500 halves the ~4 h a 1000-step serial pass needs, and the OSG
# arm is re-run at the SAME cap so neither side is advantaged -- a baseline at
# half the budget of the system it is compared against is not a baseline.
STEPS=${STEPS:-500}
mkdir -p "$OUT"; rm -f "$OUT/.failed"
log() { echo "[$(date +%H:%M:%S)] $*"; }

wait_for_vram() {
  for _ in $(seq 1 240); do
    local used total
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    [ $(( total - used )) -ge "$MIN_FREE_MIB" ] && return 0
    sleep 20
  done
  log "    WARNING: never saw ${MIN_FREE_MIB} MiB free; starting anyway"
}

log "starting perception servers (idempotent -- reuses a live session)"
bash scripts/serve_perception.sh >/dev/null 2>&1
for _ in $(seq 1 40); do
  up=0
  for p in 13182 13183 13184 13185 13186; do
    c=$(curl -s -o /dev/null -m 2 -w '%{http_code}' "http://localhost:$p/" 2>/dev/null)
    [ -n "$c" ] && [ "$c" != "000" ] && up=$((up+1))
  done
  [ "$up" -ge 5 ] && break; sleep 15
done
log "servers ${up}/5 | step budget ${STEPS} | MAX_PARALLEL=${MAX_PARALLEL}"
[ "$up" -lt 5 ] && { log "servers did not start -- ASCENT cannot run"; exit 1; }

# FEWEST EPISODES FIRST. Serial ASCENT is ~9.2 min an episode (1.82 steps/s
# measured over 74 episodes), so 26 episodes is ~4 h and a deadline may cut it
# short. This order maximises how many SCENES are complete when it does, and a
# complete scene pairs with the OSG arm; a half-finished one does not.
#   00800 2 eps | 00873 3 | 00878 6 | 00862 7 | 00821 8
for scene in 00800-TEEsavR23oF 00873-bxsVRursffK 00878-XB4GS9ShBRE \
             00862-LT9Jq6dN3Ea 00821-eF36g7L6Z9M; do
  if [ -f "$OUT/$scene/summary.json" ]; then log "=== $scene already run, skipping"; continue; fi
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 20; done
  wait_for_vram
  log "=== $scene"
  (
    python scripts/run_eval.py +experiment=mf5_ascent_crossanchor \
      "ycb.scenes=[$scene]" agent.max_steps=$STEPS eval.debug_frames=true \
      "output_dir=$OUT/$scene" > "$OUT/$scene.log" 2>&1
    status=$?
    sr=$(grep -o '"success_rate": [0-9.]*' "$OUT/$scene.log" | tail -1)
    if [ "$status" -ne 0 ]; then
      echo "$scene exit=$status" >> "$OUT/.failed"
      echo "[$(date +%H:%M:%S)]     $scene FAILED exit=$status -- $(grep -E 'Error|Exception' "$OUT/$scene.log" | tail -1)"
    else
      echo "[$(date +%H:%M:%S)]     $scene exit=0 $sr"
    fi
  ) &
  sleep 30   # stagger the scene loads
done
wait

bash scripts/serve_perception.sh --stop >/dev/null 2>&1
if [ -s "$OUT/.failed" ]; then log "=== ASCENT baseline had FAILURES:"; cat "$OUT/.failed"; exit 1; fi
log "=== ASCENT baseline complete"
