#!/usr/bin/env bash
# 00862 twice: starting on the UPPER storey (the prior's, so the agent must
# descend) and on the LOWER one (the goal's, so the staircase is removed).
#
# The pair is the control this benchmark has been missing. Across three runs the
# arm reached the goal storey 16 of 26 times but closed on the object 6 times,
# and nothing so far separates "the stairs beat it" from "the approach beat it".
# Same scene, same map, same agent, one flag apart.
set -u
cd "$(dirname "$0")/.."
set -a; [ -f docker/.env ] && . ./docker/.env; set +a
log() { echo "[$(date +%H:%M:%S)] $*"; }
S=00862-LT9Jq6dN3Ea

while ! grep -q "00862 map build done" outputs/map_00862.log 2>/dev/null; do
  if grep -q "servers did not start" outputs/map_00862.log 2>/dev/null; then
    log "map build failed -- stopping"; exit 1
  fi
  sleep 60
done
log "map ready: $(ls outputs/maps_p1500_ascent/${S}__*.json 2>/dev/null | wc -l) snapshot(s)"

run () {  # name, extra override
  local name=$1 extra=$2
  log "=== $name"
  python scripts/run_eval.py +experiment=mf5_osg_on_ascent_map \
    "ycb.scenes=[$S]" $extra \
    ycb.obstacle_map_in=outputs/maps_p1500_ascent ycb.obstacle_map_union=true \
    ycb.seed_storeys_from_obstacle_map=true ycb.storey_seed_tol_m=0.75 \
    ycb.map_in=outputs/maps_p1500_osg ycb.map_in_occupancy=false \
    eval.debug_frames=true "output_dir=outputs/mf5_00862_${name}" \
    > "outputs/mf5_00862_${name}.log" 2>&1
  local st=$?
  log "    $name exit=$st $(grep -o '\"success_rate\": [0-9.]*' "outputs/mf5_00862_${name}.log" | tail -1)"
}

# UPPER: the normal cross-anchor episode -- start where the prior says the
# object is, discover it moved, go down.
run upper "ycb.start_on_prior_floor=true"
# LOWER: start already on the goal storey. No staircase in the problem.
run lower "ycb.start_on_target_floor=true"
log "both starts done"
