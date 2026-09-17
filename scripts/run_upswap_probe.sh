#!/usr/bin/env bash
# Two of the just-swapped episodes, as ascents, to see whether UP works now.
#
# 00878/50002 comes from the scene that scored 5/6 on descents -- if the ascent
# fails there, direction is the cause, not the scene. 00862/50006 never left the
# upper storey as a descent (dy 0.00), so as an ascent it starts on the floor it
# could not leave; if it now climbs, the phantom storey at +1.477 blocks one
# direction only.
set -u
cd "$(dirname "$0")/.."
set -a; [ -f docker/.env ] && . ./docker/.env; set +a
log() { echo "[$(date +%H:%M:%S)] $*"; }

run () {
  local scene=$1 sid=$2
  log "=== $scene ep$sid (now UP)"
  python scripts/run_eval.py +experiment=mf5_osg_on_ascent_map \
    "ycb.scenes=[$scene]" \
    "eval.episode_ids=[${scene}__cross_anchor_01__${sid}__s0]" \
    ycb.obstacle_map_in=outputs/maps_p1500_ascent ycb.obstacle_map_union=true \
    ycb.seed_storeys_from_obstacle_map=true ycb.storey_seed_tol_m=0.75 \
    ycb.map_in=outputs/maps_p1500_osg ycb.map_in_occupancy=false \
    eval.debug_frames=true "output_dir=outputs/upswap_${scene:0:5}_${sid}" \
    > "outputs/upswap_${scene:0:5}_${sid}.log" 2>&1
  log "    exit=$? $(grep -o '\"success_rate\": [0-9.]*' "outputs/upswap_${scene:0:5}_${sid}.log" | tail -1)"
}

run 00878-XB4GS9ShBRE 50002
run 00862-LT9Jq6dN3Ea 50006
log "upswap probe done"
