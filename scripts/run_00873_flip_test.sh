#!/usr/bin/env bash
# One trial: make 00873's DESCENT the master chef can (50005) instead of the
# cracker box, and measure that episode alone.
#
# 50005 was an ascent and ascents are 0/11 in this arm. Its cross_anchor pose
# already sat on `table_133`, the ONLY food-affordable anchor on 00873's upper
# storey, so turning it into a descent forces 50001 off that anchor and into an
# ascent -- the scene's one-descent budget moves rather than grows.
#
# The prior MUST be rebuilt: `static` moved for both targets, and pass 2's whole
# premise is that the scene graph says where pass 1 saw the object. Reusing the
# old graph would point the agent at a position nothing was ever authored at.
set -u
cd "$(dirname "$0")/.."
set -a; [ -f docker/.env ] && . ./docker/.env; set +a
log() { echo "[$(date +%H:%M:%S)] $*"; }
S=00873-bxsVRursffK

log "starting perception servers"
bash scripts/serve_perception.sh >/dev/null 2>&1
for _ in $(seq 1 40); do
  up=0
  for p in 13182 13183 13184 13185 13186; do
    c=$(curl -s -o /dev/null -m 2 -w '%{http_code}' "http://localhost:$p/" 2>/dev/null)
    [ -n "$c" ] && [ "$c" != "000" ] && up=$((up+1))
  done
  [ "$up" -ge 5 ] && break; sleep 15
done
log "servers ${up}/5"; [ "$up" -lt 5 ] && exit 1

log "pass 1: remapping $S onto the flipped static layout"
python scripts/run_eval.py +experiment=mf5_mapping_both \
  "ycb.scenes=[$S]" agent.max_steps=1500 eval.num_episodes=3 \
  ycb.obstacle_map_out=outputs/maps_00873flip_ascent \
  ycb.map_out=outputs/maps_00873flip_osg \
  ycb.obstacle_map_union=true eval.behaviour_log=true eval.save_viz=false \
  "output_dir=outputs/prior_00873flip" > outputs/prior_00873flip.log 2>&1
log "pass 1 exit=$? snapshots=$(ls outputs/maps_00873flip_ascent/${S}__*.json 2>/dev/null | wc -l)"

log "stopping servers"
bash scripts/serve_perception.sh --stop >/dev/null 2>&1
sleep 20

log "pass 2: the single episode 50005 (master chef can, now a DESCENT)"
python scripts/run_eval.py +experiment=mf5_osg_on_ascent_map \
  "ycb.scenes=[$S]" \
  "eval.episode_ids=[${S}__cross_anchor_01__50005__s0]" \
  ycb.obstacle_map_in=outputs/maps_00873flip_ascent ycb.obstacle_map_union=true \
  ycb.seed_storeys_from_obstacle_map=true ycb.storey_seed_tol_m=0.75 \
  ycb.map_in=outputs/maps_00873flip_osg ycb.map_in_occupancy=false \
  eval.debug_frames=true "output_dir=outputs/mf5_00873flip" \
  > outputs/mf5_00873flip.log 2>&1
log "pass 2 exit=$?"
log "flip test done"
