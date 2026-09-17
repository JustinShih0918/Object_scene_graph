#!/usr/bin/env bash
# Pass 1 for 00862 in TWO episodes, one starting on each storey.
#
# Mapping coverage was a lottery: a mapping episode starts on its target's
# static storey (`ycb.start_on_prior_floor`) and only maps another one if it
# happens to climb. That is why 00878's 2.80 m storey came out with NO map at
# all -- exactly 1 of its 8 episodes ever went up -- and why the fix was 8
# episodes per scene, ~90 minutes, to buy the odds.
#
# Choosing the starts removes the lottery instead of paying for it. 00862's
# statics put 50001 upstairs (+3.31) and 50003 downstairs (+0.11), so those two
# episodes seed both storeys by construction: ~24 minutes, not ~96.
set -u
cd "$(dirname "$0")/.."
set -a; [ -f docker/.env ] && . ./docker/.env; set +a
log() { echo "[$(date +%H:%M:%S)] $*"; }
S=00862-LT9Jq6dN3Ea

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
log "servers ${up}/5"; [ "$up" -lt 5 ] && { log "servers did not start"; exit 1; }

log "pass 1: two episodes -- 50001 starts UPPER (+3.31), 50003 starts LOWER (+0.11)"
python scripts/run_eval.py +experiment=mf5_mapping_both \
  "ycb.scenes=[$S]" agent.max_steps=1500 \
  "eval.episode_ids=[${S}__static__50001__s0,${S}__static__50003__s0]" \
  ycb.start_on_prior_floor=true \
  ycb.obstacle_map_out=outputs/maps_p1500_ascent \
  ycb.map_out=outputs/maps_p1500_osg \
  ycb.obstacle_map_union=true eval.behaviour_log=true eval.save_viz=false \
  "output_dir=outputs/prior_00862" > outputs/prior_00862.log 2>&1
log "pass 1 exit=$? snapshots=$(ls outputs/maps_p1500_ascent/${S}__*.json 2>/dev/null | wc -l)"

log "stopping servers"
bash scripts/serve_perception.sh --stop >/dev/null 2>&1
sleep 20

log "coverage audit"
python scripts/audit_map_coverage.py \
  --maps outputs/maps_p1500_ascent --graphs outputs/maps_p1500_osg \
  --layout-root /habitat-data-collector/outputs/dualmap_multifloor \
  --scenes $S --out outputs/audit/map_coverage_00862.json --label 00862-two-starts
log "audit exit=$?"
log "00862 map build done"
