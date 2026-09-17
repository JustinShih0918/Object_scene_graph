#!/usr/bin/env bash
# Top up the thin mapping scenes, re-audit, then run pass 2.
#
# The 3-episode budget was calibrated on 00800's OLD static layout (docs
# CROSS_ANCHOR_STATUS 4b-I). Re-authoring moved the statics, and a mapping
# episode starts on its target's storey, so the 3 episodes that survive the cap
# no longer sample both storeys: 00878's 2.80 m storey went entirely unmapped
# and took 6 authored targets with it. The maps that PASSED this audit before
# had 8 snapshots on 00808 and 00878, so that is the number to restore.
#
# `run_prior_maps.sh` is count-aware, so this TOPS UP from 3 rather than
# redoing the three episodes already on disk.
set -u
cd "$(dirname "$0")/.."
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "starting the perception servers"
bash scripts/serve_perception.sh >/dev/null 2>&1
for _ in $(seq 1 40); do
  up=0
  for p in 13182 13183 13184 13185 13186; do
    c=$(curl -s -o /dev/null -m 2 -w '%{http_code}' "http://localhost:$p/" 2>/dev/null)
    [ -n "$c" ] && [ "$c" != "000" ] && up=$((up+1))
  done
  [ "$up" -ge 5 ] && break
  sleep 15
done
log "servers responding: ${up}/5"
[ "$up" -lt 5 ] && { log "servers did not come up -- stopping"; exit 1; }

log "topping the thin scenes up to 8 episodes"
OBS=outputs/maps_p1500b_ascent OSG=outputs/maps_p1500b_osg \
  OUT=outputs/prior1500b MAX_EPISODES=8 \
  bash scripts/run_prior_maps.sh 00800-TEEsavR23oF 00808-y9hTuugGdiq 00878-XB4GS9ShBRE
log "top-up finished"

log "stopping the perception servers (pass 2 needs the whole card)"
bash scripts/serve_perception.sh --stop >/dev/null 2>&1
sleep 20

log "coverage audit"
python scripts/audit_map_coverage.py \
  --maps outputs/maps_p1500b_ascent --graphs outputs/maps_p1500b_osg \
  --layout-root /habitat-data-collector/outputs/dualmap_multifloor \
  --layout-root outputs/collector_layouts \
  --scenes 00800-TEEsavR23oF,00808-y9hTuugGdiq,00821-eF36g7L6Z9M,00873-bxsVRursffK,00878-XB4GS9ShBRE \
  --out outputs/audit/map_coverage_1500c.json --label 1500c-topup
log "audit exit=$?"

# Gate on the SEVERE failure only -- a storey that holds authored targets and
# has no map at all. A marginal free-fraction is worth running and reporting
# (00878 measured 0.600 against a 0.60 gate on the maps that produced this
# arm's only 00878 results); a goal storey that does not exist in the map
# cannot be navigated to at any budget, and would burn 97 minutes proving it.
python - <<'PY'
import json, sys
r = json.load(open("outputs/audit/map_coverage_1500c.json"))
bad = []
for name, sc in sorted((r.get("scenes") or {}).items()):
    for st in sc.get("storeys") or []:
        if float(st.get("free_frac") or 0.0) <= 0.0 and int(st.get("n_targets") or 0) > 0:
            bad.append(f"{name} storey y={st.get('height')} holds "
                       f"{st['n_targets']} target(s) and has NO map")
print("SEVERE:", *bad, sep="\n  ") if bad else print("no unmapped goal storey")
sys.exit(3 if bad else 0)
PY
if [ $? -eq 3 ]; then
  log "SEVERE gate tripped -- pass 2 NOT started"; exit 1
fi

log "starting pass 2"
OBS=outputs/maps_p1500b_ascent OSG=outputs/maps_p1500b_osg \
  OUT=outputs/mf5_pass2_reauth MAX_PARALLEL=3 \
  bash scripts/run_mf5_pass2.sh 00800-TEEsavR23oF 00808-y9hTuugGdiq \
    00821-eF36g7L6Z9M 00873-bxsVRursffK 00878-XB4GS9ShBRE
log "chain done"
