#!/usr/bin/env bash
# Map the MISSING storeys directly, instead of topping every scene up blindly.
#
# The audit failed because whole storeys had no map, and the cause is not the
# episode COUNT but WHICH episode runs: a mapping episode starts on its
# target's storey and may or may not climb. Per-episode coverage, read off the
# snapshots themselves (`floor_summaries`):
#
#   00878  2.80 m storey  <- ONLY ep50001 of eight ever reached it
#   00808  3.26 m storey  <- ep50003 and ep50006 reach it; the first three do not
#
# So two episodes close both gaps, against the ~15 a top-up to 8 would run.
# `eval.episode_ids` selects them exactly (config/eval.py:47).
#
# This is safe to target because the walks are DETERMINISTIC: the start pose is
# seeded from `ycb.seed + semantic_id*1009` and pinned to the target's storey,
# so re-running an episode whose target did not move reproduces it -- measured,
# three 00878 episodes came back with byte-identical cell counts (78273, 38739,
# 23666) after the re-authoring.
set -u
cd "$(dirname "$0")/.."
set -a; [ -f docker/.env ] && . ./docker/.env; set +a
log() { echo "[$(date +%H:%M:%S)] $*"; }

fill() {  # scene, episode_id suffix, why
  local scene=$1 sid=$2 why=$3
  local ep="${scene}__static__${sid}__s0"
  log "=== $scene ep$sid ($why)"
  python scripts/run_eval.py +experiment=mf5_mapping_both \
    "ycb.scenes=[$scene]" agent.max_steps=1500 \
    "eval.episode_ids=[$ep]" \
    ycb.obstacle_map_out=outputs/maps_p1500b_ascent \
    ycb.map_out=outputs/maps_p1500b_osg \
    ycb.obstacle_map_union=true \
    eval.behaviour_log=true eval.save_viz=false \
    "output_dir=outputs/prior1500b/gapfill_${scene}_${sid}" \
    > "outputs/prior1500b/gapfill_${scene}_${sid}.log" 2>&1
  log "    exit=$? snapshots now: $(ls outputs/maps_p1500b_ascent/${scene}__*.json 2>/dev/null | wc -l)"
}

fill 00878-XB4GS9ShBRE 50001 "the only episode that reaches the 2.80 m storey"
fill 00808-y9hTuugGdiq 50003 "reaches the 3.26 m storey"

log "stopping the perception servers (pass 2 needs the whole card)"
bash scripts/serve_perception.sh --stop >/dev/null 2>&1
sleep 20

log "coverage audit"
python scripts/audit_map_coverage.py \
  --maps outputs/maps_p1500b_ascent --graphs outputs/maps_p1500b_osg \
  --layout-root /habitat-data-collector/outputs/dualmap_multifloor \
  --layout-root outputs/collector_layouts \
  --scenes 00800-TEEsavR23oF,00808-y9hTuugGdiq,00821-eF36g7L6Z9M,00873-bxsVRursffK,00878-XB4GS9ShBRE \
  --out outputs/audit/map_coverage_1500c.json --label 1500c-gapfill
log "audit exit=$?"

python - <<'PY'
import json, sys
r = json.load(open("outputs/audit/map_coverage_1500c.json"))
bad = []
# `scenes` is a LIST of scene records, not a name->record mapping. Getting this
# wrong made the gate raise AttributeError and exit 1, which the `-eq 3` test
# read as "not severe" -- a broken gate that silently allowed the run through.
for sc in r.get("scenes") or []:
    name = sc.get("scene")
    for st in sc.get("storeys") or []:
        if float(st.get("free_frac") or 0.0) <= 0.0 and int(st.get("n_targets") or 0) > 0:
            bad.append(f"{name} storey y={st.get('height')} holds "
                       f"{st['n_targets']} target(s) and has NO map")
if bad:
    print("SEVERE:", *bad, sep="\n  ")
    sys.exit(3)
print("no unmapped goal storey")
PY
if [ $? -eq 3 ]; then log "SEVERE gate tripped -- pass 2 NOT started"; exit 1; fi

log "starting pass 2"
OBS=outputs/maps_p1500b_ascent OSG=outputs/maps_p1500b_osg \
  OUT=outputs/mf5_pass2_reauth MAX_PARALLEL=3 \
  bash scripts/run_mf5_pass2.sh 00800-TEEsavR23oF 00808-y9hTuugGdiq \
    00821-eF36g7L6Z9M 00873-bxsVRursffK 00878-XB4GS9ShBRE
log "chain done"
