#!/usr/bin/env bash
# The close-look A/B (docs/SR_PROPOSAL_CLOSE_LOOK.md): three arms x three scenes,
# dynamic trials only, paired against the tight-ring baseline already on disk
# at outputs/osg_dualmap_tightring. Each (arm, scene) is one process because the
# prior map is per scene; MAX_PARALLEL bounds how many share the GPU.
#
#   scripts/run_close_look_ab.sh              # all nine, 3 at a time
#   MAX_PARALLEL=4 scripts/run_close_look_ab.sh
#   ARMS="inanchor" scripts/run_close_look_ab.sh   # a subset
#   ARMS="drop len4 flat" OUT_ROOT=outputs/osg_searchorder scripts/run_close_look_ab.sh
#
# An arm named X runs the preset `${PRESET_PREFIX}X` (default prefix
# `dualmap_protocol_osg_look_`), so a new arm is a new preset and nothing else.
# Reads the hosted-VLM credential from .env; outputs land under
# $OUT_ROOT/<arm>/<scene>/, logs beside them. Idempotent: an (arm, scene)
# whose episodes.jsonl already holds the scene's trials is skipped, so a
# crashed batch is resumed by running the script again.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
: "${MAX_PARALLEL:=3}"
: "${ARMS:=inanchor opportunistic both}"
: "${SCENES:=00829-QaLdnwvtxbs 00848-ziup5kvtCCR 00880-Nfvxx8J5NCo}"
: "${TRIALS_PER_SCENE:=35}"
: "${OUT_ROOT:=outputs/osg_closelook}"
: "${PRESET_PREFIX:=dualmap_protocol_osg_look_}"

run_one() {
  local arm=$1 scene=$2
  local out="$OUT_ROOT/$arm/$scene"
  mkdir -p "$out"
  if [ -f "$out/episodes.jsonl" ] && [ "$(wc -l < "$out/episodes.jsonl")" -ge "$TRIALS_PER_SCENE" ]; then
    echo "[skip] $arm $scene already complete"
    return 0
  fi
  echo "[start] $arm $scene $(date +%H:%M:%S)"
  python scripts/run_eval.py "+experiment=${PRESET_PREFIX}${arm}" \
    "dualmap.scenes=[$scene]" \
    'dualmap.conditions=[in_anchor,cross_anchor]' \
    "ycb.map_in=outputs/maps_v5/$scene" \
    "output_dir=$out" \
    "+run_tag=CLOSELOOK_${arm^^}" \
    > "$out.log" 2>&1 \
    && echo "[done] $arm $scene $(date +%H:%M:%S)" \
    || echo "[FAILED] $arm $scene $(date +%H:%M:%S) -- see $out.log"
}

jobs_running() { jobs -rp | wc -l; }

for arm in $ARMS; do
  for scene in $SCENES; do
    while [ "$(jobs_running)" -ge "$MAX_PARALLEL" ]; do sleep 20; done
    run_one "$arm" "$scene" &
    sleep 45  # stagger the scene loads
  done
done
wait
echo "[all done] $(date +%H:%M:%S)"
