#!/usr/bin/env bash
# The cross-floor half of the authored benchmark, and only that.
#
# Counted from the authored layouts, the whole 15-scene set holds 19 relocations
# that actually change storey, all of them in cross_anchor at layout index 1:
# 00808 4, 00821 3, 00878 3, 00800 2, 00869 2, 00873 2, 00810 1, 00824 1,
# 00871 1, and none in the other six scenes. An in_anchor move is never
# cross-floor, by construction.
#
# So running the full 14-episode suite per scene spends nearly all of its budget
# on same-floor episodes that cannot inform a cross-floor question.
# `cross_floor_relocations_only` keeps only the moves that change storey, and
# `start_on_prior_floor` (already set by the combined preset) starts the agent
# where the object USED to be -- which is the scenario itself: the prior map is
# right about the room and wrong about the floor.
#
# Restricted by default to the scenes that have a prior map under $MAPS.
#
#   scripts/run_crossfloor_ab.sh
#   ARMS="base v5" SCENES="00808-y9hTuugGdiq" scripts/run_crossfloor_ab.sh
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export PYTHONPATH="$(pwd)/src"

: "${SCENES:=00808-y9hTuugGdiq 00821-eF36g7L6Z9M 00800-TEEsavR23oF 00810-CrMo8WxCyVb}"
: "${ARMS:=base v5}"
: "${MAPS:=/workspace/outputs/maps_15}"
: "${OUT:=outputs/crossfloor_ab}"
: "${MIN_FREE_MIB:=2600}"
: "${MAX_PARALLEL:=2}"

preset_for() {
  case "$1" in
    base)       echo ycb_authored_15 ;;
    containers) echo ycb_authored_15_containers ;;
    fused)      echo ycb_authored_15_fused ;;
    v3)         echo ycb_authored_15_fused_v3 ;;
    v4)         echo ycb_authored_15_fused_v4 ;;
    v5)         echo ycb_authored_15_fused_v5 ;;
    v6)         echo ycb_authored_15_fused_v6 ;;
    *)          echo "$1" ;;
  esac
}

wait_for_vram() {
  for _ in $(seq 1 480); do
    local used total
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    [ $(( total - used )) -ge "$MIN_FREE_MIB" ] && return 0
    sleep 60
  done
}

run_one() {
  local scene=$1 arm=$2
  local d="$OUT/$arm/$scene"
  [ -f "$d/.complete" ] && { echo "[skip] $arm/$scene"; return 0; }
  mkdir -p "$d"
  echo "[start] $arm/$scene $(date +%H:%M:%S)"
  python3 scripts/run_eval.py "+experiment=$(preset_for "$arm")" \
    "ycb.scenes=[$scene]" 'ycb.layout_types=[cross_anchor]' \
    ycb.cross_floor_relocations_only=true \
    "ycb.map_in=$MAPS/$scene" \
    "output_dir=$d" "+run_tag=XF_${arm}" \
    > "$d/run.log" 2>&1 \
    && { touch "$d/.complete"; echo "[done] $arm/$scene $(date +%H:%M:%S)"; } \
    || echo "[FAILED] $arm/$scene $(date +%H:%M:%S) -- see $d/run.log"
}

for scene in $SCENES; do
  for arm in $ARMS; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 30; done
    wait_for_vram
    run_one "$scene" "$arm" &
    sleep 40
  done
done
wait
echo "[all done] $(date +%H:%M:%S)"
