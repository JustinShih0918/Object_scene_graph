#!/usr/bin/env bash
# The three multi-storey scenes, baseline against one fused arm.
#
# 00808, 00814 and 00800 are the authored-15 scenes whose prior maps have two
# storeys and the best target coverage (9, 8 and 7 of the benchmark's objects
# respectively). Their maps under $MAPS are reused: the static manifest's cache
# key does not depend on the relocation source, so the instrument fix does not
# invalidate them and no map pass is needed.
#
#   ARM=v5 scripts/run_fusion_3scene.sh
#   SCENES="00814-p53SfW6mjZe 00800-TEEsavR23oF" scripts/run_fusion_3scene.sh
#
# MAX_PARALLEL is sized to free VRAM, not to CPU: a multi-storey process
# measures 1.6-1.9 GB and this card is shared.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
export PYTHONPATH="$(pwd)/src"

: "${SCENES:=00808-y9hTuugGdiq 00814-p53SfW6mjZe 00800-TEEsavR23oF}"
: "${ARM:=v5}"
: "${ARMS:=base $ARM}"
: "${MAPS:=outputs/maps_15}"
: "${OUT:=outputs/fusion_3scene}"
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
    "ycb.scenes=[$scene]" 'ycb.layout_types=[in_anchor,cross_anchor]' \
    "ycb.map_in=$MAPS/$scene" \
    "output_dir=$d" "+run_tag=F3_${arm}" \
    > "$d/run.log" 2>&1 \
    && { touch "$d/.complete"; echo "[done] $arm/$scene $(date +%H:%M:%S)"; } \
    || echo "[FAILED] $arm/$scene $(date +%H:%M:%S) -- see $d/run.log"
}

for scene in $SCENES; do
  for arm in $ARMS; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 30; done
    wait_for_vram
    run_one "$scene" "$arm" &
    sleep 45   # stagger so two processes never allocate at the same moment
  done
done
wait
echo "[all done] $(date +%H:%M:%S)"
