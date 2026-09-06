#!/usr/bin/env bash
# Conditions Y and X on the full 102-episode dynamic set, one flag apart.
#
#   Y = ycb_dynamic_cross  (Q: the search anchor dropped once the agent has been
#                           there) -- the control. Q has never been run on this
#                           102-episode set; the recorded Q/Q2 numbers are the
#                           96-episode set, before 00880 grew by six.
#   X = ycb_dynamic_room   (Y + exploration.room_posterior_llm)
#
# The control runs FIRST: if the night is cut short, a treatment with no control
# on the same episodes is worth nothing, and the existing V/W runs differ from
# both by more than one flag.
set -u

cd /workspace
set -a; . ./.env; set +a

SCENES=(00829-QaLdnwvtxbs 00848-ziup5kvtCCR 00880-Nfvxx8J5NCo)
TARGETS='[bowl,tin can,cracker box,red plate,blue plastic pitcher,bleach bottle]'
LOGDIR=/workspace/outputs/room_campaign
mkdir -p "$LOGDIR"

run_one () {
  local preset="$1" tag="$2" scene="$3"
  local log="$LOGDIR/${tag}_${scene}.log"
  echo "[$(date -Is)] START $tag $scene" | tee -a "$LOGDIR/campaign.log"
  python scripts/run_eval.py \
    +experiment="$preset" \
    ycb.layout_root=outputs/substituted_layouts \
    ycb.scenes="[$scene]" "ycb.targets=$TARGETS" \
    ycb.layout_types=[in_anchor,cross_anchor] ycb.layout_indices=[1,2,3] \
    ycb.map_in="outputs/maps_v5/$scene" \
    llm.text_model=nvidia/nemotron-3.5-lightning-30b-a3b \
    eval.debug_frames=false eval.save_viz=false \
    +run_tag="$tag" > "$log" 2>&1
  echo "[$(date -Is)] END   $tag $scene exit=$?" | tee -a "$LOGDIR/campaign.log"
}

for scene in "${SCENES[@]}"; do run_one ycb_dynamic_cross Y "$scene"; done
for scene in "${SCENES[@]}"; do run_one ycb_dynamic_room  X "$scene"; done

echo "[$(date -Is)] CAMPAIGN DONE" | tee -a "$LOGDIR/campaign.log"
