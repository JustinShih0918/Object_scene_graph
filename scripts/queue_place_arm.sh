#!/usr/bin/env bash
set -u
cd /workspace
set -a; . ./.env; set +a
until grep -q "X2 DONE" outputs/room_campaign/campaign.log 2>/dev/null; do sleep 60; done
for scene in 00848-ziup5kvtCCR 00829-QaLdnwvtxbs 00880-Nfvxx8J5NCo; do
  echo "[$(date -Is)] START X4 $scene" >> outputs/room_campaign/campaign.log
  python scripts/run_eval.py +experiment=ycb_dynamic_room_place \
    ycb.layout_root=outputs/substituted_layouts \
    ycb.scenes=[$scene] "ycb.targets=[bowl,tin can,cracker box,red plate,blue plastic pitcher,bleach bottle]" \
    ycb.layout_types=[in_anchor,cross_anchor] ycb.layout_indices=[1,2,3] \
    ycb.map_in=outputs/maps_v5/$scene \
    llm.text_model=nvidia/nemotron-3.5-lightning-30b-a3b \
    eval.debug_frames=false eval.save_viz=false +run_tag=X4 \
    > outputs/room_campaign/X4_$scene.log 2>&1
  echo "[$(date -Is)] END X4 $scene exit=$?" >> outputs/room_campaign/campaign.log
done
echo "[$(date -Is)] X4 DONE" >> outputs/room_campaign/campaign.log
