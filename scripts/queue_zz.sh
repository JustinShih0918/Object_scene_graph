#!/usr/bin/env bash
set -u
cd /workspace
set -a; . ./.env; set +a
for scene in 00848-ziup5kvtCCR 00880-Nfvxx8J5NCo 00829-QaLdnwvtxbs; do
  echo "[$(date -Is)] START ZZ $scene" >> outputs/room_campaign/campaign.log
  python scripts/run_eval.py +experiment=ycb_dynamic_presgate \
    ycb.layout_root=outputs/substituted_layouts \
    ycb.scenes=[$scene] "ycb.targets=[bowl,tin can,cracker box,red plate,blue plastic pitcher,bleach bottle]" \
    ycb.layout_types=[in_anchor,cross_anchor] ycb.layout_indices=[1,2,3] \
    ycb.map_in=outputs/maps_v5/$scene \
    llm.text_model=nvidia/nemotron-3.5-lightning-30b-a3b \
    eval.debug_frames=false eval.save_viz=false +run_tag=ZZ \
    > outputs/room_campaign/ZZ_$scene.log 2>&1
  echo "[$(date -Is)] END ZZ $scene exit=$?" >> outputs/room_campaign/campaign.log
done
echo "[$(date -Is)] ZZ DONE" >> outputs/room_campaign/campaign.log
