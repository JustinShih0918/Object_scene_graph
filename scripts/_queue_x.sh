#!/usr/bin/env bash
# Start X as soon as the Y control finishes. One flag apart from Y.
set -u
cd /workspace
set -a; . ./.env; set +a
until grep -q "Y DONE" outputs/room_campaign/campaign.log 2>/dev/null; do sleep 60; done
for scene in 00829-QaLdnwvtxbs 00848-ziup5kvtCCR 00880-Nfvxx8J5NCo; do
  echo "[$(date -Is)] START X $scene" >> outputs/room_campaign/campaign.log
  python scripts/run_eval.py +experiment=ycb_dynamic_room \
    ycb.layout_root=outputs/substituted_layouts \
    ycb.scenes=[$scene] "ycb.targets=[bowl,tin can,cracker box,red plate,blue plastic pitcher,bleach bottle]" \
    ycb.layout_types=[in_anchor,cross_anchor] ycb.layout_indices=[1,2,3] \
    ycb.map_in=outputs/maps_v5/$scene \
    llm.text_model=nvidia/nemotron-3.5-lightning-30b-a3b \
    eval.debug_frames=false eval.save_viz=false +run_tag=X \
    > outputs/room_campaign/X_$scene.log 2>&1
  echo "[$(date -Is)] END X $scene exit=$?" >> outputs/room_campaign/campaign.log
done
echo "[$(date -Is)] X DONE" >> outputs/room_campaign/campaign.log
