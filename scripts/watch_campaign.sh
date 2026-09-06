#!/usr/bin/env bash
# Emit one line per finished scene, carrying the RESULT rather than the event.
set -u
cd /workspace
seen=""
while true; do
  while read -r ts tag scene rest; do
    key="$tag/$scene"
    case " $seen " in *" $key "*) continue;; esac
    seen="$seen $key"
    python scripts/report_scene.py "$tag" "$scene" 2>&1 | head -3
  done < <(grep -oE "END [A-Z0-9]+ [0-9]{5}-[A-Za-z0-9]+" outputs/room_campaign/campaign.log \
             | awk '{print "x", $2, $3}')
  if grep -q "Z DONE" outputs/room_campaign/campaign.log 2>/dev/null; then
    echo "CAMPAIGN COMPLETE: Z finished all scenes"; exit 0
  fi
  if ! pgrep -f "queue_z.sh" >/dev/null && ! pgrep -f "run_tag=Z" >/dev/null; then
    echo "WARNING: Z driver and run both gone before 'Z DONE' — check outputs/room_campaign/"; exit 1
  fi
  sleep 60
done
