#!/usr/bin/env bash
# Finish ASCENT's DualMap run, surviving perception-server deaths.
#
# The first pass lost 79 of 186 trials to servers dying mid-condition: the
# driver only re-checked them at a CONDITION boundary, and its retry pass was
# skipped precisely when it was needed because that pass tests `servers_ok`
# first. Nothing in the logs shows why they die -- no CUDA OOM, host memory at
# 5 of 124 GB -- and the panes are gone, so this does not try to diagnose it.
# It makes the run converge anyway: keep restarting the servers and re-running
# until no trial is pending, which is safe because every layer already resumes.
#
#   bash scripts/run_ascent_dualmap_resume.sh
set -u
cd "$(dirname "$0")/.."
log() { echo "[$(date +%H:%M:%S)] $*"; }
OUT=${OUT:-outputs/ascent_dualmap}
MAX_PARALLEL=${MAX_PARALLEL:-2}
ROUNDS=${ROUNDS:-12}

servers_ok() {
  local up=0 p c
  for p in 13182 13183 13184 13185 13186; do
    c=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://localhost:$p/" 2>/dev/null)
    [ -n "$c" ] && [ "$c" != "000" ] && up=$((up+1))
  done
  [ "$up" -ge 5 ]
}

ensure_servers() {
  servers_ok && return 0
  log "    servers down -- restarting"
  bash scripts/serve_perception.sh >/dev/null 2>&1
  for _ in $(seq 1 40); do servers_ok && { log "    servers back"; return 0; }; sleep 15; done
  return 1
}

pending_total() {
  python3 - "$OUT" <<'PY'
import glob, json, sys
done = set()
for p in glob.glob(f"{sys.argv[1]}/**/episodes.jsonl", recursive=True):
    for line in open(p, encoding="utf-8"):
        if line.strip():
            done.add(json.loads(line)["episode_id"])
want = json.load(open("data/splits/dualmap_186.json"))["trials"]
print(sum(1 for t in want if t["trial_id"] not in done))
PY
}

for round in $(seq 1 "$ROUNDS"); do
  left=$(pending_total)
  log "=== round ${round}: ${left} trial(s) pending"
  [ "$left" -eq 0 ] && { log "=== all 186 trials done"; break; }
  ensure_servers || { log "servers will not start -- stopping"; exit 1; }
  for cond in cross_anchor static in_anchor; do
    ensure_servers || break
    ARM=ascent PRESET_PREFIX=dualmap_protocol_ \
      SPLIT="data/splits/dualmap_${cond}.json" OUT="$OUT" \
      CONDITIONS="$cond" MAP_ROOT= MAX_PARALLEL="$MAX_PARALLEL" CHUNK=4 \
      bash scripts/run_dualmap_arm.sh >/dev/null 2>&1
    log "    ${cond}: $(pending_total) still pending overall"
  done
done

log "final pending: $(pending_total)"
bash scripts/serve_perception.sh --stop >/dev/null 2>&1
log "=== resume finished"
