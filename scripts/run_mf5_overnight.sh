#!/usr/bin/env bash
# Everything still owed for stage 1, in ONE process, in order.
#
#   1. the corrected OSG scene-graph pass on 00800 (starts on the object's floor)
#   2. re-pick all five ASCENT obstacle maps under the fixed retention rule
#
# ONE process on purpose. The previous version split this across scripts that
# waited on each other with `while kill -0 $PID`, and `kill -0` SUCCEEDS on a
# zombie: stage 1a's shell was reaped into `[bash] <defunct>`, so the waiter
# spun forever. `until ! pgrep -f "run_eval.py..."` has the same class of bug
# from the other side -- `pgrep -f` matches the waiting shell's OWN command
# line, so it never exits. Sequential statements in one script cannot do either.
#
# WHY ZOMBIES PERSIST HERE, because it makes the first bug permanent rather
# than transient: PID 1 in this container is `conda run --no-capture-output -n
# habitat bash`, which is not an init and never reaps orphans. 87 of them had
# accumulated by 02:00, all parented to PID 1. They cost nothing (124 processes
# against a 4.19M pid_max) but they never go away, so a PID that exited hours
# ago still answers `kill -0`. Never wait on a PID in this container; wait on a
# string in a log, or do not split the work across processes at all.
set -u
cd "$(dirname "$0")/.."
set -a; [ -f docker/.env ] && . ./docker/.env; set +a

OBS=${OBS:-outputs/maps_mf5_ascent}
OSG=${OSG:-outputs/maps_mf5_osg}
OUT=${OUT:-outputs/mf5_stage1}
SCENES=(00800-TEEsavR23oF 00808-y9hTuugGdiq 00821-eF36g7L6Z9M
        00873-bxsVRursffK 00878-XB4GS9ShBRE)
mkdir -p "$OBS" "$OSG" "$OUT"
log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------- 1. stage 1b
log "=== corrected OSG scene-graph pass on 00800 ==="
rm -rf "$OSG/00800-TEEsavR23oF.json" "$OSG/00800-TEEsavR23oF.npz" \
       "$OUT/00800-TEEsavR23oF_osg"
# `start_on_prior_floor` is not set either way, because on a STATIC layout it
# is INERT: `ycb_env.py:617-620` passes `source_layout = None` when the layout
# is the static one, so `prior_floor_y` is never populated and the flag has
# nothing to require. Composing it True changed nothing -- identical starts
# (3.16 x4, 0.16, 3.16), identical 213 tracks, identical zero target tracks.
# The real cause of those starts is that a static pass samples anywhere at
# least `start_min_geodesic_m` from the target, which on a multi-storey house
# lands on another floor; the flag is for DYNAMIC layouts, where the static
# layout genuinely is the relocation source.
python scripts/run_eval.py +experiment=mf5_osg_unified \
  'ycb.scenes=[00800-TEEsavR23oF]' 'ycb.layout_types=[static]' \
  'ycb.layout_indices=[1]' ycb.cross_floor_relocations_only=false \
  "ycb.map_out=$OSG" eval.save_viz=false \
  "output_dir=$OUT/00800-TEEsavR23oF_osg" \
  > "$OUT/00800-TEEsavR23oF_osg.log" 2>&1
log "    exit=$? $(grep -o '"success_rate": [0-9.]*' "$OUT/00800-TEEsavR23oF_osg.log" | tail -1)"

python - "$OSG/00800-TEEsavR23oF.json" "$OUT/00800-TEEsavR23oF_osg/episodes.jsonl" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path
snap, eps = Path(sys.argv[1]), Path(sys.argv[2])
if not snap.exists():
    print("    NO SNAPSHOT WRITTEN"); raise SystemExit
b = json.loads(snap.read_text())
labs = Counter(t["label"] for t in b["tracks"])
YCB = {"bowl", "tin can", "cracker box", "banana", "red plate", "coffee can",
       "blue plastic pitcher", "toy airplane", "scissors", "mug", "yellow bottle"}
hit = {k: v for k, v in labs.items() if k in YCB}
print(f"    storeys={[round(f['height_y'], 2) for f in b['floors']]} tracks={len(b['tracks'])}")
print(f"    YCB TARGET TRACKS: {hit or 'NONE'}   <- the gate that matters")
if eps.exists():
    for e in (json.loads(l) for l in eps.read_text().splitlines() if l.strip()):
        print(f"      {e['target']:<22} succ={e['success']:.0f} "
              f"start_y={e.get('start_y', 0):.2f} floors={e.get('n_floors_seen')} "
              f"in_view={e.get('gt_kf_in_view')} d2g={e.get('distance_to_goal', 0):.1f}")
PY

# ------------------------------------------------------------- 2. re-pick maps
log "=== re-picking all five obstacle maps under the fixed retention rule ==="
KEEP="outputs/maps_mf5_ascent_oldrule_$(date +%H%M)"
mkdir -p "$KEEP"; cp -a "$OBS"/. "$KEEP"/ 2>/dev/null
log "    previous maps kept in $KEEP"
for scene in "${SCENES[@]}"; do
  rm -f "$OBS/$scene.json" "$OBS/$scene.npz"
  rm -rf "$OUT/${scene}_ascent"
  log "--- $scene"
  python scripts/run_eval.py +experiment=mf5_ascentnav_map \
    "ycb.scenes=[$scene]" "ycb.obstacle_map_out=$OBS" \
    eval.behaviour_log=true eval.save_viz=false \
    "output_dir=$OUT/${scene}_ascent" > "$OUT/${scene}_ascent.log" 2>&1
  python - "$scene" "$OBS" "$KEEP" <<'PY'
import json, sys
from pathlib import Path
scene, new, old = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
def tot(root):
    p = root / f"{scene}.json"
    if not p.exists():
        return None
    b = json.loads(p.read_text())
    fl = [f for f in b["floors"] if f["explored_cells"] > 0]
    return f"{len(fl)} storeys / {sum(f['explored_cells'] for f in fl)} cells"
print(f"    {scene}: was [{tot(old)}]  ->  now [{tot(new)}]")
PY
done

log "=== all done ==="
