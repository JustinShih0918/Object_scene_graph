# Navigating the presence pipeline on ASCENT's obstacle map

## What this is for

The cross-anchor benchmark asks an agent to re-find an object that moved while
it was not looking. Until now one pass built everything: an agent walked the
static scene and left a snapshot holding both the scene graph (object tracks,
presence beliefs, rooms) and the per-storey costmap, and pass 2 navigated from
that single file.

This splits the two. The occupancy now comes from **ASCENT's navigation**
(`src/navigation/`), which explores better than OSG's on this repo's own
measurements, while the scene graph still comes from the OSG pass. Pass 2 then
plans over a map it did not build.

| | builder | artifact | carries |
|---|---|---|---|
| 1a | `ascentnav` | `ycb.obstacle_map_out` | per-storey `ObstacleMap`: obstacle, explored and stair masks, the episode anchor |
| 1b | `nav_agent` | `ycb.map_out` | object tracks, presence beliefs, storeys, connectivity |
| 2 | `nav_agent` | reads `ycb.obstacle_map_in` + `ycb.map_in` | navigates the moved layout |

`ycb.map_in_occupancy=false` is what makes 2 meaningful: the scene-graph
snapshot supplies the objects and the storeys and **none** of its occupancy, so
every cell the planner reads comes from ASCENT's map.

```bash
bash scripts/run_mf5_mapping.sh 00800-TEEsavR23oF     # 1a then 1b, one scene
python scripts/check_obstacle_map_reuse.py \
    --maps outputs/maps_mf5_ascent --run outputs/mf5_stage1/00800-TEEsavR23oF_ascent \
    --png /tmp/map.png                                # look at the picture
python scripts/run_eval.py +experiment=mf5_osg_on_ascent_map \
    'ycb.scenes=[00800-TEEsavR23oF]' \
    ycb.obstacle_map_in=outputs/maps_mf5_ascent \
    ycb.map_in=outputs/maps_mf5_osg ycb.map_in_occupancy=false
```

## The frame, which is the whole risk

`navigation/mapping/map_store.py` stores the stack; `apply_to_costmap` puts it
into an OSG `Costmap2D`. That is a **resample, not a copy**, because three
things separate the lattices: `BaseMap._xy_to_px` swaps the axes and flips the
row, ASCENT's frame is `(x, -z)` where OSG's `PLANE` is `(x, z)`, and the map is
rotated to the episode's start facing. Copying with a shifted origin -- all
`_CostmapView` does, and it is only ever drawn -- puts the walls **10 m out at
heading 0 and 22 m at pi/2** (measured).

What pins it: `tests/unit/test_navigation_map_store.py` checks the loader's
inverse against the obstacle map's own forward projection at four start
headings, and the `--png` is how a human confirms it on a real pass. On 00800,
bowl's map contains 81.6% of bowl's own path as free space and the trajectory
runs down its corridors.

## Three findings that cost time, recorded so they do not again

**`traj_on_map` is a coverage diagnostic, not a validity test.** It measures
whether the agent stood on its own *explored* area, and `explored_area` is what
the agent SAW: `obstacle_map.py:374` erases every cell within an agent-radius
dilation of an obstacle on every step, and `reveal_fog_of_war` propagates only
through navigable cells, so a stairwell interior is never marked at all. One
episode each on 00800, all three maps correctly framed: bowl 0.54/0.77, banana
0.00/0.18, pitcher **0.11 with zero climb steps**. A guard at 0.80 rejects every
map there is. It is recorded and never gated on.

**`start_on_prior_floor` is INERT on a static layout, and blaming it wasted a
run.** `ycb_env.py:617-620` sets `source_layout = None` when the layout IS the
static one, so `prior_floor_y` is never set and the flag has nothing to require.
Composing it True and re-running produced byte-identical starts (3.16 x4, 0.16,
3.16), the same 213 tracks and the same zero target tracks. The observation that
prompted it is real -- five of six 00800 episodes begin at y=3.16 while their
targets sit at y=0.6-1.2, none descend (`traj_y_range` <= 1.05 m), the pass
scores 0/6 having never had a target in view -- but the cause is that a static
pass samples starts anywhere at least `start_min_geodesic_m` from the target,
and on a multi-storey house that routinely lands on another storey. The flag is
for DYNAMIC layouts, where the static layout genuinely is the source.

**Retention ranks cells first, storeys only as a tie-break.** Ranking on storeys
first repeats, one layer up, the mistake of ranking on `len(_floors)` -- a
storey is allocated the moment a staircase is *detected*. Measured: on 00821 an
episode that ran 215 steps over a 3.4 x 1.2 m box while touching four storeys
beat episodes covering 10 x 28 m, and the stored map came out at 2975 cells
against 30-36k for its neighbours; 00878 showed the same signature at 5 storeys
x ~4500 cells.

## What the gate should be

Not SR, and not storey count. **Does the snapshot contain target tracks.**

A single-storey scene graph is fine: the agent starts on the object's prior
floor, the map insists the object is there, it is not, and the presence filter
and search posterior have to re-find it -- possibly upstairs. What the
experiment cannot survive is a map with no target in it, because then there is
nothing to be stale about.

Pass 1a's SR is not a number worth reading either: `ascentnav` drives on D-FINE,
which is closed-set COCO and cannot see a cracker box or a pitcher. It explores
its full budget regardless, and the map is what it is there to produce.

## Resolved: the scene-graph pass was out of budget

Measured on 00800, static layout, same arm otherwise:

| run | storeys | tracks | YCB target tracks | SR |
|---|---|---|---|---|
| baseline, `max_steps=500` | 1 | 213 | **none** | 0/6 |
| `max_steps=1500` | **2** | 527 | **bowl 3, banana 1, cracker box 1, red plate 1** | 1/6 |
| `obstacle_map_in`, 500 steps | 1 | 217 | none | 0/6 |

At 1500 steps the agent reaches the second storey and `bowl` succeeds on step
1104 -- past the old cap. So the pass was never failing to *detect* anything;
it was failing to *arrive*. Two earlier conclusions drawn from the 500-step
runs were wrong and are recorded here because they were stated confidently:
that `climb_attempts=0` meant "no stairs, therefore no portals, therefore no
climb regardless of budget", and that reusing the obstacle map would be the
lever that moved. Budget was the lever. `n_stair_tracks` is still 0-1 even in
the runs that change floors, so whatever carries the agent upstairs is not the
stair-track path, and that is worth understanding before anything is built on
it.

**Reuse is capped at the starting storey on a fresh agent**, which is why the
third row does nothing. `load_obstacle_map` pastes only into floor layers that
already exist, and an agent at episode start has exactly one, so 17002 of
29950 cells arrived and the upper storey -- the one the pass needed -- stayed
unknown (`unmatched_floors: 1`, visible in `prior_obstacle_map`). The reuse
machinery is working; its reach is bounded. Pass 2 loads a multi-storey
`map_in` first and therefore has the layers, but a mapping pass does not.
Lifting the cap means creating layers from the stored storeys, and an
`ObstacleMap` records no world height to place them at -- which is the same
gap the floor matching already documents.

## Measured: pass 2 runs, and the climb machinery is unreachable

`outputs/mf5_pass2_00800` (v1) and `outputs/mf5_pass2_v2` (v2, with the stair
height bridge) on 00800, `cross_anchor_01`, 500 steps, occupancy from ASCENT
only (`ycb.map_in_occupancy=false`):

| | toy airplane | cracker box |
|---|---|---|
| steps used / budget | 180 / 500 | 362 / 500 |
| distance to goal | 23.2 m | 16.2 m |
| start storey -> goal storey | 0 -> 1 | 1 -> 0 |
| `goal_floor_reached` | False | False |
| `climb_attempts` | 0 | 0 |
| attempts used | 3 of 3 | 3 of 3 |
| states entered | approach, done | approach, close_look, done |

The occupancy reuse itself works: 43139 cells written across BOTH storeys,
`unmatched_floors: 0`, and the presence filter runs (120 and 140 events).

**v2 is trajectory-identical to v1.** Same step counts, same distances, same
flags. The height ramp did write heights -- `stair_max_rise_m` appears where v1
had no stair stats at all -- but it changed no decision, because the code that
would read it is never reached.

Why it is never reached, in order:

* `flights_seen` and `portals_seen` are ABSENT from `agent_stats`. Both are
  written unconditionally (`max(get(...), len(...))`) once `try_switch` gets
  past its gate, so an absent key means the function returned early or was
  never called.
* `_try_floor_switch` is called from exactly one place: inside `_explore`
  (`nav_agent.py:1013`), as the `floor_switch` callback handed to the
  exploration strategy.
* The exploration ROUND never runs. `frontier_select_log` and
  `search_log_events` are both EMPTY across 180 and 362 steps.

The state log looked at first like the agent never enters `explore` at all --
it reads `approach -> done` three times. That reading is WRONG and the
correction is the finding. `rearm` does set `state = State.EXPLORE`, and
`state_log` misses it for a mundane reason: `act` captures `prev_state` at the
top of the step, `rearm` runs between steps, so the APPROACH -> EXPLORE
transition happens outside `act` and is never appended.

The agent does go back to EXPLORE. What it does not get is a ROUND. Two things
stand between the two, and they compound:

* `_act_inner` runs `candidates.check` BEFORE the EXPLORE branch, in EXPLORE
  and in GOTO_FRONTIER both. A commit there returns to APPROACH in the same
  step.
* `ExplorationStrategy.select` is rate-limited to one run per
  `exploration.select_every` steps (5 here).

Attempt 1 fails at step 70; the next commit lands at step 74. Those four
EXPLORE steps are all inside the rate limit, so `select` never fires, and
`floor_switch` -- the only call site of `_try_floor_switch` -- is never
invoked. Commits at 8, 74, 156 (and 3, 68, 123), episodes over at 180 and 362
of 500 steps.

So the whole multi-floor path -- `find_flights`, `pursuit_flight`,
`climb_flight_carrot`, and the stair bridge that feeds them -- is unreachable
in this configuration. **The bridge is still unproven, not disproven.**

What consumes the attempts is the candidate channel, not the map. `p=0.8176`
on every commit and `seen_live: true` on five of six: these are live tracks the
open-vocab detector created on the starting storey under the target's own
label, with the appearance channel declining to admit them
(`feature_admitted: false`, `feature_sim: -1.0`) and `target_bypassed` at 11
and 4. Each attempt is spent walking to one, failing, and committing to the
next one on the same floor. `eval.attempts: 3` then ends the episode.

Both episodes require a storey change to score. The agent does not spend its
budget failing to climb; it never asks to.

### The fix: `agent.explore_after_failed_attempt_steps`

Default 0, which is the shipped behaviour and leaves every measured arm
untouched; `mf5_osg_on_ascent_map` sets 25. After a failed attempt `rearm`
holds `candidates.check` open for that many steps AND calls
`ExplorationStrategy.force_select_next`, so the round runs immediately instead
of waiting on the rate limit. Both halves are needed: holding the commit
without dropping the rate limit buys silence, and dropping the rate limit
without holding the commit buys a round that a commit pre-empts in the same
step.

It is a hold, not a ban. Nothing is blacklisted, and candidates resume at the
end of the window -- `explore_hold_steps` counts what it cost.

The alternative considered and NOT taken was tightening
`scene_graph.target_bypasses_gates`, which is what lets a target-labelled
detection become a track without clearing the admission gates
(`target_bypassed` 11 and 4 here, `feature_admitted: false` on every commit).
That would change what the map contains, which is a bigger change and one that
every other arm is measured with.

## What to open

The pipeline runs across two packages and three passes, and nothing about it is
discoverable from one file. In dependency order:

### 1a -- ASCENT builds the occupancy

| file | what to look for |
|---|---|
| `src/navigation/mapping/obstacle_map.py` | ASCENT's own map, vendored. `_map`, `explored_area`, the four stair masks. Line 374 erases an agent-radius band each step, which is why `explored_area` is not a coverage measure |
| `src/navigation/mapping/map_store.py:138` | `save_obstacle_maps` -- the stack plus the episode anchor `{xy, heading}`, JSON with a sibling `.npz` |
| `src/osg/eval/prior_map.py:28` | `save_obstacle_map_for_scene` -- the RETENTION rule: rank on explored cells, storeys only as a tie-break |
| `configs/experiment/mf5_ascentnav_map.yaml` | the arm that produces it |

### 2 -- pasting it into an OSG costmap

This is the part with the real risk in it, and the part to read first if
anything looks geometrically wrong.

| file | what to look for |
|---|---|
| `src/navigation/mapping/map_store.py:293` | `apply_to_costmap` -- a RESAMPLE, not a copy. Three transforms separate the lattices; a naive copy puts the walls 10 m out at heading 0 and 22 m at pi/2 |
| `src/navigation/mapping/map_store.py:438` | `_ramp_stair_heights` -- the ASCENT->OSG multi-floor bridge. An `ObstacleMap` stores no heights, and `find_flights` reads `costmap.height`, so without this the climb machinery sees nothing. STILL UNPROVEN -- see above |
| `src/osg/eval/prior_map.py:76` | `load_obstacle_map` -- floors matched bottom-up by height; capped at layers that already exist |
| `tests/unit/test_navigation_map_store.py` | the inverse checked against the forward projection at four start headings |
| `scripts/check_obstacle_map_reuse.py --png` | how a human confirms it on a real pass |

### 1b -- the scene graph, and building both in one pass

| file | what to look for |
|---|---|
| `src/osg/graph/map_store.py:182` | `save_map` -- the schema-v2 snapshot: tracks, presence beliefs, storeys |
| `src/osg/graph/map_store.py:360` | `restore_grid` -- returns immediately when `restore_occupancy=False`, which is what lets ASCENT's map be the SOLE occupancy |
| `src/osg/agent/world_model.py` | OSG's world model detached from OSG's FSM, so `ascentnav` can drive while the scene graph is built alongside. `observe` is the mapping half of `NavAgent._act_inner` |
| `configs/experiment/mf5_mapping_both.yaml` | one pass, both artifacts. BUILT AND UNIT-TESTED, NEVER YET RUN ON A SCENE |

### 3 -- pass 2, and where it currently stops

| file | what to look for |
|---|---|
| `configs/experiment/mf5_osg_on_ascent_map.yaml` | the arm; `ycb.map_in_occupancy=false` is what makes the experiment mean anything |
| `src/osg/eval/episode.py:44` | `run_episode` -- `load_prior_map` then `load_obstacle_map`, in that order and for a reason: the snapshot creates the storey layers the paste needs |
| `src/osg/agent/nav_agent.py:516` | `rearm` -- and `explore_after_failed_attempt_steps`, the hold |
| `src/osg/agent/nav_agent.py:1144` | `_try_floor_switch` -- reachable ONLY from `_explore` (line 1032) |
| `src/osg/agent/floor_policy.py:382` | `try_switch` -- the gate that decides whether to leave a storey at all |
| `src/osg/mapping/stairs.py:172` | `find_flights` -- returns `[]` when `costmap.height is None`, which is what `_ramp_stair_heights` exists to prevent |
| `src/osg/agent/nav_agent.py:1878` | `_flight_carrot` -- the waypoint-at-a-time climb, as opposed to the navmesh pointing at the top |

### What to read in a finished run

`episodes.jsonl`, per episode:

* `prior_obstacle_map` -- `cells_written`, `matched_floors`, `unmatched_floors`.
  A non-zero `unmatched_floors` means storeys were stored that this run had no
  layer for. `traj_on_map` is a DIAGNOSTIC and must never be gated on.
* `goal_floor_reached`, `climb_attempts`, `floor_switches` -- whether the storey
  change the episode needs ever happened.
* `agent_stats`: `flights_seen` and `portals_seen` ABSENT means `try_switch`
  returned before looking; `explore_hold_steps` is what the hold cost.
* `frontier_select_log` and `search_log_events` EMPTY means no exploration
  round ran at all -- the signature of the failure above.
* `goal_commit_log` -- `prior: true` is a snapshot object, `seen_live: true` a
  track made this episode. All-live commits mean the run is chasing its own
  detector, not the map.

Videos land in `viz/debug/<scene>_ep<ID>.mp4` and are re-encoded to H.264 on
close (`eval.debug_video_crf`, default 30; 0 opts out).

## Open

**Superseded by the table above; kept for the reasoning that was eliminated.**
The OSG scene-graph pass reaches no objects at 500 steps.
On 00800 it ends with 213 furniture tracks and zero target tracks, in three
independent runs: as configured, with `start_on_prior_floor` flipped, and with
`agent.rednet_stairs=true`. All three are byte-identical, which is itself the
finding -- none of those levers touches the cause. `climb_attempts` is 0 in
every episode and `n_stair_tracks` is 0 or 1, so the agent never approaches a
staircase, and a start sampled on the wrong storey is therefore terminal.

Three levers remain, and they are not equivalent:

* **`agent.max_steps` above 500.** The agent ends 9-23 m from its target in a
  large two-storey house; it may simply run out of budget. Cheapest to test,
  changes the arm the least.
* **Let the mapping pass read `ycb.obstacle_map_in`.** ASCENT already mapped
  2-4 storeys of every scene, so the OSG pass would start with the building
  known and spend its whole budget on objects rather than geometry. This uses
  the machinery this document describes and is the most interesting option, but
  it makes the scene-graph pass depend on the obstacle-map pass.
* **Make the OSG agent climb.** The honest fix, and the largest.

Until one is chosen the cross-anchor experiment cannot run, because a map with
no target track in it has nothing to be stale about.

Each is one command, and the gate is the same for all three -- does the
snapshot come back with YCB target tracks in it:

```bash
# lever 1: more budget
python scripts/run_eval.py +experiment=mf5_osg_unified \
    'ycb.scenes=[00800-TEEsavR23oF]' 'ycb.layout_types=[static]' \
    'ycb.layout_indices=[1]' ycb.cross_floor_relocations_only=false \
    agent.max_steps=1500 ycb.map_out=outputs/maps_try_steps \
    eval.save_viz=false output_dir=outputs/try_steps

# lever 2: map the objects over ASCENT's geometry
python scripts/run_eval.py +experiment=mf5_osg_unified \
    'ycb.scenes=[00800-TEEsavR23oF]' 'ycb.layout_types=[static]' \
    'ycb.layout_indices=[1]' ycb.cross_floor_relocations_only=false \
    ycb.obstacle_map_in=outputs/maps_mf5_ascent \
    ycb.map_out=outputs/maps_try_obstacle \
    eval.save_viz=false output_dir=outputs/try_obstacle

# the gate, for either
python - <<'EOF'
import json
from collections import Counter
from pathlib import Path
YCB = {"bowl","tin can","cracker box","banana","red plate","coffee can",
       "blue plastic pitcher","toy airplane","scissors","mug","yellow bottle"}
for d in ("outputs/maps_try_steps", "outputs/maps_try_obstacle"):
    for p in Path(d).glob("*.json"):
        b = json.loads(p.read_text())
        labs = Counter(t["label"] for t in b["tracks"])
        print(d, {k: v for k, v in labs.items() if k in YCB} or "NO TARGET TRACKS")
EOF
```

Lever 2 is the one worth running first: it costs nothing extra (the maps are
already built), and if the OSG pass reaches the objects once the geometry is
handed to it, that is also evidence the obstacle-map reuse does what it was
built to do.
