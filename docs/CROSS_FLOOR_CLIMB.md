# Why the agent does not climb

Three questions were open after the fused pipeline came back a null: why 55
switch attempts produce 0.17 m of ascent, whether a remembered staircase can be
reached when the gate refuses, and whether the cross-floor sample can be made
bigger. All three are answered.

## The sample, corrected

Counted from the authored layouts, over all 15 scenes and all three layout
indices:

| vertical move | count |
|---|---:|
| more than 0.5 m | 19 |
| more than 1.8 m, a real storey | 13 |

The first count was too loose: a 0.84 m move is floor to shelf, not floor to
floor. 1.8 m is `floor.new_level_m`, the threshold the estimator itself uses.

Every one is in `cross_anchor` at layout index 1. An in-anchor move is never
cross-floor, by construction. Five prior maps were built for this
(`scripts/build_crossfloor_maps.sh`), taking the runnable set from 7 episodes on
4 scenes to **12 on 6**. Of the remainder, 00824 and 00871 have no storey change
at all, and 00869's single episode cannot be constructed.

## The answer: the climb was delegated to something that was removed

`portals.py` and `docs/MULTI_FLOOR.md` both define a portal as "not the staircase
itself, but a place to walk toward, **after which the navmesh handles the
climb**". Commit 866ab0e took the YCB line sensor-only on 2026-09-08, replacing
habitat's follower with a PointNav policy, and nothing downstream was changed.
The step that was going to do the climbing no longer exists.

A `Portal.centroid_xy` is the mean grid position of cells whose only observed
surface is a storey away. That is a point under or across from the other floor,
not a stair mouth. The per-floor costmap is 2D and cannot represent a ramp, so
the PointNav policy has nothing to follow upward.

Restoring the navmesh tests this directly. On 00821's cracker box, the cleanest
episode in the set:

| arm | switch attempts | final height | climbed | reached the floor |
|---|---:|---:|---:|---|
| base | 55 | -3.47 | 0.17 m | no |
| v8 | 53 | -3.47 | 0.17 m | no |
| navmesh | **1** | **+0.13** | **3.60 m** | **yes** |

One attempt instead of fifty-five, and the whole storey.

## The campaign

Six scenes, 13 paired episodes, 12 of them genuine storey changes.

| | base | v8 | navmesh |
|---|---:|---:|---:|
| success, cross-floor | 0/12 | 0/12 | **2/12** |
| reached the object's floor | 2 | 2 | 2 |
| climbed more than 1 m | 1 | 2 | 3 |
| floor switch attempts | 76 | 60 | 9 |

Four episodes climbed under the navmesh and did not move at all without it:
00878's tin can rose 3.0 m against 0.0, its coffee can 2.8 m against 0.0,
00821's cracker box 3.6 m against 0.17, 00800's cracker box 0.82 m against 0.0.
The navmesh also lost one: 00808's yellow bottle climbed 3.2 m under base and
0.23 m under the navmesh.

**The navmesh arm is a diagnostic and never a headline.** It hands the agent
habitat's ShortestPathFollower planning on the ground-truth mesh and
`is_reachable` as an oracle, neither of which a sensor-only system has. Its two
successes measure the size of the gap, not progress against it.

Climbing is necessary and not sufficient: the navmesh arm reached 00821's upper
floor and still ended 7.98 m from the goal with the target never in view.

## What the two code changes were worth

Both fire and neither converts.

- `floor.prior_stairs_override_gate` (v7) lets a remembered staircase stand in
  for the "nothing near is left here" clause, which a large storey never
  satisfies. It fired twice. Those episodes reached `find_portals` instead of
  returning early, which is what it was for.
- `floor.portal_failure_memory` (v8) stops a portal that led nowhere being
  proposed again. It skipped 61 re-proposals and remembered failures in two
  episodes, taking climbs over a metre from 1 to 2. On 00821 it correctly
  dropped the failed patch and had nothing better to offer, because that scene's
  prior map is single-storey and holds no recorded traversal.

`floor.prefer_prior_stairs` (v6) stays off: it fired 106 times and took
reached-the-floor from 2 to 1.

## What would actually fix it

A sensor-only agent needs a traversable representation of stairs, not a better
portal or a better prior. Three candidates, in the order their evidence supports:

1. **A climb behaviour.** When a pursuit reaches a portal and stops making
   vertical progress, the failure is that forward is blocked by a step the
   policy will not take. A scripted ascent at the foot of a detected stair
   region is the smallest thing that could work.
2. **A multi-level costmap.** Per-floor 2D grids cannot express the connection
   between them. The stair cells are already detected; they need to be
   navigable, not masked out.
3. **Stair-aware goals.** Aim at the stair region's lower mouth rather than at
   the patch of upper floor that is visible from below.

Until one of those exists, cross-floor numbers on the sensor-only line measure
an agent that cannot ascend on purpose, and no work in the floor decision layer
can register.

## The climb, built and measured

Everything above was diagnosis. This is what happened when the sensor-only
climb was actually built.

**What was there.** `State.CLIMB` was in the enum; `_carrot_action`,
`_on_a_staircase` and `_left_the_stairs` were on `NavAgent`; the `ascent` policy
dispatched to a `_do_climb` that was never defined; nothing assigned the state
or any attribute it read. `StairDetector.accumulate` ran only on the one frame
of a periodic look-down, `down_look_every` defaulted to 0 so it never ran on any
YCB preset, and `StairDetector.extract` had no caller at all. In short, no
sensor-only climb existed, and no stair evidence was ever collected.

**What was built**, one arm at a time, each default-off:

| arm | adds |
|---|---|
| v9 | `agent.climb_enabled`: a pursuit that reaches the stairs becomes a climb (ASCENT's depth-ray carrot, tilt-down on descent, exit on committed storey / budget / stall, failure → portal memory). `floor.climb_targets: stairs_first`. |
| v10 | stair evidence on every keyframe; look down every 30 steps; extracted up/down regions offered as direction-aware targets |
| v11 | entry reach outside the mover's own 0.9 m stop radius; `floor.hold_pursuit` so a pursuit in flight is not re-issued every round (which is also what base's 55 attempts were) |
| v12 | carrot aimed at the detector's stamped stair cells; a turn after a run of blocked forwards; budget sized for a switchback |

**What it did**, six scenes, twelve genuine storey changes, paired:

| | base | navmesh | v10 | v11 | v12 |
|---|---:|---:|---:|---:|---:|
| success, cross-floor | 0 | 2 | 0 | 0 | 0 |
| reached the object's floor | 1 | 1 | 0 | 0 | 0 |
| climbs started | 0 | 0 | 4 | 11 | 11 |
| climbs that committed a storey | 0 | 0 | 0 | 0 | 0 |
| most height gained inside a climb | 0 | 0 | 0.34 m | 0.34 m | 0.34 m |
| floor switch attempts | 76 | 9 | 96 | 18 | 15 |
| look-downs | 0 | 0 | 87 | 83 | 69 |
| stair regions offered | 0 | 0 | 17 | 18 | 21 |

Every mechanism is live and measurable. Eleven climbs engaged, each ended
properly and was remembered, switch attempts fell from 76 to 15, the agent
looked down 69 times and was offered 21 stair regions. And the best any climb
managed was 0.34 m.

**Why it does not ascend.** On 00821's cracker box, where the navmesh climbs
the 3.6 m storey in one attempt, the stair target the detector produced sat at
(2.98, 2.62) against a true foot near (0.37, 2.0): about 2.7 m to the side of
the flight. v12's cell carrot then spent 386 steps steering at stamped cells
around that wrong position, with no forced-forwards logged, so the mover was
moving and simply never on a tread. The detector's `stairs` mask, seen from
below, stamps cells under the upper flight rather than at its foot. The depth
ray (v9-v11) has the same problem from a different direction: it works from ON
the flight, and the agent is never on it.

**The other half never opens.** On 00878's tin can, where the navmesh climbs
3.0 m and the prior map records two traversals, no sensor-only arm made a single
switch attempt. The posterior asked once, the model answered "stay", and that
settled it; the geometric gate stayed closed; and the gate override never ran
because the agent spent almost all 500 steps in approach and surface states,
where the storey question is never asked. The commit table says why: on all
twelve episodes the first commit is to a live detection of the target on the
start floor, a false positive on the wrong storey, and the prior anchor is
chosen in three.

**What is worth keeping.** v11's plumbing: reach outside the mover's radius,
`hold_pursuit`, portal failure memory, evidence every keyframe, look-down. Those
are correct regardless of the carrot and they cut the thrash from 76 attempts
to 15. Leave `climb_targets: stairs_first` and `climb_cell_carrot` off until the
stair evidence can be shown to land at a foot: on the two scenes checked it
does not.

**What would actually climb.** The navmesh arm shows the rest of the pipeline
can reach the floor once a mover can take stairs. The sensor-only mover cannot:
a frozen PointNav policy trained on flat point-goals treats a flight as a wall,
and the only thing that has moved an agent up one here is the depth carrot for
0.34 m. Three options, in order of evidence: a local planner on a costmap that
represents the flight as traversable (the stair cells are already detected;
they are masked out, not used); a stair-following controller that keeps the
agent square to the risers and pushes; or stair evidence taken from the height
layer's rising-tread signature rather than from the detector's mask, so the
target is the foot and not the balustrade.
