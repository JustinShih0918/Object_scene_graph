# Overnight iteration toward 70 / 50 on DualMap's released benchmark

Branch `experiment/sr-70-50`, started 2026-09-09 19:15. Goal: in-anchor 70%,
cross-anchor 50% at DualMap's own rule (1 m to the object, three attempts),
so that the paper can put our number beside DualMap's measured 64.8 / 30.2 on
the same 107 trials. Everything below is appended as it happens; the decision
rule is stated first so the reader can check that it was followed.

## Protocol

1. **Small batch first.** Every arm runs on the *current best run's failures*
   (`scripts/make_hard_subset.py --reference <best>`; scissors and mug dropped,
   they are a perception wall), six shards, ~25 minutes. A subset of failures
   can only show conversions, never regressions, so it answers one question:
   does the mechanism fire and convert anything at all.
2. **Full 107 for anything that converts.** Sharded (`data/splits/dualmap_all.json`,
   `SHARDS_PER_SCENE=2`), ~50 minutes. Paired against the best run with the sign
   test; the funnel and the mechanism counters are read before SR.
3. **One change per arm.** An arm's preset composes on the current best preset
   and sets exactly one thing.
4. **What "best" means.** Highest in-anchor + cross-anchor on the full 107 with
   no split worse by more than one trial than the previous best.
5. **GPU budget.** 10 GB card exposing 9.6 GB; a single-floor process is 1.3 GB.
   At most seven processes; the 15-scene authored run is paused for the night.

## Where we start

| run | in-anchor | cross-anchor | note |
|---|---:|---:|---:|
| DualMap, measured, seed 12 | 35/54 (64.8%) | 16/53 (30.2%) | its own harness on its own data |
| tight ring (baseline of the day) | 24/54 | 15/53 | |
| + look before absence (`look`) | 25/54 | 16/53 | |
| + flat proximity (`look_flat`) | 24/54 | 18/53 | own-surface arrivals 7 -> 13 |

Ceilings from the static-pass split (docs/SR_PROPOSAL_CLOSE_LOOK.md): the
target is in the prior map for 39/54 in-anchor and 39/53 cross-anchor trials;
the rest are scissors, mug and five cracker boxes at 20% / 7%. So 70/50 means
about 35/39 in-anchor and 26/39 cross-anchor on the trials the detector can see.

## Levers, in the order they are tried

1. **Stale-anchor policy** (`look_flat_anchor`): stop once at the silent stale
   anchor; retire its unseen prior-map twins after a refutation. Running.
2. **Proximity until refuted** (`search_drop_proximity_after_absence` on
   `search_proximity_len_m = 1.0`), so in-anchor's second attempt searches the
   stale surface's neighbourhood before sweeping.
3. **Stop on ambiguous absence** (`abandon_below_p` 0.3): a second stop where the
   belief has not collapsed, paid in attempts.
4. **Perception, scissors and mug**: names on the dumped frames first (offline),
   then a foveated second look at close container surfaces
   (`scene_graph.foveate_containers`) if the names do nothing. If neither moves
   in-situ recall off zero, the honest option is to substitute detectable
   objects in the released layouts, which costs a DualMap rerun on the
   substituted layouts (their harness, ~10 hours for three seeds) and is
   recorded here as a cost, not taken tonight.

## Log

### 19:30 — Perception: scissors and mug cannot be named in situ, by anything cheap

`scripts/probe_nearmiss.py` over every dumped in-view keyframe of the tight-ring
run (`outputs/nearmiss_tightring/{scissors,mug}.log`): 352 scissors frames and
160 mug frames, ranges 1.4-5.2 m, the frames on which the object was in front
of the camera and unoccluded.

| condition | scissors | mug |
|---|---:|---:|
| the run's own name and settings | 0/352 | 0/160 |
| every furniture competitor removed | 0/352 | 0/160 |
| window around the object, upscaled (160 px, 96 px) | 0/352, 0/352 | 0/160, 0/160 |
| ten alternative names | 0/352 for all ten | 0/160 for nine; `cup` 9/160 (0.06) |

The detector produces nothing over these two assets at any name, with no
competitor to lose to, at any magnification: not a threshold, not a name, not
scale. The bake-off (PERCEPTION_INVESTIGATION.md) already showed DualMap's own
detector at 0.06 on scissors. These 24 trials (18 scissors, 6 mug) are a
ceiling for both systems: without them the best attainable is 45/54 in-anchor
(83%) and 44/53 cross-anchor (83%), so 70/50 does not need them. The only
ways to move them are a different recogniser (a VLM on crops, at a per-frame
cost this pipeline cannot pay) or substituting detectable objects in the
released layouts, which requires rerunning DualMap on the substituted layouts
(~10 h of their harness for three seeds). Neither is taken tonight; the night
goes to the 83 trials the detector can see.

### 19:40 — Stale-anchor arm, first run: twin retirement alone (the stop was defective)

`outputs/osg_anchor/flat_anchor`, full 107, paired against `look_flat`
(`outputs/osg_anchor/TWINS_AB.md`). In this run the stale-anchor stop was
granted 29 times and taken zero: the close look's silent path re-approached
the ring and the once-only grant was spent before the arrival (fixed in
`ddacbca`). So this run measures the twin retirement, which fired in 58 of
107 trials.

| | look_flat | + twin retirement |
|---|---:|---:|
| in-anchor SR | 24/54 | 24/54 (+1 / -1) |
| cross-anchor SR | 18/53 | **20/53** (+4 / -2, p = 0.69) |
| cross-anchor: target ever named | 27 | 31 |
| cross-anchor: seen, never named | 16 | 13 |
| cross-anchor: episodes that selected a far surface | 40 | 44 |
| median steps | 371 | 500 |
| trials at budget | 48 | 54 |

Retiring the ghosts converts walking into searching: more surfaces selected,
more targets named, two more cross-anchor successes -- and every freed step is
spent, so the median episode now runs to the budget. The budget is the
binding constraint again, one level down. In-anchor is untouched, as expected
without the stop.

### 20:45 — Stale-anchor policy, fixed stop: cross-anchor 20, in-anchor still 24

`outputs/osg_anchor2` (`ANCHOR_AB.md`), full 107 against `look_flat`:
in-anchor 24/54 (+1 / -1), cross-anchor **20/53** (+4 / -2). `absence_abandon`
fell 52 to 5: the stops replaced the abandons. Then why no in-anchor gain?
Tracing all 54 in-anchor trials:

* The stale stop was taken in **9** trials, at a median **1.47 m** from the
  object: one inside the metre, four at 1.0-1.5 m, four beyond. The stop is
  taken on the approach's ring, 0.65-0.8 m from the track centre, and the
  object moved a median 0.7 m the other way. Right idea, wrong spot.
* In 00848 the pitcher (3 trials) and plate commit first to a **ghost** 8 m
  from the object -- a prior-map false positive that the detector keeps
  re-detecting live, in three fragment tracks 0.2 m apart -- and spend all
  three attempts stopping on it: the twin rule cannot touch a track that is
  seen live, and a failed attempt strikes one fragment while the next commit
  takes the next.
* Three 00848 cracker-box trials end the budget 0.4-1.8 m from the object
  without a stop: the true track's belief was driven to the negative clamp by
  missed expectations on the way and one live sighting cannot lift it over
  the bar.

Two fixes, both behind flags (`stale_stop_at_nearest_free`,
`failed_attempt_disables_place`): the stop walks to the nearest navigable
point to the track centre first, and a failed attempt disables every track
of the label within `fp_disable_radius_m` of the stop's centre. The third
item is the presence clamp and is left for later.

### 20:45 — Proximity until refuted, on the subset: rejected

`outputs/osg_next/flat_anchor_drop` on look_flat's 32 failures, against the
fixed stale-anchor arm on the same ids: in-anchor 2/8 against 1/8,
cross-anchor **2/24 against 4/24** (1 gained, 3 lost). Anchoring the search
on the stale spot until it is refuted costs cross-anchor what it was meant
to give in-anchor. Not carried forward.

### 21:15 — Second cut on the 21 in-anchor failures: 2 converted, the rest never commit

`outputs/osg_next/flat_anchor_v2` on the fixed arm's in-anchor failures with
scissors and mug dropped (`V2_INANCHOR.md`). The nearest-free stop fired 6
times and now lands at 0.66-1.8 m (was 1.15-1.9 on the ring); two soup-can
trials converted at 0.94 and 0.66 m. Place retirement fired 33 times. The
other 19 stay failures, and the funnel says where: **admitted, never
committed 9 of 21** -- the detector did name the true object, and the agent
never made it a goal. The true track's belief sits at the negative clamp,
driven there by missed expectations on the way, and a single sighting is not
enough to lift it over the bar. Two arms next, both one knob:
`presence.l_clamp` 6 -> 2 (a sighting can bring a track back) and
`rank_candidates_by_presence` off (confidence counts again).

### 21:25 — Second stop on ambiguous absence: null

`outputs/osg_next/flat_anchor_stop2` on look_flat's 32 failures, against the
fixed stale-anchor arm on the same ids: every funnel row identical, 5/32
both. `absence_abandon` was already 4 on these trials after the stale stop
took over the silent arrivals, so a knob that only changes what happens
after an abandon has almost nothing left to act on. Not carried forward.

### 21:50 — Belief clamp 6 -> 2: null

`outputs/osg_next/flat_anchor_v2_clamp` on the same 21 in-anchor failures:
funnel identical to the second cut, 2/21. The true track's belief is not
what keeps it from being committed; the block is elsewhere in the candidate
gates, and the next step is to read those gates on the ten trials directly.

### 22:00 — Found it: "unreachable", twice from the same pose, retires the true track

Reading the candidate gates on the in-anchor failures where the target was
admitted and never committed: the true object HAS a live track in most of
them -- 00848 pitcher: 0.00 m, belief 0.95, 19 observations; 00880 plate:
0.05 m, 0.95, 10 observations; 00880 cracker box: 0.03 m, 0.95 -- and it
was never a goal. `candidate_reject_log` says why, 37 times on the in-anchor
failures and 16 on the cross-anchor ones, every entry the same word:
**unreachable**. The viewpoint planner finds no standable ring pose at that
moment (a plate on a desk against a wall, a pitcher on a bed), the
reachability check strikes the track, the next step asks the same question
from the same pose and strikes it again, and `max_identity_rejections = 2`
retires it for the episode. The restrike guard (`unreachable_restrike_m`)
was written for exactly this and has been shipping at 0.0, off. Arm: 1.0 m,
so retiring a track takes two strikes from places more than a metre apart.
