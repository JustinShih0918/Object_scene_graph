# Rerunning DualMap on its own released benchmark

Two measurements live here. The first is DualMap, run unmodified on the dataset
its authors released, because the published numbers cannot be reproduced without
an external harness. The second, from "Running our own pipeline on the same
benchmark" onward, is our stack over the same 186 trials, from the same start
poses, scored by the same rule -- so the comparison is paired trial by trial
rather than table against table.

## What the official repository provides

Upstream: `https://github.com/Eku127/DualMap.git`, commit `157235e`, used unmodified
(`git status` clean). Dataset: the authors' released `HM3D_collect` (scenes
`00829-QaLdnwvtxbs`, `00848-ziup5kvtCCR`, `00880-Nfvxx8J5NCo`), each shipping
`static_scene_config.json`, `dynamic_scene_config/{in_anchor,cross_anchor}/`,
`global_map/`, `class_bbox.json` and `class_num.json`.

Two facts determine everything below.

1. **Neither official repository computes navigation metrics.** Grepping DualMap
   and `habitat-data-collector` for `spl|success_rate|geodesic|path_length`
   returns nothing. `evaluation/` holds only semantic-segmentation code, which
   is the mapping table, not the navigation table.
2. **The documented navigation workflow is manual.** `resources/doc/app_simulation.md`
   has the operator edit `config/actions.yaml` (`get_goal_mode: inquiry`,
   `inquiry_sentence`, `calculate_path: true`), watch Rerun/RViz, and judge the
   outcome by eye: "If no local path is planned, the navigation attempt is
   considered failed."

So following the guide reproduces the *demo*, not the numbers. Measuring SR and
SPL over 186 trials × 3 seeds requires an external harness. Ours
(`scripts/run_dualmap_released_native.py`) imports the released DualMap and
collector packages and drives them through their own public interfaces
(`actions.yaml`, `core.calculate_path`, `core.trigger_find_next`); it replaces
ROS 2 only as transport. No upstream source file is modified.

## Metrics

DualMap defines SR as "the percentage of queries in which the agent stops within
1 meter of the queried object", and for dynamic scenes "success further requires
finding the target within three attempts". That is the criterion implemented here.

**DualMap does not publish SPL.** Paper Table II and Appendix Tables IX/X report
success only — Tables IX/X are literally binary `Type | Success` columns. SPL in
our tables is measured by the harness against Habitat's geodesic shortest path
and has no published counterpart to validate against.

Published SR (paper Table II):

| Split | 00829 | 00848 | 00880 | Trials | Avg. |
|---|---:|---:|---:|---:|---:|
| Static | 73.1% | 69.2% | 69.2% | 78 | 70.5% |
| In-anchor | 66.7% | 66.7% | 61.1% | 54 | 64.8% |
| Cross-anchor | 55.6% | 61.1% | 64.7% | 53 | 60.3% |

## Protocol reconstruction

* **Dynamic splits** come straight from the released layout JSONs: 6 queried
  objects × 3 layouts × 3 scenes, minus the cracker-box trial that the appendix
  does not report for `00880` `0128-2`. 54 in-anchor + 53 cross-anchor = 107,
  matching the paper.
* **Static split** uses the query lists in Appendix Table VIII. Those total
  **79** (26 + 27 + 26), while Table II reports **78** trials, and the per-scene
  percentages (73.1% = 19/26, 69.2% = 18/26) imply 26 per scene. The paper does
  not say which `00848` query was dropped, so all 79 are run and the discrepancy
  is reported rather than silently resolved.
* **Targets.** YCB objects use the released poses. HM3DSem classes use the
  instance boxes in `class_bbox.json`, which the release ships "for evaluation".

## Deviations from the harness's previous behaviour, and why

Three changes were needed. Each is in our harness, not in DualMap.

1. **A false match no longer ends the trial.** DualMap clears
   `ignore_global_obj_list` whenever a local path is planned (`dualmap/core.py`),
   so after a false match it has forgotten which anchors it already visited and
   the next global plan re-selects the one just rejected. The harness previously
   terminated the trial there, which contradicts the paper's three-attempt rule
   and the guide's `trigger_find_next` ("very useful in dynamic object
   navigation"). The harness now restores the anchors already tried and requests
   a fresh global plan. Only DualMap's public state is set.
2. **Distances are measured to the object, not its centroid.** `class_bbox.json`
   gives full box extents. A bed is 2.33 m across, so its centre is more than a
   metre from any navigable floor and "within 1 m" would be unsatisfiable by
   construction. Distances are now taken to the box. YCB targets keep a zero
   extent, so the dynamic splits are mathematically unchanged.
3. **The collector seed is a run parameter**, so the agent start pose can be
   varied across seeds. The official protocol has no seed: the layouts are fixed
   files, and seeding is our addition to estimate run-to-run variance.

## Results: DualMap

Full tables: `outputs/dualmap_official_bench/RESULTS.md`, aggregated across seeds by
`scripts/report_dualmap_official.py`. Three seeds (12, 13, 14),
186 trials each; 556 of 558 completed, 2 lost to an upstream crash.

| Split | SR (ours, 3 seeds) | SR (published) | SPL (ours) |
|---|---:|---:|---:|
| Static | 68.1% ± 1.1 | 70.5% | 0.327 ± 0.044 |
| In-anchor | 62.3% ± 2.8 | 64.8% | 0.431 ± 0.019 |
| Cross-anchor | 30.2% ± 0.0 | 60.4% | 0.088 ± 0.019 |

Static and in-anchor reproduce within ~2.5 points. **Cross-anchor does not
reproduce**: 30.2% against a published 60.4%, and the seed-to-seed standard
deviation is 0.0 — 16/53 on every seed. That is a systematic difference, not
sampling noise, and it survives the false-match fix that lifted cross-anchor
from 22.6% to 30.2%.

What was ruled out for cross-anchor, with evidence:

* Not a missing candidate: all 53 trials produce a global plan every seed.
* Not an unwalked local path: 6+ keyframes always remain after one is planned.
* Not a truncated retry budget: 42/53 trials now use all three attempts.
* Not a broken online update: the preloaded abstract map for `00829` holds 21
  objects, 9 of them carrying `related_objs`, and candidates do change across
  attempts in 21/53 trials.
* Not a similarity ceiling artefact: scores top out at 0.640 in every split,
  including in-anchor, which is MobileCLIP's natural cosine range.

What remains is candidate *ranking* after relocation: 30 of 37 cross-anchor
failures end more than 3 m from the target, so the agent commits to wrong
anchors rather than narrowly missing correct ones. The release publishes no
per-trial start poses, so the authors' exact observation trajectories cannot be
reconstructed to close this gap.

## Two upstream crashes

`00848 static kettle` and `00880 static painting` raise `ValueError: zero-size
array to reduction operation minimum` at `utils/object_detector.py:1792`
(`update_bbox` via `overlap_check`), an empty-mask edge case entirely inside
DualMap. The failures are seed-dependent — seed 12 ran `kettle` cleanly. They
are excluded from the denominators and listed in the results rather than scored.

## Where the local path points

Full tables: `outputs/dualmap_official_bench/LOCAL_PATH_ANALYSIS.md`, regenerated by
`scripts/analyze_dualmap_local_paths.py`. Saved paths are in DualMap's z-up frame;
Habitat `(x, z)` is `(path_x, -path_y)`, verified by the global path's first point
coinciding with the agent pose and its last point with the local path's first point.

| Split | Local paths planned | Goal within 1 m of object | Median goal error |
|---|---:|---:|---:|
| Static | 189 | 63.5% | 0.57 m |
| In-anchor | 104 | 67.3% | 0.65 m |
| Cross-anchor | 98 | 38.8% | 2.81 m |

Cross-anchor's failure is **localisation, not control**. When the goal is right the
agent nearly always reaches it (33/38 cross-anchor, 67/70 in-anchor); the problem is
that 61% of cross-anchor local paths aim at the wrong object, a median 2.81 m away.
The chain is: wrong anchor selected globally, so the local map holds the wrong
objects, so the best local match is a different instance.

Position-correct SR — crediting success only when DualMap actually localised the
object and the agent reached it — is **50.2% static, 41.4% in-anchor, 20.8% cross-anchor**.

This makes the headline SR *generous* rather than harsh. Of 48 cross-anchor successes,
only 33 involved correct localisation; 14 had no local path at all (the agent reached
the anchor and the object happened to lie within 1 m), and 1 succeeded with a wrong goal.

It also rules out the reconciliation with the published number. Counting any planned
local path gives cross-anchor 61.6%, close to the published 60.4% — but only 38 of
those 98 paths point at the real object. Requiring correctness drops it to 23.9%,
*below* our 30.2% headline. Every criterion that checks where the object actually is
lands in the 20-30% band:

| Criterion (cross-anchor) | SR |
|---|---:|
| Any local path planned | 61.6% |
| Agent within 1 m of object (headline) | 30.2% |
| Local-path goal correct | 23.9% |
| Local-path goal correct and reached | 20.8% |

## Running our own pipeline on the same benchmark

Everything above measures DualMap. This section measures ours, on the same 186
trials, and what lets the two columns sit beside each other is that almost
nothing about a trial is chosen twice.

**One definition, imported by both harnesses.** The trial list, what each query
points at, and the success rule are `src/osg/eval/dualmap_release.py`.
`scripts/run_dualmap_released_native.py` imports it, and so does the new
`dualmap_protocol` eval mode (`src/osg/sim/dualmap_env.py`), so "both were scored
by the same criterion" is a property of the import graph rather than a claim in a
paragraph. `tests/unit/test_dualmap_release.py` recomputes the target geometry
and every per-attempt distance of all 186 DualMap trials already on disk from
that module; a change to it that would silently rescore one system fails the
suite instead of moving a number in a table.

**The same start.** Each of our trials begins where that trial's DualMap run
began, read from its own `result.json`. Trials are therefore PAIRED -- same
query, same layout, same place -- which is what makes the per-trial agreement
table below meaningful; without it, ten points between two 53-trial runs is
indistinguishable from a different draw of starts.

**The same rule.** A stop scores when the agent is within 1 m horizontally of any
instance of the queried object, within three attempts. In this mode our agent's
own success distance no longer decides its attempts -- the protocol's does,
through `attempt_scored` -- and habitat's own success and SPL are still recorded
alongside, under `habitat_*`, rather than reported as the comparison.

### Results

Ours is one run against seed 12's starts; DualMap's column is that same seed
(its three-seed means are 68.1 / 62.3 / 30.2).

| Split | Trials | SR (ours) | SR (DualMap, measured) | SR (published) | SPL (ours) | SPL (DualMap) |
|---|---:|---:|---:|---:|---:|---:|
| Static | 79 | 50.6% | 67.1% | 70.5% | 0.213 | 0.285 |
| In-anchor | 54 | 33.3% | 64.8% | 64.8% | 0.120 | 0.438 |
| Cross-anchor | 53 | 22.6% | 30.2% | 60.4% | 0.057 | 0.072 |
| **All** | 186 | **37.6%** | **55.9%** | n/a | **0.142** | **0.269** |

**DualMap wins this benchmark, and not narrowly.** On paired trials the sign test
over discordant pairs gives p = 0.029 for static and p < 0.001 for in-anchor;
cross-anchor is the one split where the difference is not significant
(7 ours-only against 11 theirs-only, p = 0.48), and 00829 cross-anchor is the one
cell we win, 44.4% against 33.3%.

| Split | Both | Ours only | DualMap only | Neither | p |
|---|---:|---:|---:|---:|---:|
| Static | 31 | 9 | 22 | 17 | 0.029 |
| In-anchor | 15 | 3 | 20 | 16 | 0.000 |
| Cross-anchor | 5 | 7 | 11 | 30 | 0.481 |

Full tables: `outputs/osg_dualmap_protocol/COMPARISON.md`; the breakdown below is
`outputs/osg_dualmap_protocol/GAP_ANALYSIS.md`, regenerated by
`scripts/report_osg_vs_dualmap.py` and `scripts/analyze_osg_dualmap_gap.py`.

### What the shortfall is made of

Three mechanisms account for most of it, and they are not the same mechanism in
each split.

**1. Static: our prior map cannot be queried by text.** DualMap preloads a map
whose objects each carry a CLIP embedding, so any query is answered by
retrieval. Ours preloads tracks labelled from a fixed 47-class detector
vocabulary, so a query outside it has nothing in the prior to retrieve and the
agent has to find it by exploration. Splitting our own static trials on that one
fact:

| Static queries | Trials | Our SR |
|---|---:|---:|
| Named in our prior map | 40 | 70.0% |
| Absent from our prior map | 39 | 30.8% |

Where the prior can answer, we are level with DualMap (70.0% against its 67.1%).
Where it cannot -- `ottoman`, `washbasin counter`, `decoration`, `kitchen island`,
33 distinct queries in all -- we lose more than half. This is an architectural
difference, not a tuning one, and it is the single largest item in the table.

**2. Dynamic: 16 failures stopped in the 1.0-1.5 m band.** Our approach drives to
a sampled viewpoint, and `ViewpointPlanner` samples rings at 0.8 / 1.2 / 1.5 /
2.0 m -- radii calibrated for HM3D ObjectNav, which scores distance to a goal
*view point*. This benchmark scores distance to the *object*. Stopping on the
1.2 m ring is a guaranteed miss by 0.2 m, and the median closest approach across
the 20 in-anchor failures where the detector did name the target is 1.195 m.

| Split | Failures | Ended 1.0-1.5 m from the object | SR if only those converted |
|---|---:|---:|---:|
| Static | 39 | 3 | 50.6% -> 54.4% |
| In-anchor | 36 | 6 | 33.3% -> 44.4% |
| Cross-anchor | 41 | 7 | 22.6% -> 35.8% |

The right-hand column is a diagnosis, not a result: it is what this benchmark
would have scored had the stop rule been aimed at its criterion rather than at
HM3D's. It is worth one measured run of `agent.success_distance` and the
viewpoint radii before any of the rest is worked on, because it is the cheapest
point on the board -- and on cross-anchor alone it would move us above DualMap.

**3. We search longer and convert less.** The two budgets are the same number and
not the same thing.

| | Ours | DualMap |
|---|---:|---:|
| Median distance travelled | 31.1 m | 18.0 m |
| Median budget used | 327 of 500 steps | 94 of 500 keyframes |
| Trials that exhausted the budget | 72/186 | 0/186 |

DualMap never runs out; we do, in 39% of trials. That is not a case for a larger
budget -- we already travel 1.7x as far -- it is the shape of the difference: its
map answers the query and it drives there, while we explore.

What the failures were made of, over the trials whose query has exactly one valid
instance (the ground-truth instrument follows one object, so multi-instance class
queries are excluded rather than counted as never-seen):

| Split | Our failures | Never saw it | Saw it, never named it | Named it, still failed |
|---|---:|---:|---:|---:|
| Static | 20 | 10 | 8 | 2 |
| In-anchor | 34 | 6 | 8 | 20 |
| Cross-anchor | 40 | 13 | 14 | 13 |

In-anchor is the split where the target was most often *found and then lost*: 20
of 34 failures had the detector name the object and the trial still ended more
than a metre away, half of them at the step budget. Cross-anchor divides evenly
between never arriving, arriving without recognising, and recognising without
converting.

### What could not be shared

| | Ours | DualMap |
|---|---|---|
| Camera height | 0.88 m | 1.5 m |
| Sensor | 1280x960, 79 deg H / 63 deg V | 1200x680, 90 deg H / 59 deg V |
| Motion | discrete, 0.25 m / 30 deg | continuous path follower |
| Prior map | our mapping pass over the static scene: 689 / 343 / 520 tracks | the released `global_map/`: 21 / 31 / 35 objects |

The prior-map row is the largest asymmetry and it favours us: each system was
given one pass over the same static scene and preloads whatever its own pipeline
produces, and ours produces an order of magnitude more objects. It is stated
because a reader who does not know it will read the static split wrongly -- our
prior is far richer and still answers fewer queries, which is the point of
mechanism 1.

The embodiment difference was measured rather than assumed. DualMap's start poses
land on our navmesh with a median snap offset of 0.000 m (max 0.063 m), and the
shortest-success-path optimum recomputed on our navmesh differs from DualMap's by
a median of 0.111 m. The pairing is real, not approximate.

**Query strings are not shared, deliberately.** The benchmark asks for an object;
which string to hand a text-conditioned detector is part of the system. DualMap
asks MobileCLIP for `pitcher`; the probe behind this repository's
`YCB_TARGET_LABELS` measured `pitcher` at 0.00 on that asset at every resolution
and `blue plastic pitcher` at 0.71. Three of the eight YCB queries are renamed for
our detector -- `soup can` to `tin can`, `plate` to `red plate`, `pitcher` to
`blue plastic pitcher` -- and the other five and all 60 HM3DSem class queries pass
through unchanged. The mapping already existed in the repository and predates this
run; every trial records both strings.

### Run provenance

186/186 trials completed, no crashes on our side. The hosted endpoint failed once
(one VLM call in `00848 cross_anchor 0129-1 banana`) out of 186 trials; no episode
recorded an LLM scorer error. The algorithm is `ycb_dynamic_best` unchanged --
the configuration calibrated on the authored dynamic benchmark -- because the
question is how the system we have compares, not how well it can be fitted to
someone else's trial list. Three scenes ran in parallel, 11:51 to 16:07.

    python scripts/run_eval.py +experiment=dualmap_protocol_osg \
      'dualmap.scenes=[00829-QaLdnwvtxbs]' \
      ycb.map_in=outputs/maps_v5/00829-QaLdnwvtxbs \
      output_dir=outputs/osg_dualmap_protocol/00829-QaLdnwvtxbs
    python scripts/report_osg_vs_dualmap.py
    python scripts/analyze_osg_dualmap_gap.py

## What the benchmark can resolve

Everything above compares two systems under the benchmark's own rule. This section
asks a prior question — whether that rule can resolve what it claims to — and it
needs no system at all. `scripts/analyze_benchmark_validity.py` writes
`outputs/osg_dualmap_protocol/BENCHMARK_VALIDITY.md`.

Take a do-nothing agent: it ignores the dynamic change entirely, walks to where the
object sat in the *static* scene, and stops. It performs no perception, no search
and no re-detection. Score it under the benchmark's own 1 m rule against the
object's *new* position.

| | in-anchor | cross-anchor |
|---|---:|---:|
| Median object displacement | **0.70 m** | 5.56 m |
| Do-nothing agent's SR | **74.1%** | 9.4% |
| DualMap (measured, seed 12) | 64.8% | 30.2% |
| Ours | 33.3% | 22.6% |

In-anchor moves objects less far than its own success tolerance, so doing nothing
outscores both real systems. That condition cannot separate a system that
re-detects a moved object from one that never noticed it moved, and an in-anchor SR
should not be read as evidence about dynamic-scene handling. Cross-anchor, whose
objects move a median 5.56 m, is the informative half.

Six trials move the object by less than a centimetre and are labelled dynamic
anyway — including **all three cross-anchor `mug` trials**, so `mug` contributes no
cross-anchor evidence at all. Separately, 102 of the 107 dynamic trials have exactly
one valid instance, so a wrong stop is a wrong *place* rather than a sibling
instance; the exception is `mug` in 00848, which the release gives two instances
sharing one semantic id.

## What DualMap's success rate is made of

A claim worth stating carefully, because the obvious version of it is false.

**DualMap does not replay stale positions.** Its saved local paths record where it
planned to go. Restricted to the 88 planned goals across three seeds whose object
moved more than 1.5 m — so that "aimed at the new position" and "aimed at the old
one" are distinguishable — **30 aimed at the new position and 1 at the old**. When
DualMap plans a local path it has genuinely re-detected the object, including
objects our own detector never sees. Any claim that it navigates to where the object
used to be is not supported.

What is true is that its success rule credits arrival without identification. A
DualMap trial can score having never planned a local path at all: its abstract map
text-matches a piece of furniture, the agent walks to that anchor, and the anchor
happens to lie within a metre of the object. Over three seeds:

| Split | Anchor-only successes | What they stopped at |
|---|---:|---|
| Static | 36 | `countertop` ×7, `cabinet` ×5, `shower wall` ×5, `couch` ×4, … |
| In-anchor | 34 | `couch` ×10, `kitchen island` ×9, `dining table` ×6, `desk` ×5, `bed` ×4 |
| Cross-anchor | 14 | `bed` ×9, `kitchen island` ×3, `dining table` ×1, `couch` ×1 |

All six of 00848's `scissors` trials are this pattern — `candidate: bed`, no local
path ever planned, three of them scored. The anchor score is identical across
layouts (`desk` at 0.5651295544750305 for `plate` in 0116/0117/0118), which is what
you expect from a static text match against the prior map: it does not depend on
where the object went.

Scoring success only when the system localised the object gives, over three seeds,
static 68.1% → 51.1%, in-anchor 62.3% → 43.2%, cross-anchor 30.2% → 23.9%.

### The comparison under that metric

Two quantities, and they are not the same. **Localised** is where a system *aimed*
— its committed goal within 1 m of a real instance, whether or not it arrived.
**Position-correct SR** is the strict success rate: aimed right *and* stopped within
1 m. Both are computed for both systems by `dualmap_release.position_correct`, on
seed 12's paired trials.

| Split | Localised (ours / DualMap) | SR raw (ours / DualMap) | SR position-correct (ours / DualMap) |
|---|---:|---:|---:|
| Static | 50.6% / 50.6% | 50.6% / 67.1% | 48.1% / 50.6% |
| In-anchor | **50.0% / 40.7%** | 33.3% / 64.8% | 33.3% / 40.7% |
| Cross-anchor | **34.0% / 28.3%** | 22.6% / 30.2% | 20.8% / 24.5% |

Our strict SR is within two points of our raw SR, because our successes already
require the object to have been detected and committed to. DualMap's falls by 16.5,
24.1 and 5.7 points. The 31.5-point raw in-anchor gap becomes 7.4; the static gap
becomes 2.5.

And **our localisation rate is the higher of the two in both dynamic splits.** We
find the moved object more often than DualMap does, and lose on the last metre.
That is a different problem from the one the raw table describes, and it is the one
worth fixing.

On the perception-controlled subset — `scissors` and `mug` removed from both
systems, for the reasons above and below — the strict metric gives in-anchor
**38.1% ours against 33.3%**, cross-anchor 24.4% against 24.4%, static 50.7% against
50.7%. This is a secondary reading and is reported as one; the full table is the
result.

## Where our own dynamic episodes fail

`scripts/analyze_dynamic_failures.py` assigns each of the 107 dynamic trials to the
first stage it failed, so the buckets are exhaustive and disjoint. Stages 1–4 read
the ground-truth instrument, which is observed beside the run and never given to the
agent.

| Stage | in-anchor (54) | cross-anchor (53) | All |
|---|---:|---:|---:|
| succeeded | 18 (33%) | 12 (23%) | 30 |
| never in view | 6 (11%) | 13 (25%) | 19 |
| seen, never named | 10 (19%) | 15 (28%) | 25 |
| named, never admitted | 3 (6%) | 2 (4%) | 5 |
| admitted, never committed | 5 (9%) | 3 (6%) | 8 |
| committed elsewhere | 4 (7%) | 1 (2%) | 5 |
| committed, never arrived | 8 (15%) | 7 (13%) | 15 |

**Perception is 44 of the 77 failures.** Success tracks in-situ detector recall
almost exactly — bowl 1.00 recall and 5/12 SR, plate 0.62 and 9/18, against
scissors 0.00 and 1/18, mug 0.00 and 2/6, pitcher 0.08 and 2/18. Nothing downstream
can be assessed on the objects the detector does not name.

**The last bucket is a fixed standoff, not a search failure.** All 15 episodes that
committed to the right object and failed stopped between 0.95 m and 1.59 m out,
median **1.27 m**, against a 1.0 m rule. Every one of them used a **0.80 m
standoff** and reached its own goal (median 0.37 m from it). They are 1.27 m from
the object because the standoff is measured from the committed *track centre*, and
the track centre is not the object centre. 14 of the 15 are within 1.5 m. This is
the cheapest available conversion on the board.

**We spend the budget walking to things that are not the target.** `goal_commit_log`
shows a median of 3 committed goals per episode; 53% of in-anchor commits and 79%
of cross-anchor commits were more than 1 m from the real object, and 24 of 44
cross-anchor episodes had *every* commit wrong. That is why 29/53 cross-anchor
episodes exhaust the 500-step budget at a median 46.2 m travelled, against DualMap's
median 18.0 m and 0/186 exhausted. This is the mirror of the anchor-only successes
we charge DualMap with — with the difference that ours are not scored for it.

## The historical 0.812 / 0.458, and why they are not comparable

`ARCHITECTURE.md`'s condition ladder reports in-anchor 0.812 and cross-anchor 0.458.
Those were measured under the authored benchmark's rule, which is geodesic distance
to the nearest goal *viewpoint* — and viewpoints are sampled on rings at 0.8 / 1.2 /
1.5 / 2.0 m around the object. The effective tolerance is therefore about 2 m to the
object, not the 0.18 m the config names. Of the 61 episodes that rule scored as
successes, only 31 were within 1 m of the object; the median was 0.98 m and the tail
reached 2.37 m.

Rescoring condition N's own 96 frozen episodes against ground truth:

| Threshold to the object | 1.0 m | 1.5 m | 2.0 m | authored rule |
|---|---:|---:|---:|---:|
| in-anchor | 0.396 | 0.729 | 0.833 | 0.812 |
| cross-anchor | 0.271 | 0.417 | 0.438 | 0.458 |

The authored rule lands on the 2.0 m column. The full decomposition to this
document's numbers:

| | in-anchor | cross-anchor |
|---|---:|---:|
| (a) condition N as published | 0.812 | 0.458 |
| (b) the same episodes, rescored at 1 m | 0.396 | 0.271 |
| (c) our released-benchmark run, N's target set | 0.381 | 0.268 |
| (d) our released-benchmark run, all targets | 0.333 | 0.226 |

(a)→(b) is the success criterion, −41.6 and −18.7 points. (c)→(d) is `scissors` and
`mug`, which condition N excluded and the released benchmark includes, −4.8 and
−4.2. (b)→(c) — layouts, start poses and any code drift — is −1.5 and −0.3 points,
which is the only place a regression could hide and is noise at n≈45.

Re-running condition N's exact configuration on 00829 confirms it directly:
in-anchor **17/18**, matching the frozen historical result exactly, and
cross-anchor 14/18 against a frozen 15/18 — a one-episode difference on n=18,
which is run-to-run variation in the asynchronous LLM scoring rather than drift.
(The one episode that flipped is instructive on its own: it began its third
approach at step 466, closed to 0.19 m of its goal, and hit the 500-step budget
before it could stop — it had spent steps 264–466 in `goto_frontier`.) Nothing
regressed; the yardstick changed.

**No number from the ARCHITECTURE.md ladder is comparable to DualMap's published
64.8 / 60.4**, and none should be quoted beside them without this conversion.

## What the old headline was made of, and what has not changed

The conversion above says the yardstick moved. This section asks what the old
yardstick was measuring, and which failures survived the change of yardstick
unchanged -- because those are the ones worth working on. Every table here is
written by `scripts/analyze_headline_anatomy.py` into
`outputs/osg_dualmap_tightring/HEADLINE_ANATOMY.md`, from condition N's 96 frozen
episodes and the tight-ring run's 107, both rescored at 1 m to the object and
pushed through the same failure stages as `analyze_dynamic_failures.py`.

**The old in-anchor number was below doing nothing.** Take the do-nothing agent
of "What the benchmark can resolve", but on the *authored* layouts: walk to where
the object used to be, stop.

| authored layouts | 1 m | 1.5 m | 2 m | condition N as scored |
|---|---:|---:|---:|---:|
| in-anchor | 35/48 | 45/48 | **48/48** | 39/48 |
| cross-anchor | 2/48 | 2/48 | 2/48 | 22/48 |

Every authored in-anchor object moved less than 2 m, and 2 m is what the
viewpoint rule tolerated, so standing still scores 100% and condition N's 0.812
is nine episodes short of it. In-anchor never measured dynamic handling on
either benchmark; the released one merely made that visible (74.1% for doing
nothing against a 1 m rule).

**What the 39 scored in-anchor successes were.** 18 were within 1 m of the
object. 16 stopped in the 1.0-1.5 m band -- all `path_consumed`, 13 of them with
a correct committed goal, which is the fixed standoff the tight ring removed. 5
stopped beyond 1.5 m. Five never detected the target at all and were credited
anyway; four had committed to the stale position; 23 finished closer to where
the object had been than to where it was. Cross-anchor's 22 were 13 real, 7 in
the standoff band and 2 beyond, none undetected.

**The same funnel, at 1 m, for both runs:**

| stage | N in (48) | N cross (48) | now in (54) | now cross (53) |
|---|---:|---:|---:|---:|
| succeeded | 19 | 13 | 24 | 15 |
| never in view | 3 | 15 | 6 | 12 |
| seen, never named | 5 | 4 | 10 | 20 |
| named, never admitted | 2 | 3 | 5 | 2 |
| admitted, never committed | 3 | 2 | 5 | 2 |
| committed elsewhere | 4 | 4 | 2 | 1 |
| committed, never arrived | 12 | 7 | 2 | 1 |

This is not code drift. Flattening condition N's frozen `config.yaml` against the
tight-ring run's Hydra config, the only key present in both that differs is
`verification.ring_radii_m`. The detector block is byte-identical (same weights
hash, `imgsz` 1280, `conf` 0.3).

**What the tight ring fixed: the legs.** Committed to a correct goal, the agent
now finishes within 1 m in 19/22 in-anchor and 13/14 cross-anchor trials,
against 17/30 and 12/19 for condition N -- and a success takes a median 54 steps
in-anchor where it took 139. The bottom row of the funnel went from 12 + 7 to
2 + 1.

**What the released targets cost: composition, not the model.** Aggregate in-situ
recall fell 0.53 to 0.29, but close-and-centred recall per target -- keyframes
with the object inside 3 m and inside 0.6 of the half-frame -- says the
detector did not change on the objects both benchmarks share: bowl 0.96 to 1.00,
plate 0.64 to 0.60, cracker box 0.23 to 0.29; pitcher 0.53 to 0.36 and soup
can 0.36 to 0.26 are the only real losses. The rest is scissors (0.02) and mug
(0.00), which condition N replaced with a bleach bottle at 0.54. On the six
shared targets, the 1 m score is 33.3% / 28.2% then and **50.0% / 34.1%** now.

**"Seen, never named" is two different things.** Split by whether the agent ever
gave the detector a proper look -- at least three close, centred keyframes and
still no detection is the detector's wall; anything less is the search never
having looked:

| | detector wall | exposure |
|---|---:|---:|
| N in / cross | 5 / 0 | 0 / 4 |
| now in / cross | 8 (scissors 4) / 8 (scissors 7) | 2 / 12 |

Of the released run's 30, sixteen are a wall and eleven of those are scissors.
On the shared targets, nine of thirteen are exposure: the object crossed the
frame at 2-4 m while the agent was in `goto_frontier`, where it now spends 58% /
64% of its steps against 44% / 48% in condition N, and where the detector's
recall is 0.28. The instrument calls that "in view"; the detector cannot.

**What is the same in both runs, and is therefore the problem.** Three things
survive the change of benchmark almost untouched:

* *Cross-anchor never gets a look.* Never-in-view is 15/48 then and 12/53 now;
  wrong commits are 85% of all commits then and 84% now, and the first commit is
  the stale prior in 38/48 and 26/43 episodes. And in **all 32** cross-anchor
  failures without a detection -- 12 never in view, 12 exposure, 8 detector wall
  -- the mapped container the object actually sat on, within 1.5 m of it in
  every case (a bed in 16), was selected by the search **0 times** and arrived
  at 0 times, across 4-21 surface selections per episode. The reason is in
  `container_prior`: proximity to the *stale* position, `exp(-d / 1.0)`, gives a
  surface 5.6 m away 0.4% of the peak before the same-room bonus, and a passing
  glance from up to 4 m then retires whatever belief is left.
* *In-anchor arrives and leaves.* The first commit is already within 1 m of the
  new position in 29/48 then and 23/44 now; of those, 7 and 5 ended more than 3 m
  away, 5 and 3 of them after `absence_abandon`. The agent stood at the right
  surface, the detector said nothing, and it walked off. With the tight ring at
  0.35-0.65 m and a 79-degree horizontal field, the frame is 0.3-0.5 m wide at
  the object and the object moved a median 0.7 m: the "informative frame" the
  absence sensor decides on usually cannot contain it.
* *The detector on small objects*, which every cheap lever has been measured
  against (`PERCEPTION_INVESTIGATION.md`) and which appearance re-identification
  would address for the 16 wall episodes, 11 of them scissors.

So the drop was the rule, the legs are fixed, and what remains is neither eyes
nor legs. It is where the camera is when it looks, and what the agent does when
a correct arrival meets a silent detector. `SR_PROPOSAL_CLOSE_LOOK.md` is the
proposal for both, and its result: the look before absence is a small, cheap
positive (ten re-detections, the median episode 55 steps shorter, SR +1 / +1);
the opportunistic look is a null, because in 28 of the 32 cross-anchor
failures the agent never came within 4 m of the object's surface at all. The
cross-anchor problem is the search order, not the look.

## The terminal approach, corrected

The viewpoint rings the terminal approach aims at -- `[0.8, 1.2, 1.5, 2.0]` --
were calibrated against HM3D ObjectNav, which scores the distance to a sampled
goal *viewpoint*. Standing on the innermost ring is a success there by
construction. This benchmark scores the distance to the *object* at 1 m, and that
ladder has no slack in it: measured over the 15 trials that committed to the
right object and still failed, the committed track was a median 0.07 m from
truth and the goal a median 0.82 m, and the episode was lost on a 0.37 m
shortfall between the agent and its own goal.

`dualmap_protocol_osg_tightring` adds three tighter rungs below 0.8 m and
measures them from the object's estimated surface rather than its centre
(`verification.ring_radius_extent_aware`). The extent term is worth only
0.11-0.13 m on a YCB object and is not where the trials come from; it is what
keeps a 0.35 m ring from landing inside a couch, whose track centre sits a metre
inside the furniture.

### What it did, over all 186 paired trials

Paired by trial id against `outputs/osg_dualmap_protocol` by
`scripts/compare_tightring.py`, which writes `outputs/osg_dualmap_tightring/TIGHTRING_AB.md`.

| | Baseline | Arm |
|---|---:|---:|
| median `goal_to_obj_m` (in-anchor / cross-anchor) | 0.80 / 0.80 | **0.48 / 0.48** |
| approaches that found no viewpoint at all | 118 | **39** |
| median steps | 327 | 294 |
| near-miss band, in-anchor | 7 | 3 |
| near-miss band, cross-anchor | 7 | **0** |

| Split | SR base | SR arm | gained | lost | p (sign test) |
|---|---:|---:|---:|---:|---:|
| Static | 50.6% | 50.6% | 8 | 8 | 1.000 |
| In-anchor | 33.3% | **44.4%** | 11 | 5 | 0.210 |
| Cross-anchor | 22.6% | **28.3%** | 8 | 5 | 0.581 |

**The SR gains are not statistically significant** and should not be reported as
if they were: 107 dynamic trials cannot resolve an effect this size, exactly as
`AB_RESULTS.md` says. What is unambiguous is the mechanism. The goal moved where
it was designed to move, the near-miss band -- the population the change targets
-- converted 14 to 3, and the extra rungs made the planner *more* likely to find
a viewpoint rather than less, which was the risk. Most individual flips in both
directions are episodes diverging to entirely different places, not marginal
misses.

### Where this leaves the comparison

| Split | Localised (ours / DM) | SR raw (ours / DM) | SR position-correct (ours / DM) |
|---|---:|---:|---:|
| Static | 49.4% / 50.6% | 50.6% / 67.1% | 48.1% / 50.6% |
| In-anchor | **48.1%** / 40.7% | 44.4% / 64.8% | **42.6%** / 40.7% |
| Cross-anchor | 28.3% / 28.3% | 28.3% / 30.2% | **26.4%** / 24.5% |

On the position-correct metric -- success credited only when the system actually
localised the object, computed for both by the same predicate -- **we now lead on
both dynamic splits.** On the perception-controlled subset (scissors and mug
removed from both, for the reasons in `PERCEPTION_INVESTIGATION.md`) the margin
is 47.6% against 33.3% in-anchor and 31.7% against 24.4% cross-anchor, and
cross-anchor raw SR passes DualMap as well, 34.1% against 29.3%.

Raw in-anchor remains well behind, and the reason is not the approach: it is that
in-anchor is the split a stale prior answers for free (74.1% for a do-nothing
agent) and DualMap's abstract map exploits exactly that.
