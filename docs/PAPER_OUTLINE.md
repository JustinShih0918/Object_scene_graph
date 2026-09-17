# Paper outline — Method section, modular

*Companion to `docs/METHOD_SECTION.md` (the drafted Method prose, §A–§E) and
`docs/DYNAMIC_EXPERIMENT_PROTOCOL.md` (the dynamic-scene experiment). This file
is the **map**: which chapter
says what, in which order, which figure carries it, and which file the detail
comes from. Nothing here is new content; it is a routing table for the writing.*

**How to use it.** Write each chapter against its own row. The "detail lives in"
column is where the sentences come from — most of them are already written in
`METHOD_SECTION.md`; the rest are in the doc named. The "evidence" column is the
number you are allowed to claim, and where it was measured. If a row's evidence
column says *not measured*, the chapter must not carry a claim.

---

## A. The chapter map

Method is §III. Seven chapters, in dataflow order, so the figure reads
left-to-right and top-to-bottom exactly as the chapters run.

| # | chapter | the one thing it claims | figure | detail lives in |
|---|---|---|---|---|
| III-A | Overview & problem setting | the map is **actively wrong**, not merely incomplete | **F1** | `ARCHITECTURE.md` §intro; `METHOD_SECTION.md` §A |
| III-B | Perception front end | open-vocab detection **+ a class-agnostic second channel**, fused into one detection set | **F2** | `METHOD_SECTION.md` §B.1; `PROPOSAL_FUSION.md`, `REGION_PROPOSAL_RESULT.md` |
| III-C | Object representation: dual-quadric ellipsoids | one projection serves association *and* visibility | **F2** (right half) | `METHOD_SECTION.md` §B.2; `objects/ellipsoid.py`, `objects/association.py`, `objects/optimization.py`, `objects/linking.py` |
| III-D | Hierarchical 3D scene graph | a containment **tree** (floor → room → container → object, geometric container gate) **plus a floor-level edge set** for stairs | **F3** | `METHOD_SECTION.md` §B.3–B.7; `ARCHITECTURE.md` "module map"; `graph/containers.py`, `mapping/stairs.py` |
| III-E | Presence belief | informative misses require an exposed expected pose; readings revise one bounded score | **F4** | `METHOD_SECTION.md` §C; `PRESENCE_BELIEF_DETAIL_NOTES.md` |
| III-F | Search posterior & unified selection | candidate gates precede a **shared** surface/frontier selection rule | **F5** | `METHOD_SECTION.md` §D; `SEARCH_POLICY_DETAIL_NOTES.md` |
| III-G | Multi-floor policy | residual surface belief is comparable **across storeys** | **F6** | `METHOD_SECTION.md` §E; `MULTI_FLOOR.md`, `CROSS_FLOOR_CLIMB.md` |

**Why this order and not the code's order.** The code's dataflow is
`perception → objects → mapping → graph → exploration → planning → verification`.
The paper swaps `mapping` in beside `graph` (III-D) because a reader does not
need the costmap until the hierarchy needs a floor to hang off, and the costmap
is not a contribution. Everything else is the code order.

**The joints between chapters** — write these sentences explicitly, they are
what makes a modular method read as one system:

- B→C: *"the fused detection set is the input to the ellipsoid layer; every
  detection carries a mask, and the mask, not the box, is what is back-projected."*
- C→D: the same projected conic `E = (µ, Σ)` is reused twice — say so at the end
  of III-C and point forward: association here (III-C), visibility there (III-E).
- D→E: the graph stores *where*; the belief stores *whether*. One sentence.
- E→F: presence removes a disproved object track from direct candidates;
  III-F then scores alternative mapped support surfaces with `b(x)`. The
  current `b(x)` equation does not multiply by `p_i`, so do not claim it does.
- F→G: each surface carries a stable storey key, so the residual score
  `\bar b(x)λ(x)d(x)` can be averaged within each storey. This connects the
  surface search to the floor decision; keep the current *mean* rule explicit.

The two-pass setup and the reload cap belong in Experiments; they establish
how the stale map is supplied, rather than how the agent updates it.

---

## B. Chapter-by-chapter: what to write where

### III-A Overview and problem setting

Write, in order: (1) ObjectNav in a **changed** world — pass 1 maps a static
layout, objects move, pass 2 starts from that stale map; (2) why a rebuild-every-
episode agent measures nothing; (3) the four mechanisms named once each
(presence, identity, absence-as-evidence, search posterior), each with a forward
pointer to its chapter; (4) **F1** walked through in one paragraph.

- Source: `docs/ARCHITECTURE.md` lines 1–16 has this argument already written in
  English; `METHOD_SECTION.md` has none of it (it starts at notation). This
  paragraph is the one genuinely missing piece of prose.
- Notation table: `METHOD_SECTION.zh.md` §1, ready to adapt. Trim it to the symbols
  that survive into the final text.
- Caveat to state once, here, and never again: `agent.navigation=navmesh` is
  privileged (it sees simulator geometry). Any SR/SPL from a navmesh arm is not
  comparable to a sensor-only one — `CLAUDE.md` "Two agent stacks"; the
  `authored_val` runs on disk are navmesh runs.

### III-B Perception front end

The chapter with the most new material relative to the current draft.

1. **Keyframe selection** — τ_d = 0.25 m, τ_θ = 30°, matched to the action
   space; keyframes ≈ 0.91–0.96 × steps. (`METHOD_SECTION.md` §B.1;
   `perception/keyframe.py`.)
2. **Named channel** — YOLOE-11L-seg, 47-class vocabulary + the query string,
   imgsz 1280. *Masks, not boxes*, and why: a box back-projects the wall behind
   the object into the same quadric. (`perception/detector.py`,
   `perception/vocabulary.py`.)
3. **Class-agnostic channel** — FastSAM-s, ≤64 regions/keyframe, area band
   [200, 0.02·|I|], MobileCLIP-S2 embedding, cosine vs. the query,
   τ_admit = 0.24, admitted at a fixed score 0.5, flagged `proposal_only`.
   (`perception/region_proposer.py`, `perception/feature_encoder.py`.)
4. **Fusion rule and why it is a separate tier** — a proposal has no calibrated
   detector score, so it cannot pass the detector-calibrated admission gates;
   it gets its own gate (n ≥ 4 proposal observations, mean-feature cosine ≥ 0.28)
   and always sorts *after* named tracks. (`METHOD_SECTION.md` §B.1, §D;
   `PROPOSAL_FUSION.md` "the proposal refusion".)

**Evidence you may claim** (all paired, all in the docs):

| claim | number | source |
|---|---|---|
| the stage fires on the set it is for | 0 → 4 solved of 20; target *named* 0/20 → 7/20 | `REGION_PROPOSAL_RESULT.md` |
| it does not flood the map | 173 admits / 7403 sub-τ keyframes = **2.3%** | same |
| full-benchmark effect | 57/107 → 59/107 (+5/−3); perception subset 0→5 | `PROPOSAL_FUSION_RESULT.md` |
| what it costs | goal commits 18→42, attempts 17→25, median best stop 5.44→4.76 m | `REGION_PROPOSAL_RESULT.md` |

Write the cost row. A stage that triples goal commits and buys +2 is an honest
result and reads as one; hiding the commits reads as a hidden one.

### III-C Object representation

`METHOD_SECTION.md` §B.2 is the
chapter, and it is deliberately organised as *what VOOM published* versus *what
we changed*. VOOM's parts — the dual quadric, the projection to the dual conic,
the NWD association, the Bures-surrogate objective — are cited and stated
inline, with **no display equation**. The two display equations are ours:

- **The refine acceptance test.** The reprojection objective constrains depth
  only through parallax, and an ObjectNav agent walks *at* an object rather than
  around it, so a refine can slide the centre metres along the ray at low 2-D
  error. Guards: finite parameters, cost ≤ ½‖r₀‖² + 10³, and ‖Δt‖ ≤ 0.5 m.
  This is the paragraph a reviewer will care about, not the objective.
- **The linking rule** and `c(o_i)` as the component mean, including the
  **co-observation gate** — without it a moved object merges with its own ghost
  and the centre lands midway between where it *was* and where it *is*, a
  position no observation can ever disprove. That sentence is what ties III-C to
  III-E; keep it.

Also note the stance the chapter takes on large objects: association is **not**
modified for them. A sofa exceeding the field of view produces two mask ellipses
of genuinely different parts, and they are correctly judged dissimilar; the fix
is downstream (linking here, extent-aware container merging in §2.4), not a
looser metric. Evidence: a flat merge radius leaves 00829 with 16 `bed` nodes in
a house with one bed; extent-aware merging gives 7, the rest being mislabelling.

### III-D Hierarchical scene graph

Order: costmap (0.05 m, {−1,0,100}, band `[y+0.15, y+1.5)`) → floor estimation
(the 1.8 m registration gap and why it sits between a 1.1 m landing and a 2.5 m
storey) → room segmentation (erosion 6, 2.0 m door, ≥ 60 cells) → containers →
serialization.

- The container gate is the chapter's claim: **category ∧ geometry**, and the
  failure it prevents (a false-positive "table" becoming a permanent anchor).
  `METHOD_SECTION.md` §B.6; `graph/containers.py`; `FUSED_PIPELINE.md`
  "the two container fixes are complementary".
- Floor ids are **stable** — new storeys take new ids, existing ids never
  renumber, because per-floor costmaps, room ids and frontier blacklists key on
  them. One sentence, it prevents a whole class of reader question.
- **Stairs are an edge, not a node** (`METHOD_SECTION.md` §B.7). The
  hierarchy is a containment tree and a staircase is contained in nothing — it
  *joins* two floors — so it is a `StairEdge (φ_from, φ_to, x_entry, x_exit,
  t, n)` on the floor layer, written only once the agent has actually traversed,
  and persisted as `connectivity` in the snapshot. Its **evidence** sits in three
  places that never enter the graph: the costmap's `stair_mask` and up/down hit
  accumulators; the `Flight` derived on demand from the height layer (a connected
  run of cells strictly between two storeys, kept if it spans ≥ 1.0 m — a table
  is a plateau and spans nothing; its lowest tread *is* the foot); and a `stairs`
  track in the object layer. Adoption order flight > `stairs` track > portal is
  measured, not preferred: the YOLOE `stairs` class fired in only **15% of
  multi-floor episodes** (12 tracks / 100), good when it fired (median score
  0.60, median 26 observations) but far too sparse to gate a floor transition.
  Geometry leads, semantics confirm. Define it here; §E / III-G only *uses* it.

### III-E Presence belief

`METHOD_SECTION.md` §C is the concise main-paper version. Its two subsections
and two equations cover the visibility gate and bounded evidence update; the
previous longer draft is in `PRESENCE_BELIEF_DETAIL_NOTES.md`. Figure **F4** is
`figures/presence_belief.svg`: it shows the three visibility outcomes above,
then the alternative arrival observers and a worked bounded update below.
Keep these distinctions prominent:

- The **one-sided occlusion gate** (§C.1) is the mechanism. Removed
  object → measured depth lands *beyond* the band → must update. Occluded object
  → depth lands *nearer* → must not. One signed comparison. Give it its own
  paragraph and its own panel in **F4**. In the implementation, $E_i$ gates
  misses only: a sighting updates even if the expectation test fails.
- The **asymmetric clamp** (§C.2) is a deliberate bound on evidence, so the
  logistic score is not described as a fully calibrated posterior. The
  [−6.0, +3.0] values and seven-miss arithmetic belong in the implementation
  table or appendix.

Candidate ranking and identity rejection belong at the entrance to §D,
before choosing a surface or frontier. Report the 251 commits to one wrong
track, the $\rho_i\ge2$ gate, and the 170-pair ranking comparison in
ablations rather than in the presence filter definition.

### III-F Search posterior and unified selection

`METHOD_SECTION.md` §D should first gate mapped target tracks by presence and
identity. It then keeps three points: geometric frontiers and mapped support
surfaces are alternative goals; their scores compete after
geodesic cost, with β setting the frontier tradeoff; and an unsuccessful
inspection persistently reduces a surface's value through λ(x). Surface
affordance, affinity, and distance from the last believed target position
explain the semantic prior. Peak-normalise that prior **before** inspection
decay, so a revisited surface does not regain its former score. Put extraction
thresholds, measured ranking/drive effects, parameter values, and diagnostics
in `SEARCH_POLICY_DETAIL_NOTES.md` or Results.

The current cross-anchor episodes have not exercised the frontier and surface
branches together. The shared index is a design claim until the proposed β
sweep on episodes with both branches live measures the tradeoff; avoid an
empirical arbitration claim without that test.

### III-G Multi-floor policy

`METHOD_SECTION.md` §E lifts the residual surface score to the **mean** over
eligible surfaces on each storey, then briefly states the margin, untested
anchor hold, failed-arrival evidence, and timing gate. An accepted request
uses a height-derived flight first, a detected `stairs` track second, and a
geometric portal last. The graph records the stair edge only after traversing
it. Keep thresholds and failure diagnostics in the supporting material.

Supporting material: `MULTI_FLOOR.md` (why the two floor signals agree, the
benchmark's floor distribution), `CROSS_FLOOR_CLIMB.md`, and — for the failure
modes you should disclose — `CROSS_ANCHOR_STATUS.md` §3C.

### Experiments: dynamic-scene prior maps

`DYNAMIC_EXPERIMENT_PROTOCOL.md` now holds the static mapping pass, relocation,
dynamic search pass, reload cap, separable occupancy arm, and map coverage
audit. Explain the audit as part of the protocol: a scene whose prior map does
not cover the authored target positions or usable stairs is held back so an
unwinnable map is not scored as a recovery failure. Use F7 here.

---

## C. Figure plan

Six Method figures plus one Experiments figure. Their scripts and render
requirements are tracked below.

| id | figure | status | command |
|---|---|---|---|
| **F1** | Architecture of the framework | **renders today**, needs an edit | `python scripts/render_architecture.py --out outputs/figures/F1_architecture.pdf` |
| **F2** | Perception + ellipsoid construction | **to write** (`render_perception_pipeline.py`) | — |
| **F3** | The built scene graph (2-D and 3-D) | **renders today** | `python scripts/render_scene_graph.py --scene 00848-ziup5kvtCCR` / `render_scene_graph_3d.py` |
| **F4** | Visibility-gated presence revision | **renders today** | `python scripts/render_presence_belief.py --out docs/figures/presence_belief.svg` |
| **F5** | Route + presence mass (Fig-5 analogue) | **renders today**, wants one re-run | `python scripts/render_route_presence.py --run … --maps … --out …` |
| **F6** | Storey belief mass over time | **renders today** (right panel of F5) | same script |
| **F7** | Two-pass protocol / prior-map coverage | **renders today** | `python scripts/plot_cross_anchor_map.py --maps … --scene … --run … --out …` |

### F1 — Architecture (III-A)

`scripts/render_architecture.py` already draws the two-band layout (scene-graph
builder on top, surface search underneath) with real insets. **Two edits before
it is the paper's F1:**

1. The perception row is one box, `Open-Vocabulary Detection`. It must become
   two parallel channels feeding one fusion node — `YOLOE (named)` and
   `FastSAM → MobileCLIP (class-agnostic)` → `Detection set` — or III-B has no
   figure support.
2. There is no `Multi-view refinement` box between `Association + Ellipsoid
   Fitting` and `Object Nodes`. Add it; it is a titled subsection of the method
   (§2.2) with no presence in the figure.

Keep the band structure. It is the part that maps onto the chapter order.

### F2 — Perception and ellipsoid construction (III-B, III-C)

New script. Three panels, left to right, matching §2.1 → §2.2 (initialisation)
→ §2.2 (multi-view refinement):

- **(a) Two channels.** One keyframe RGB with YOLOE masks drawn in one colour
  and FastSAM regions that passed τ_admit in another, each labelled with its
  score (detector score vs. CLIP cosine). Source frames:
  `outputs/mf5_pass2_p1500/<scene>/keyframes/` and `viz/debug/*.mp4`; crops
  already used by `render_architecture.py` come from the saved map's tracks.
- **(b) Initialisation.** The same frame's depth, the 2-D ellipse from the
  mask's second moments, and the back-projected 3-D ellipsoid.
- **(c) Multi-view optimisation.** ≥3 camera frusta around one track, the
  per-view observed conics, and the refined ellipsoid. All of this is in the
  saved map: each track record carries its observations. (`graph/map_store.py`
  `_track_from_record`, `objects/ellipsoid.py`.)

This is the figure that shows what was added on top of YOLOE, which is the
user-facing point of III-B. It is worth the script.

### F3 — The scene graph (III-D)

Already publication-quality. Verified render:
`00848-ziup5kvtCCR — 5 rooms · 70 support surfaces · 343 objects (15 on a surface)`,
with room fills, container ellipses, on-surface vs. free objects, and six track
crops. `render_scene_graph_3d.py` gives the lifted-ellipsoid 3-D view (45
ellipsoids, 4 rooms) if you want the Fig-2-inset look.

Use the 2-D for the method chapter and the 3-D as the F1 inset, not both at
full size.

### F4 — Presence belief (III-E)

New script, and the most valuable new figure in the set, because it shows the
mechanism no baseline has. Two panels:

- **(a) The three channels, as geometry.** One object, three frames: seen
  (Z=1,E=1), removed → depth beyond the band (Z=0,E=1), occluded → depth nearer
  than the band (E=0). Draw the expected band `[z_c − e − δ, …]` on the depth
  profile and the resulting ΔL under each. Renderable from the saved
  `verify_debug/` crops plus `presence_events`.
- **(b) The clamp, as a trace.** L_i over steps for one track that is believed,
  disproved on arrival, and then re-detected — showing there is no absorbing
  state. `presence_events` in any `authored_val` run has this
  (96–145 events/episode).

### F5 / F6 — Route and storey mass (III-F, III-G)

`scripts/render_route_presence.py` (new, written, working). Left panel: the
storey's costmap, the route, the robot at start, every track drawn as a disc
whose radius is its belief, the committed goal and the true target. Right panel:
the logged `floor_mass` per storey over the episode with the chosen storey
starred. The current experiment uses a **mean** residual surface score;
relabel the plot axis in `render_route_presence.py` before using this panel.

Two renders done, and they say something useful about the data:

- `outputs/authored_val/00821-eF36g7L6Z9M/dynamic`, episode
  `…__in_anchor_01__50002__s0` (**success**, 348 steps, SPL 0.40): the map, the
  two `tin can` tracks (p = 0.82 believed, the other disproved), the commit and
  the true target all render. Single storey, so the mass panel is flat.
- `outputs/mf5_pass2_p1500/00808-y9hTuugGdiq`, episode `…__cross_anchor_01__50005__s0`
  (failure, 1000 steps): the mass panel is the interesting one — storey 0 holds
  5.7 of the mass until ≈ step 300, both storeys then collapse to ≈ 0, and the
  agent spends the remaining 700 steps with nothing to believe. That is a real
  diagnostic and belongs in the paper *if* the story is the failure analysis.

⚠ **What blocks the publication version.** No run on disk has **both** a dense
trajectory and presence logs. `eval.behaviour_log=true` (which writes
`step_trace`, the per-step pose) was set on the ASCENT mapping passes
(`outputs/prior1500/*`, `outputs/mf5_stage1/*`) and on none of the OSG runs; the
OSG runs (`authored_val/*`, `mf5_pass2_*`) have `presence_events`,
`search_log_events` and `frontier_select_log` but no pose trace. So F5 currently
draws a **decision polyline** (agent position at each frontier selection, the
frontier chosen, the commit, the final pose) and the legend says so. To get the
real path, re-run one good episode with the flag — see §E.

### F7 — Prior map and the two-pass protocol (Experiments)

`scripts/plot_cross_anchor_map.py` / `eval/obstacle_map_viz.py` draws the stored
per-storey occupancy with the episode's target marked, stair cells in orange and
the flight endpoints labelled. Runs write it automatically when
`eval.obstacle_map_png` is on. This is the figure that makes the audit gate
legible: it is how the "lower storey's northern room was never mapped" finding
was made (`CROSS_ANCHOR_OBSTACLE_MAP.md` §"what the map picture shows").

### Two more worth proposing

- **F8 — Qualitative strip, ASCENT-Fig-5 style.** Three scenes side by side,
  each: top-down map + route + target inset photo. This is the figure reviewers
  expect and the one the reference paper uses; F5 is the *mechanism* version of
  it. Same script, `--no-mass-panel`, three panels. Cheap once F5's re-run
  exists.
- **F9 — The ablation ladder as a bar chart.** presence off → on, identity off →
  on, proposal off → on, search posterior off → on, on the same trials.
  `DYNAMIC_SCENES.md` §"Ablation ladder" defines it; the numbers are scattered
  across `PROPOSAL_FUSION_RESULT.md`, `PERCEPTION_RESULT_FINAL.md`,
  `SR_PROPOSAL_CLOSE_LOOK.md`. Assembling them into one chart is a half-day and
  is probably the highest-value figure in the results section, not the method.

---

## D. Tables

| id | table | source |
|---|---|---|
| T1 | Symbols | `METHOD_SECTION.zh.md` §1; adapt for English paper |
| T2 | Calibrated constants, by module | `tests/unit/golden/config_snapshot.json` (496 keys, flattened, already the canonical list) |
| T3 | Cross-sensor readings: (r, q, ΔL) per sensor | `PRESENCE_BELIEF_DETAIL_NOTES.md` §C.4 |
| T4 | The two halves of the index compared | `SEARCH_POLICY_DETAIL_NOTES.md` §D.4 (optional implementation table) |
| T5 | Benchmark protocol | `METHODOLOGY.md` §1 |

T2 is worth doing properly: the repo's claim is that every default carries its
measurement, and the golden snapshot is that claim in machine-checkable form.
A table with a `measured by` column per constant is a credibility argument no
competing paper can make cheaply.

---

## E. What has to be re-run before the figures are final

Two jobs, both small, both needed only for figure quality, not for any claim.

1. **One OSG episode with the pose trace, for F5/F8.**
   ```bash
   python scripts/run_eval.py +experiment=mf5_osg_on_ascent_map \
     ycb.scenes=[00821-eF36g7L6Z9M] eval.num_episodes=1 \
     eval.behaviour_log=true eval.save_viz=true \
     output_dir=outputs/figures_run
   ```
   Pick a **success**; `00821-eF36g7L6Z9M__in_anchor_01__50002__s0` is the one
   that already renders. Note the `authored_val` runs used
   `agent.navigation=navmesh`, so if the figure's caption quotes SR/SPL it must
   say so.

2. **The β sweep, for III-F's open item.** β ∈ {0.3, 1.0, 3.0} on a set where
   both halves are live, reporting `search_surface`, `search_surface_outbid` and
   `frontier_util`. Without it the "unified index" claim is a design claim, not
   a measured one — which is fine, as long as the text says which.

Everything else — F1, F2, F3, F4, F6, F7 — is renderable from the logs and maps
already on disk.

---

## F. Reading order for the drafting session

1. `docs/ARCHITECTURE.md` — the argument, in English, best single read.
2. `docs/METHOD_SECTION.md` — the drafted §§B–E prose. Most of III-C through
   III-G is already written here.
3. `docs/DYNAMIC_EXPERIMENT_PROTOCOL.md` — the stale-prior setup for the
   dynamic-scene Experiments section; `docs/METHODOLOGY.md` covers baseline
   benchmarking and paired-run diagnostics.
4. Per chapter, the doc named in the table in §A.
5. `docs/AB_RESULTS.md` only when you need to source a specific number — it is
   4600 lines of chronological log, indexed by stage (S0–S76), not a reference.
