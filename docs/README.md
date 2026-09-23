# Documentation index

52 files, and they are not all the same kind of thing. Read the four in **Start here** and
ignore the rest until you need them; everything below that line is either a frozen
measurement or the reasoning behind one decision.

## Start here

| doc | what it is |
|---|---|
| [PAPER.md](PAPER.md) | The ICRA 2027 submission in full, plus a paper→code map and a number→run provenance table. **The system is MD-SG in the paper and OSG in the code.** |
| [ARCHITECTURE.md](ARCHITECTURE.md) | The OSG pipeline module by module, and why each constant is what it is. |
| [SETUP.md](SETUP.md) | Clone with submodules, `docker/.env`, weights, the two conda envs. |
| [USAGE.md](USAGE.md) | How to run the pipeline, and what the default arm scores. |

Also load-bearing: [`METHODOLOGY.md`](METHODOLOGY.md) (how this repo measures ObjectNav, and
every difference from native ASCENT), [`UNIFIED_PIPELINE.md`](UNIFIED_PIPELINE.md) (one agent
across three datasets), [`FIGURES.md`](FIGURES.md) (every paper figure, its caption and its
renderer), [`ZERO_SHOT_AUDIT.md`](ZERO_SHOT_AUDIT.md) (every piece of hand-authored
per-benchmark knowledge in the pipeline).

## The logs — measurements, in order

Append-only. Large because they are complete, not because they are untidy.

| doc | |
|---|---|
| [AB_RESULTS.md](AB_RESULTS.md) | 236 KB. Every staged change S0–S76, its measured effect, the decision. The results log CLAUDE.md names. |
| [DYNAMIC_SCENES.md](DYNAMIC_SCENES.md) | 161 KB. The dynamic-scene design and its experiment log. |
| [CROSS_ANCHOR_OBSTACLE_MAP.md](CROSS_ANCHOR_OBSTACLE_MAP.md) | The chronological log of navigating a moved layout over ASCENT's obstacle map — every measurement and every wrong turn. Carries an internal retraction: everything before the transpose fix was measured on a map 8–9 m from the truth. |
| [CROSS_ANCHOR_STATUS.md](CROSS_ANCHOR_STATUS.md) | The summary of the above: where it stands, what was found by layer, what is left. **Read this one first.** |
| [OVERNIGHT_SR_ITERATION.md](OVERNIGHT_SR_ITERATION.md) | Written as it happened, on branch `experiment/sr-70-50`. |
| [DUALMAP_OFFICIAL_RERUN.md](DUALMAP_OFFICIAL_RERUN.md) | DualMap rerun on its own released benchmark; the source of the 30.2% cross-anchor figure the paper compares against. |
| [CROSS_ANCHOR_REPRODUCTION.md](CROSS_ANCHOR_REPRODUCTION.md) | 2026-09-20: why the paper's cross-anchor headline no longer re-derives, what was ruled out, and what is left. **Read before quoting Table I.** |

## Running on a real robot

| doc | what it is |
|---|---|
| [ROS2.md](ROS2.md) | The ROS 2 / Nav2 layer for the Stretch 3: why two interpreters, the frame conventions, who owns the base, the manual floor switch, the two-run protocol, and how to check all of it with no robot. |
| [THOR.md](THOR.md) | Where that layer runs: a Jetson AGX Thor beside the robot, two containers, and every knob that can only be settled on the device. |
| [DEMO_FLOW.md](DEMO_FLOW.md) | The two-floor moved-object demo as a checklist of commands: bring-up, the two passes, the carry, the one-line floor switch. Just the order; the reasons are in the two rows above. |

## Design notes

[MULTI_FLOOR.md](MULTI_FLOOR.md) · [CROSS_FLOOR_CLIMB.md](CROSS_FLOOR_CLIMB.md) ·
[FUSED_PIPELINE.md](FUSED_PIPELINE.md) · [PROPOSAL_FUSION.md](PROPOSAL_FUSION.md) ·
[DYNAMIC_EXPERIMENT_PROTOCOL.md](DYNAMIC_EXPERIMENT_PROTOCOL.md) ·
[ABLATION_FAILURE_MODES.md](ABLATION_FAILURE_MODES.md) ·
[APPEARANCE_REID_AND_CANDIDATE_BUDGET_PLAN.md](APPEARANCE_REID_AND_CANDIDATE_BUDGET_PLAN.md) ·
[SR_PROPOSAL_CLOSE_LOOK.md](SR_PROPOSAL_CLOSE_LOOK.md)

## Frozen results

One table each, from one run. **Most of the runs they name are gone from `outputs/`** — the
table is the record, and re-deriving it means re-running the arm.

[SENSOR_ONLY_RESULT.md](SENSOR_ONLY_RESULT.md) ·
[WHY_THE_SENSOR_ARM_LOSES.md](WHY_THE_SENSOR_ARM_LOSES.md) ·
[CREEP_FIX_RESULTS.md](CREEP_FIX_RESULTS.md) ·
[REACHABILITY_CEILING.md](REACHABILITY_CEILING.md) ·
[UNREACHABLE_SET_EXPERIMENT.md](UNREACHABLE_SET_EXPERIMENT.md) ·
[RELEASED_MAPS_RESULT.md](RELEASED_MAPS_RESULT.md) ·
[RESIDUE_AFTER_RELEASED_MAPS.md](RESIDUE_AFTER_RELEASED_MAPS.md) ·
[FUSION_RESULTS.md](FUSION_RESULTS.md) ·
[FUSION_RESULTS_00808.md](FUSION_RESULTS_00808.md) ·
[PROPOSAL_FUSION_CLS_RESULT.md](PROPOSAL_FUSION_CLS_RESULT.md) ·
[PROPOSAL_COMMIT_BAR.md](PROPOSAL_COMMIT_BAR.md) ·
[REGION_PROPOSAL_RESULT.md](REGION_PROPOSAL_RESULT.md) ·
[REGION_FALSE_ADMISSION.md](REGION_FALSE_ADMISSION.md) ·
[PERCEPTION_RESULT_FINAL.md](PERCEPTION_RESULT_FINAL.md) ·
[PERCEPTION_GAP.md](PERCEPTION_GAP.md) ·
[PERCEPTION_INVESTIGATION.md](PERCEPTION_INVESTIGATION.md) ·
[FEATURE_MEMORY_PROBE.md](FEATURE_MEMORY_PROBE.md) ·
[FEATURE_MEMORY_ONLINE.md](FEATURE_MEMORY_ONLINE.md) ·
[ASSET_SUBSTITUTION_EVIDENCE.md](ASSET_SUBSTITUTION_EVIDENCE.md)

## Superseded — kept, and labelled at the top of each file

| doc | superseded by |
|---|---|
| [INVESTIGATION.md](INVESTIGATION.md) | S71. Historical (2026-07); names a stack that no longer exists. |
| [MULTI_FLOOR.md](MULTI_FLOOR.md) | Partly. Its `frontier_cost_free_cell=true` recommendation was reversed with a measurement. |
| [PROPOSAL_FUSION_RESULT.md](PROPOSAL_FUSION_RESULT.md) | `PROPOSAL_FUSION_CLS_RESULT.md` — 59/107 against 62/107. |
| [DUALMAP_SWAP.md](DUALMAP_SWAP.md) | Nothing. Paused before any trial ran. |
| [METHOD_SECTION.zh.md](METHOD_SECTION.zh.md) | `PAPER.md`. Kept only for its Chinese review commentary. |

## Chinese-language summaries

[PROGRESS_zh.md](PROGRESS_zh.md) · [REPORT_zh.md](REPORT_zh.md) ·
[METHOD_SECTION.zh.md](METHOD_SECTION.zh.md)
