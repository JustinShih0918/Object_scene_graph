# How to run OSG's ObjectNav pipeline

The default configuration is the S71 arm: ASCENT's control flow
(`src/ascentnav/`) on ASCENT's served perception models, sensor-only, on the
frozen PointNav mover. On the 100-episode HM3D v1 split `scenes20_ep0to4` it
scores **63.0% SR / 0.36 SPL**; native ASCENT on the same episodes scores
65.0% / 0.36 (`docs/AB_RESULTS.md`, S71). Everything below runs inside the
`nav` container (see `README.md` → Quick start for building it).

## 1. One-time setup

```bash
python scripts/download_data.py --username <TOKEN_ID> --password <TOKEN_SECRET> --uids hm3d_val_v0.2
python scripts/download_data.py --episodes-only        # ObjectNav episodes
python scripts/download_weights.py --pointnav --rednet # the mover and the stair segmenter
bash scripts/fetch_ascent_weights.sh                   # BLIP-2, MobileSAM, GroundingDINO, RAM++, D-FINE
docker exec docker-ollama-1 ollama pull qwen2.5:7b     # the planner LLM
```

The five ASCENT models run in the `ascent` conda env
(`/workspace/.conda-envs/ascent`; built by `docker/Dockerfile.ascent`), not in
habitat's — BLIP-2's `lavis` and habitat-sim cannot share an interpreter.

## 2. Start the model servers

```bash
bash scripts/serve_perception.sh          # tmux session `osg_models`, five windows
bash scripts/serve_perception.sh --stop
```

| window | port | role in the pipeline |
|---|---|---|
| `blip2itm` | 13182 | value-map cosine every step; the commit gate latches at ≥ 0.15 |
| `sam` | 13183 | one MobileSAM mask per D-FINE box |
| `gdino` | 13184 | GroundingDINO `"stair ."` ≥ 0.60, ANDed with RedNet for the stair mask |
| `ram` | 13185 | RAM++ scene tags for the LLM prompt, once per step |
| `dfine` | 13186 | D-FINE COCO detector, target detections at conf ≥ 0.8 |

Logs land in `relative_work/ascent/debug/vlm_logs/<window>.log`. The GPU
budget with all five plus habitat is ~12 GB.

Only run **one evaluation at a time** against these servers. They hold
per-image state on the model object, so concurrent requests used to race and
return HTTP 500; the servers now serialise model calls behind a lock and the
client retries once, but a second eval still halves everyone's throughput.

The run **refuses to start** if any server it needs is down
(`PerceptionUnavailable`, raised by `probe_served_models` before Habitat
loads), and a server that stops answering mid-run raises the same error
instead of returning a neutral value. That is deliberate: under ASCENT's gate a
silent 0 from BLIP-2 is an agent that never STOPs, and a 0% run would look
like a bad algorithm.

## 3. Run an evaluation

```bash
python scripts/run_eval.py                                   # the default: 100 episodes, ~3.5 h
python scripts/run_eval.py output_dir=outputs/my_run         # name the run
python scripts/run_eval.py eval.behaviour_log=true           # + per-step trace for the comparison script
python scripts/run_eval.py eval.num_episodes=3               # smoke
python scripts/run_eval.py +experiment=final_sensor          # the full v1 val split (2000 episodes)
```

`+experiment=ascentnav` names the default explicitly and composes to the
identical config (pinned by `tests/unit/golden/experiment_fingerprints.json`).

Outputs, under `output_dir`:

- `episodes.jsonl` — one record per episode: `success`, `spl`, `steps`,
  `distance_to_goal`, `agent_stats` (LLM calls, gate latches, climb attempts,
  give-ups…), `state_log`, `giveup_log`, `approach_stop_reason`, and with
  `eval.behaviour_log=true` the full `step_trace`.
- `summary.json` — the config used, the algorithm field inventory, SR / SPL /
  mean steps.
- `timing.csv` — per-episode wall time.

## 4. Compare against ASCENT

`relative_work/ascent/debug/behaviour_100/` holds native ASCENT's recorded
100-episode run on the same split (65.0%). Pair any OSG run against it:

```bash
python scripts/compare_ascent_osg.py relative_work/ascent/debug/behaviour_100 outputs/my_run
```

The report gives paired SR by floor class and category, the episodes each
side wins, and — scored against the dataset's own object positions — the
trace metrics that carried the S71 diagnosis:

| metric | what it tells you | ASCENT | S71 |
|---|---|---|---|
| SAW episodes | the target was within 3 m / ±40° at some step | 90 | 87 |
| never-saw → STOP / timeout | the search was truncated before the target was in frame | 8 / 1 | 9 / 3 |
| climb-mode steps, same-floor episodes | budget lost to staircases on episodes that need none | 5.2% | 5.8% |
| earliest STOP, STOPs before 33 | premature irreversible stops | 23, 3 | 23, 3 |
| P(success \| committed) | conversion once the agent walks at a detection | 0.711 | 0.693 |

A change that improves SR should move one of these; a change that moves SR
without moving any of them is noise at n = 100 (±3 points).

## 5. MP3D ObjectNav

A second benchmark, not a variant split: 21 goal categories, 11 val scenes,
2195 val episodes. The embodiment is identical to HM3D's, so the agent carries
over unchanged; the goal set does not.

```bash
python scripts/download_mp3d.py --episodes-only     # public, 173 MB
python scripts/download_mp3d.py --example-scene     # free single scene, no licence
python scripts/download_mp3d.py --verify            # what is present, what is missing
```

The 11 val scenes are **license-gated**: sign the Matterport Terms of Use at
https://niessner.github.io/Matterport/, they email you `download_mp.py`, then

```bash
python scripts/download_mp3d.py --mp-script /path/to/download_mp.py
```

which runs its `--task habitat` subset (~15 GB, not the 1.3 TB raw release)
and expects the result unpacked to `data/scene_datasets/mp3d/<scene>/<scene>.glb`.

There is no MP3D downloader bundled anywhere in the tree — not under
`relative_work/ascent` (whose README delegates to habitat-lab's DATASETS.md),
and not in either conda env's `habitat_sim.utils.datasets_download`, which
offers only `mp3d_example_scene`. The Matterport ToU is the only route to the
val scenes. The episode layout here (`.../objectnav/mp3d/v1/{split}/`) matches
habitat's DATASETS.md, this repo's HM3D tree, and ASCENT's own README, so
native ASCENT's `eval_ascent_mp3d.yaml` reads the same files.

**The detector is the one thing that does not transfer.** Closed-set COCO can
name 8 of the 21 categories (chair, table, sofa, bed, plant, sink, toilet,
tv_monitor). The other 13 — picture, cabinet, cushion, chest_of_drawers, stool,
towel, shower, bathtub, counter, fireplace, gym_equipment, seating, clothes —
are 1115 of the 2195 val episodes, and D-FINE scores a structural zero on every
one. So:

| command | detector | scores |
|---|---|---|
| `+experiment=ascentnav_mp3d` | YOLOE, open-vocabulary | all 2195 episodes |
| `+experiment=ascentnav_mp3d_coco8` | D-FINE, as on HM3D | the 1080 COCO-nameable episodes — like-for-like with the HM3D number, and **not** an MP3D ObjectNav result |
| `eval=mp3d_example_smoke` | either | 3 episodes on the free scene, no licence needed |

The ASCENT priors cover the 6 HM3D categories only; the planner degrades to
empty priors for the rest, which given S72 (the LLM touches 1.7% of frontier
decisions) costs close to nothing.

## 6. Variants worth running

| command | what it tests |
|---|---|
| `+experiment=ascentnav_union_stairs` | OSG's looser stair sensing (RedNet union, no passive entry, lip test) in the same control flow |
| `agent.stair_disable_burns_map=true` | run the reference's dead stair-burn branch (F3) so a failed staircase is not retried |
| `exploration.llm_multi_floor=true` | the reference's multi-floor prompt, which its own run never reached (F2) |
| `agent.downstair_detector=lip` | OSG's "missing floor" down-stair trigger instead of the reference's mirrored depth |
| `verification.enabled=true` | add OSG's forced-choice VLM verifier on top (needs `NVIDIA_API_KEY`) |
| `eval=scenes20_crossfloor` | only the cross-floor episodes |

Pre-S71 presets (`matched_*`, `ycb_*`, `ascent_sensor*`, `full_v1_navmesh`)
still compose to exactly what they were measured with: each pulls
`configs/legacy_defaults.yaml` first, which restores the old base (YOLOE-11s,
NIM, the VLM verifier, the text-LLM frontier scorer).

## 7. Tests

```bash
pytest tests/unit -q                       # ~20 s, no GPU or data
python tests/unit/test_config_snapshot.py  # regenerate the config golden after a deliberate default change
```

The transcription is pinned by `tests/unit/test_ascentnav_{navigate,explore,
planner,perception,stairs,geometry}.py`; each test names the reference line it
was transcribed from.

## 8. Where things are

| | |
|---|---|
| the agent | `src/ascentnav/agent.py` (dispatch, `_navigate`, `_explore`), `stairs.py`, `planner.py`, `perception.py` |
| the maps | `src/ascentnav/mapping/` (vendored from ASCENT) |
| served-model clients | `src/osg/perception/ascent_models.py`, `detector.py` (`DFineDetector`), `image_text.py` (`Blip2ItmScorer`) |
| the mover | `src/osg/planning/pointnav_driver.py` (weights bit-identical to ASCENT's) |
| config | `configs/config.yaml` → `agent/s71`, `exploration/s71`, `verification/s71`, `detector/dfine`, `llm/qwen_local`, `scene_graph/place365`, `eval/scenes20_ep0to4` |
| results log | `docs/AB_RESULTS.md` (S71 is the current record), `src/ascentnav/README.md` (fidelity notes F1–F14) |
| methodology | `docs/METHODOLOGY.md` — protocol, evidence rules, differences from native ASCENT |
