# The ICRA 2027 video

Three minutes, 20 MB, one story: a map that is actively wrong, a pipeline that notices, and a
robot that recovers the object anyway. The cut leads with simulator footage — the agent moving
through HM3D in first person with its detections on screen — and uses the paper's figures only
as short inserts. `scripts/video/shotlist.json` is the storyboard as data; the table below is
printed from it (`python scripts/video/build_video.py --print-storyboard`), so the doc and the
cut cannot drift apart.

```bash
python scripts/video/make_assets.py            # every shot -> outputs/video/segments/*.mp4 (~10 min)
python scripts/video/build_video.py            # -> outputs/video/icra2027_rough.mp4, checked against 180 s / 20 MB
python scripts/video/build_video.py --narration voice.wav      # once the voiceover is recorded
python scripts/video/make_assets.py --frame hero 12.0          # one PNG at t=12 s of a shot, to look at it
```

Everything under `outputs/video/` is generated. The robot footage goes in
`outputs/video/robot/` (see *Robot shot list*); `build_video.py` picks it up when the files
exist and shows a labelled placeholder slate otherwise, so the rough cut can be watched today.

## Storyboard

| t | shot | beat | on screen | narration |
|---|---|---|---|---|
| 0:00 (+6 s) | `title` | Title over the hero clip's detection frame | MD-SG — title, venue | MD-SG: object navigation with a scene graph that lets a map be wrong, notice, and recover. |
| 0:06 (+10 s) | `problem_clip` | Problem, shown: the agent walks to the remembered table, the verifier rejects, the map was wrong | first-person RGB + detections, costmap inset, verifier image inset | The map says the cracker box is on this table. It was, before someone moved it. Pass one mapped the building; the object was relocated; pass two starts confidently wrong. |
| 0:16 (+4 s) | `R1` | ROBOT SLOT R1 (optional): the bottle being moved by hand | [ROBOT FOOTAGE R1] |  |
| 0:20 (+5 s) | `problem_still` | The three-step protocol, on the real floor plates | layout_conditions panel (d): cracker box, table → kitchen island, 2.71 m of it vertical | That is the benchmark: map, move, search again with the old map, within a floor or across floors. |
| 0:25 (+10 s) | `contributions` | Contributions over the pipeline figure | three bullets + full_pipeline.png | We formulate dynamic multi-floor object navigation, build an online open-vocabulary scene-graph pipeline that revises its object beliefs, and release the first benchmark with controlled same-floor and cross-floor relocation. |
| 0:35 (+13 s) | `perception_clip` | Perception to scene graph, on footage | detections + the scene-graph tree sliding in | Every detection is back-projected into an ellipsoid track and filed under floor, room and container. Stair edges join two storeys only after a traversal has actually completed. |
| 0:48 (+13 s) | `presence_clip` | Presence-belief revision, on footage | belief bar dropping as the agent inspects an empty surface | A missed detection is not absence. A miss counts only when the object should have been visible; nearer depth means occlusion and no update. And the belief ceiling is low, so a few honest misses can overturn a long history. |
| 1:01 (+5 s) | `eq3_card` | Eq. 3, the three-row log-odds table | positive / eligible miss / ineligible miss | That is the whole filter: one positive channel, one negative channel, and a third that refuses to count what it could not have seen. |
| 1:06 (+12 s) | `floor_clip` | Belief-guided search and the floor decision, on footage | floor-mass readout from search_log_events step 13 | Every mapped surface carries a search posterior. The floor score is the mean over its surfaces, and every failed inspection multiplies belief down. The agent leaves a floor because what it has searched is spent, not because it saw something new. |
| 1:18 (+5 s) | `floor_still` | Floor score panel (computed by the real code on 00873, query 'tin can') | floor_score_formula.png right panel | A switch also needs a margin, and is held while a credible mapped track on this floor is still untested. |
| 1:23 (+26 s) | `hero` | Hero: ours vs ASCENT on the identical episode | side by side, shared step clock, ours freezes on the detection | Same episode, same prior map. The bottle was mapped upstairs. MD-SG puts the search mass downstairs, takes the flight it already mapped, and finds the bottle in two hundred and seven steps. ASCENT, with the same map, never does. |
| 1:49 (+5 s) | `montage_1` | Montage 1/4: two flights down |  | Down two flights. |
| 1:54 (+5 s) | `montage_2` | Montage 2/4: one flight up |  | Up one. |
| 1:59 (+5 s) | `montage_3` | Montage 3/4: the shortest recovery |  | Straight to the stairs. |
| 2:04 (+5 s) | `montage_4` | Montage 4/4: persistence |  | And across a building the map got wrong. |
| 2:09 (+5 s) | `numbers_1` | Table II | ASCENT 0 % / 0.000 vs ours 40.0 % / 0.162 (n = 25) | On the five-scene multi-floor benchmark ASCENT succeeds in none of twenty-five episodes; MD-SG in forty percent. |
| 2:14 (+6 s) | `numbers_2` | Table I cross-anchor + Table III trend | DualMap† 30.2 % vs ours 52.8 %; ablation 41.5 / 35.8 / 52.8 | On DualMap's single-floor benchmark, cross-anchor success rises from thirty to fifty-three percent, and removing either belief mechanism costs eleven to seventeen points. |
| 2:20 (+4 s) | `R2_1` | ROBOT SLOT R2.1 | [ROBOT FOOTAGE R2.1] | On a Stretch 3 the same pipeline publishes goals to Nav2 and stops steering. In the mapping run it sees the bottle on table A |
| 2:24 (+3 s) | `R2_2` | ROBOT SLOT R2.2 | [ROBOT FOOTAGE R2.2] | and maps table B empty. |
| 2:27 (+3 s) | `R2_3` | ROBOT SLOT R2.3 | [ROBOT FOOTAGE R2.3] | Then it is asked for the blue bottle, |
| 2:30 (+3 s) | `R2_4` | ROBOT SLOT R2.4 | [ROBOT FOOTAGE R2.4] | after the bottle has been moved. |
| 2:33 (+6 s) | `R2_5` | ROBOT SLOT R2.5 | [ROBOT FOOTAGE R2.5] | It revisits table A, finds it absent, and shifts its belief to table B, |
| 2:39 (+5 s) | `R2_6` | ROBOT SLOT R2.6 | [ROBOT FOOTAGE R2.6] | drives there, and retrieves it. |
| 2:44 (+4 s) | `topology` | Deployment: Thor + bridge + Stretch | three boxes; 'the pipeline keeps thinking and stops steering' | Two containers on a Jetson Thor beside the robot; the robot keeps its own Nav2 and drivers. |
| 2:48 (+8 s) | `close` | Closing card over R3 | takeaway + paper / code line | Revisable, floor-aware scene memory: a first-class design axis for navigation in buildings that change. |

Total: 176 s of 180 s. Footage tally: ≈ 90 s simulator, 28 s robot slots, ≈ 58 s cards and
figure inserts.

## Narration

About 330 words; at a conference pace (~150 wpm) that is 2:15 of speech inside a 2:56 cut, so
the hero and the montage get their own silence. Read it as one take against the rough cut;
the timing column above is where each sentence starts.

> MD-SG: object navigation with a scene graph that lets a map be wrong, notice, and recover.
>
> The map says the cracker box is on this table. It was, before someone moved it. Pass one
> mapped the building; the object was relocated; pass two starts confidently wrong. That is
> the benchmark: map, move, search again with the old map, within a floor or across floors.
>
> We formulate dynamic multi-floor object navigation, build an online open-vocabulary
> scene-graph pipeline that revises its object beliefs, and release the first benchmark with
> controlled same-floor and cross-floor relocation.
>
> Every detection is back-projected into an ellipsoid track and filed under floor, room and
> container. Stair edges join two storeys only after a traversal has actually completed.
>
> A missed detection is not absence. A miss counts only when the object should have been
> visible; nearer depth means occlusion and no update. And the belief ceiling is low, so a few
> honest misses can overturn a long history. That is the whole filter: one positive channel,
> one negative channel, and a third that refuses to count what it could not have seen.
>
> Every mapped surface carries a search posterior. The floor score is the mean over its
> surfaces, and every failed inspection multiplies belief down. The agent leaves a floor
> because what it has searched is spent, not because it saw something new. A switch also
> needs a margin, and is held while a credible mapped track on this floor is still untested.
>
> Same episode, same prior map. The bottle was mapped upstairs. MD-SG puts the search mass
> downstairs, takes the flight it already mapped, and finds the bottle in two hundred and
> seven steps. ASCENT, with the same map, never does.
>
> Down two flights. Up one. Straight to the stairs. And across a building the map got wrong.
>
> On the five-scene multi-floor benchmark ASCENT succeeds in none of twenty-five episodes;
> MD-SG in forty percent. On DualMap's single-floor benchmark, cross-anchor success rises from
> thirty to fifty-three percent, and removing either belief mechanism costs eleven to
> seventeen points.
>
> On a Stretch 3 the same pipeline publishes goals to Nav2 and stops steering. In the mapping
> run it sees the bottle on table A and maps table B empty. Then it is asked for the blue
> bottle, after the bottle has been moved. It revisits table A, finds it absent, and shifts its
> belief to table B, drives there, and retrieves it. Two containers on a Jetson Thor beside
> the robot; the robot keeps its own Nav2 and drivers.
>
> Revisable, floor-aware scene memory: a first-class design axis for navigation in buildings
> that change.

## Robot shot list — what to film

This is paper Fig. 6 as a shot list, with the durations the cut reserves. Specs: landscape
16:9, at least 1280×720, at least 24 fps, steady (tripod or a held phone braced on something),
one continuous take per beat where possible. Also record the screen — rviz, or the terminal
showing the goal pose, `/osg/floor`, and the presence printout — for a picture-in-picture;
R2.5 in particular needs the belief visibly dropping, which only the screen can show.

Drop each clip at the path in the last column; anything longer than the slot is cut at the
slot length, anything shorter freezes on its last frame. Then `python scripts/video/build_video.py`.

| slot | s | what is in the frame | file |
|---|---|---|---|
| R1 | 4 | a hand lifts the blue bottle off table A and sets it on table B (optional; the cut works without it) | `outputs/video/robot/R1.mp4` |
| R2.1 | 4 | mapping run: the robot's camera view with the bottle on table A; PiP shows the track admitted | `outputs/video/robot/R2_1.mp4` |
| R2.2 | 3 | mapping run: the robot at table B, which is empty | `outputs/video/robot/R2_2.mp4` |
| R2.3 | 3 | the query: `python scripts/run_robot.py +experiment=stretch3_search ros2.target="blue bottle"` being typed or spoken | `outputs/video/robot/R2_3.mp4` |
| R2.4 | 3 | the bottle moved from A to B (R1 can be reused) | `outputs/video/robot/R2_4.mp4` |
| R2.5 | 6 | search run: the robot arrives at table A and finds it empty; PiP with the presence belief dropping | `outputs/video/robot/R2_5.mp4` |
| R2.6 | 5 | search run: the robot drives to table B, the detection lands on the bottle, it stops | `outputs/video/robot/R2_6.mp4` |
| R3 | 3+ | a wide shot of robot and bottle; its middle frame is the closing card's background | `outputs/video/robot/R3.mp4` |

The runs themselves are `docs/ROS2.md`'s two: `stretch3_map` then, after the bottle is moved,
`stretch3_search`. Keep both runs' `outputs/robot_maps/<tag>.json` — a reviewer may ask.

## Where the footage comes from

Every simulator clip is a pass-2 `cross_anchor_01` episode from `outputs/RERUN/mf5` (the
2026-09-18 reproduction of Table II, 11/25 = 44.0 %, the run
`docs/CROSS_ANCHOR_REPRODUCTION.md` says reproduces) or its ASCENT twin from
`outputs/ascent_crossanchor` (0/25). The debug videos are 8 fps diagnostics written by
`src/osg/eval/debug_video.py`: the left panel is live RGB with YOLOE masks and boxes (target
class in red), the right panel the costmap. `make_assets.py` centre-crops the RGB to 16:9,
crops the costmap to its non-grey bounding box and places it as an inset, and burns in the
step, floor and FSM state from that episode's `episodes.jsonl`. Speed-ups are frame skips;
no frame is synthesised.

| shot | episode | steps | why this one |
|---|---|---|---|
| `problem_clip` | `00873-bxsVRursffK …50001` cracker box | 1–54 | commits to the prior-map position at step 3 (p 0.94), the VLM verifier rejects the arrival at step 40 (`verify_debug/…call001_crackerbox_REJ.jpg`, answer "bare"), the search resumes at 54 |
| `presence_clip` | same | 156–196 | approach a mapped track (p 0.77) → close look at 171 → `absent_on_arrival` at 190, p 0.44 |
| `perception_clip` | `00821-eF36g7L6Z9M …50001` cracker box | 226–282 | the richest detection coverage of the set, ending on the recovery (282 steps, SPL 0.69) |
| `floor_clip` | `00878-XB4GS9ShBRE …50006` yellow bottle | 40–112 | the descent; the floor mass at step 13 was 4.55 (below) vs 0.38 (mapped floor) |
| `hero` | same, both policies | 1–207 / 1–489 | the identical episode: ours succeeds (SPL 0.84, floor 3 → 2 at step 158), ASCENT fails 2.4 m away |
| `montage_1` | `00821 …50001` | 190–282 | floor 1 → 0 at 223 |
| `montage_2` | `00862-LT9Jq6dN3Ea …50006` yellow bottle | 160–292 | the one *upward* recovery, floor 1 → 2 at 188 |
| `montage_3` | `00878 …50001` cracker box | 1–132 | the shortest recovery, climb at 15–84 |
| `montage_4` | `00800-TEEsavR23oF …50001` cracker box | 690–789 | the third climb attempt succeeds |

Figures: `docs/figures/layout_conditions.png` (panel d), `docs/figures/full_pipeline.png`,
`outputs/figures/scene_graph_multifloor_00873_hi.png` (left half), and
`docs/figures/floor_score_formula.png` (right panel) — all regenerable per `docs/FIGURES.md`;
`docs/figures/` is gitignored.

## What the footage does not support — say nothing beyond this

- **`00873 …50001` is not a recovery.** It is scored as a success (SPL 0.795) by the 2-D
  distance rule while the box sits 2 m below the agent, which never left the upper storey.
  The cut uses its first 54 steps and its steps 156–196 for the two belief beats and never
  shows its ending.
- **Table I's dynamic rows and all of Table III do not reproduce on a rerun**
  (`docs/CROSS_ANCHOR_REPRODUCTION.md`: in-anchor 55.6 %, cross-anchor 37.7 % on 2026-09-20,
  traced to the prior map, not the code). `numbers_2` shows the paper's numbers exactly as
  printed, for six seconds, and claims nothing more.
- **The multi-floor rerun gives 44.0 %; the card says 40.0 %**, the paper's number.
- The floor-mass readout in `floor_clip` is from step 13 of that episode; the burn-in says so.
  The belief values in `problem_clip` and `presence_clip` are the logged `goal_commit` and
  `presence_events` values; where no value was logged (after the verifier's rejection) the bar
  is hidden rather than invented.
- ASCENT's panels are ASCENT's own obstacle and value maps; the composite labels them.
- `floor_still` is computed by the real code, on 00873 with the query "tin can" — a different
  scene from the clip it follows.
- The paper's teaser overlays are schematic; the cut does not use them.
- Nothing in the robot section is measured: `Ros2Env.metrics()` returns NaN by design, and the
  narration says "demonstration", never a rate.

## Encoding

1280×720, 24 fps, H.264 high profile, two-pass at the bitrate that lands the whole file near
18.5 MB (≈ 780 kbps video for 176 s), mono AAC at 64 kbps. `build_video.py` refuses to report
success if the result exceeds 20 MB or 180 s. The robot clips are re-encoded to the same
frame on the way in, so a phone video drops in unchanged.
