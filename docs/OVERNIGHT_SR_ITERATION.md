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
