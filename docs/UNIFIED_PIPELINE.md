# The unified pipeline: one agent, three datasets

The fusion branch had grown two arms that were never the same agent:

* `dualmap_protocol_osg_sensor_v4_fuse_cls` — the DualMap released benchmark,
  **62/107** (in-anchor 34/54, cross-anchor 28/53). Region-proposal fusion,
  presence, the absence VLM, the search posterior, island snap, the close
  last-metre approach. No floor machinery.
* `ycb_authored_15_fused_v16` — the authored 15-scene and multi-floor dynamic
  benchmarks. The floor stack, per-storey costmaps, stair evidence, the flight
  climb. No region-proposal fusion, and the DualMap approach tuning turned off.

They shared 32 mechanism settings at non-default and disagreed on 54. This
document fuses them into **one agent config** carried by a single mixin, so the
only thing that changes between the three benchmarks is which dataset is loaded.

## The mixin

`configs/experiment/osg_unified_pipeline.yaml` carries every mechanism flag and
nothing about the dataset. It is the union of:

* the **floor / climb group** (32 keys), taken from `_fused_v16`:
  `floor.enabled`, `per_floor_costmap`, `stairs`, `use_prior_stairs`,
  `cross_floor`, `climb_targets: flights_first`, `no_level_on_flight`,
  `climb_enabled`, `climb_relink_flights`, `navmesh_3d_goals`,
  `mapping.multi_floor`, `scene_graph.containers_floor_relative`,
  `floor_mass_rule: mean` + `floor_mass_margin: 1.15`, `floor_llm`,
  `frontier_cost_free_cell`, and the rest;
* the **DualMap perception tuning** (22 keys), taken from `_v4_fuse_cls`:
  the region-proposal fusion (`enabled`, `every_keyframe`, `commit_min_obs 4`,
  `commit_tau 0.28`, `commit_tau_by_class` for bowl/plate/mug), `agent_radius 0.1`,
  `navmesh_snap_on_agent_island`, the close approach, the stale-anchor
  verification group;
* the **shared approach/search tuning** (4 keys) the DualMap and 15-scene bases
  carried but the bare multi-floor base did not: the fine viewpoint rings,
  `ring_radius_extent_aware`, `close_look_before_absence`,
  `search_proximity_len_m: 100`.

Three dataset presets include it:

| preset | eval base | dataset |
|---|---|---|
| `dualmap_osg_unified` | `dualmap_protocol_osg_sensor_v2` | DualMap released, 107 trials |
| `mf5_osg_unified` | `ycb_dynamic_multifloor` | 5-scene multi-floor, `outputs/dualmap_multifloor` |
| `ycb15_osg_unified` | `ycb_authored_15` | 15-scene authored, `outputs/dualmap_authoring` |

## The agent is provably identical across the three

Composing the three presets and diffing every key that is not under
`eval.` / `dualmap.` / `ycb.`:

```
dualmap_unified vs mf5_unified  : 0 mechanism differences
dualmap_unified vs ycb15_unified: 0 mechanism differences
```

And `dualmap_osg_unified` differs from the locked `_v4_fuse_cls` in **exactly
the 32 floor-group keys and nothing else** — the region-proposal fusion,
verification, and approach tuning that produce 62/107 are untouched. So the
whole question of whether the fusion preserves the DualMap result reduces to one
thing: is the floor group inert on a single-floor scene?

## Why the floor group is inert on one storey

Every floor-group flag either gates on more than one level existing, or is a
no-op when the single level sits at the agent's own standing height:

* `floor.enabled` + `estimate_only: false` runs the estimator, which finds one
  level on a single-floor scene and never emits a switch.
* `navmesh_3d_goals` snaps a goal at its floor height; on one floor that height
  equals the agent's standing height, which the code notes is "a genuine no-op
  there".
* `containers_floor_relative` subtracts the storey height from a surface's top;
  on floor 0 that height is 0, so the band is unchanged.
* `floor_mass_rule: mean` + `floor_mass_margin` only change how a *second*
  storey is scored against this one; with one storey there is nothing to score.
* `climb_*`, `cross_floor`, `use_prior_stairs`, `no_level_on_flight` all require
  a stair or a second level to fire.

The two that touch single-floor code paths regardless are
`frontier_cost_free_cell` (sets `cost_prefer_free` in the frontier scorer) and
`per_floor_costmap` (wraps the costmap in a stack). The bit-identity smoke below
is what actually decides them.

## Bit-identity smoke (the check that matters)

The floor group is **not** inert by default on a single-floor scene, and the
control run proved it is the floor group and not run-to-run numerics:

| trial (00829 cross_anchor) | `_v4_fuse_cls` in this env vs the locked run | ungated `dualmap_osg_unified` vs the lock | floor_switch_attempts |
|---|---|---|---:|
| bowl | **identical** | differs (stopped at 448 vs 500, 1.04 m vs 2.73 m) | 1 |
| cracker box | **identical** | differs (244 vs 242 steps) | 0 |
| scissors | **identical** | differs (10.99 m vs 9.87 m) | 2 |

`_v4_fuse_cls` reproduces the lock bit-for-bit, so the environment is faithful;
the drift is entirely the floor group, and it tracks `floor_switch_attempts`.
The cause: on a single-floor scene `find_portals` reads a tall bookcase as a
portal, and once the floor's frontiers are far the undirected geometric gate
opens and the agent walks off to it.

### The fix: an undirected switch requires a second known level

`floor.switch_requires_second_level` (default off, on in the mixin) makes an
UNDIRECTED switch a no-op unless the estimator already knows >= 2 levels. This
is safe by construction for the multi-floor benchmark: `apply_map` seeds every
storey of a schema-v2 prior map into the estimator on load, so a genuine
cross-floor scored run has >= 2 levels from step 0 and its switches -- directed
and undirected -- fire exactly as before. A truly single-floor scene has one
level and nowhere to go, so it is gated and reproduces `_v4_fuse_cls`.

Map-building is unaffected: prior maps are built with the baseline
`ycb_authored_15`, deliberately, so the map is never an experimental variable.

### Bit-identity: two of three, and what the residue is

Round 1 -- the two discovery gates (`switch_requires_second_level`):

| trial | ungated | gated |
|---|---|---|
| bowl | differs (448 vs 500 steps, 1.04 m vs 2.73 m) | **identical** |
| cracker box | differs (244 vs 242 steps) | **identical** |
| scissors | differs (10.99 m vs 9.87 m) | still differs (8.45 m) |

Round 2 tested `frontier_cost_free_cell`, which the fusion had taken from the
multi-floor base (`true`) while `_v4_fuse_cls` sets it `false`. **It was not the
cause**: flipping it to `false` reproduced the round-1 scissors numbers exactly
(8.446324780034837, the same `final_xy`, 23 in-view frames), so on these trials
it changes nothing. It is left at the DualMap value anyway, because it is a
general frontier-costing heuristic rather than a floor mechanism and the mixin
should not silently import the multi-floor base's opinion on it.

The residue on scissors is the floor stack itself, not a discovery behaviour:
`_v4_fuse_cls` runs `floor.estimate_only: true` ("log the estimated floor but
keep using the latched floor_y"), while the unified mixin must set it `false` so
that the estimated storey is the one actually used -- which is the whole point on
a multi-floor scene. On one storey the estimated height differs from the latched
one by centimetres, which shifts the mapping height band and therefore which
points become obstacles. That is an unavoidable consequence of running a
floor-estimating pipeline at all, and it cannot be gated away without disabling
the thing multi-floor needs.

So the honest position is: the fusion is bit-identical on the trials where the
floor stack has nothing to say, and perturbs the mapping slightly where the
estimated floor height differs from the latched one. Scissors fails in both arms
(it is 0/9 cross-anchor in the reference table -- the known-undetectable class),
so the drift changed no outcome here. Whether it changes any outcome across the
107 is what the full run measures, and that is the number that matters for
"keep 62/107", not bit-identity.
