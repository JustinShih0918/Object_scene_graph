# Reproducing the cross-anchor headline: it is not in the tree

*2026-09-20. The paper's single-floor cross-anchor result (28/53 = 52.8%, Table I)
does not reproduce today. This records what was tested, what was ruled out, and
what is left.*

## What reproduces and what does not

Re-run of both benchmarks on the current tree, against the paper:

| | rerun | paper |
|---|---|---|
| Table II multi-floor | **11/25 = 44.0%**, SPL **0.196** | 10/25 = 40.0%, SPL 0.162 |
| Table I static | 51/79 = 64.6% | 64.6% (per scene 69.2/55.6/69.2 vs 65.4/74.1/53.8) |
| Table I in-anchor | 30/54 = 55.6% | 63.0% |
| Table I cross-anchor | **20/53 = 37.7%** | **52.8%** |

The multi-floor arm reproduces. The single-floor dynamic conditions do not: 16
lost against 4 gained on the 107 shared trials, spread across every query.

## The tree that produced 62/107 is `74e0eab`

`715be7d` (09-13 07:11) is docs-only and its subject records the result —
*"Per-class fused arm on the full 107: 62, +5/-0 against _sensor_v2"*. Its parent
`74e0eab` (09-13 04:00) is therefore the code, and the locked run's shard tag
`r040922` puts its start at 04:09:22, nine minutes later.

## That tree changes nothing

Built 00829's prior map on `74e0eab` itself, same recipe
(`campaign/rebuild_maps.sh`, `LAYOUT_ROOT=outputs/collector_layouts`):

| built by | tracks | label+position signature |
|---|---|---|
| `74e0eab` | 534 | `ef3c22ad1d5c8e14` |
| HEAD | 534 | `ef3c22ad1d5c8e14` |
| what the locked run loaded | **682** | — |

Content-identical, not merely equal in count. Built twice on HEAD: 534 both
times, so the pass is deterministic.

Then 00829's 18 cross-anchor trials on `74e0eab`, using the 534-track map that
tree builds today:

| | result | mean closest approach |
|---|---|---|
| locked run (`74e0eab` + 682 map) | 12/18 | 1.22 m |
| `74e0eab` + 534 map | **10/18** | 1.12 m |
| HEAD + 534 map | **10/18** | 1.12 m |

**`74e0eab` and HEAD agree on 18 of 18 trials.** The agent code is equivalent;
the deficit is the map.

## Ruled out

| candidate | evidence |
|---|---|
| the unified preset / floor group | `_v4_fuse_cls` (the paper's own preset) scores 49/107 today against `_osg_unified`'s 50/107; they agree on 106 of 107 trials. The floor group is worth one episode. |
| `ultralytics` version | `ultralytics==8.3.*` has resolved to 8.3.253 since 2026-01-13; 8.4 came after. Its 09-14 install timestamp is the image rebuild copying files. |
| the YOLOE fp16 dtype fix (`7c2d847`) | 2026-07-15, long before. |
| detector weights | `weights_sha256` identical; the two runs' `config` blocks match bit-for-bit. |
| vocabulary | unchanged, 47 categories. |
| data inputs | `outputs/collector_layouts`, `outputs/recall/recall_model.json`, the HM3D scenes and DualMap's release root are all untouched since 2026-09-09 or earlier. |
| target tracks in the prior | the thinner map is not starving the mechanism: mean target tracks per trial 5.84 -> 5.40, and FEWER trials start with no target track (9 -> 4). |
| detector recall | slightly BETTER today: 0.352 -> 0.381 on in-view keyframes, admitted 0.246 -> 0.261. |

**Retracted.** An earlier reading of this data said the agent now "stops further
away" -- mean closest approach 0.92 m -> 1.13 m across the 107 shared trials,
straddling the 1.0 m success rule. It does not survive subsetting: on the 18
00829 cross-anchor trials the locked run is the one that ends further out
(1.22 m vs 1.12 m). The sign depends on the subset, so there is no last-half-metre
story here.

## What is left

The container. It was rebuilt **2026-09-14 21:02**, after the maps (~09-10) and
after the locked run (09-13 04:09), and the 09-13 commits reworked it
substantially — *"One Dockerfile for everything"*, *"Derive the ascent env from
the one that works, not from a narrative"*, *"Put the base image back to
11.8-devel: it is the part that provably compiled"*. Every other input is
accounted for, and the map build is deterministic given it.

What that buys, measured: 682 -> 534 tracks, which is 86.2 -> 75.0 qualified
support surfaces per trial (-13%). Cross-anchor is the condition that depends
most on the surface inventory -- the mapped pose is empty and the agent must
search *other* surfaces -- and it is the condition that fell furthest
(static ~0, in-anchor -4, cross-anchor -9).

## What would settle it

Rebuild the prior maps under the pre-09-14 image. If they come back at 682 and
cross-anchor returns to ~28/53, the cause is named and the headline is
recoverable. If the image is gone, the options are to re-baseline the paper on
the current environment or to pin an image and treat the locked run as a record
that cannot be re-derived.

Whatever the outcome: `pyproject.toml` pins `ultralytics==8.3.*` and the repo has
no lockfile for the habitat env. A run records its config fingerprint and its
detector's sha256, and neither would have caught this. Recording a `pip freeze`
beside `summary.json` would have.

## Reproducing this investigation

```bash
# the three runs behind the table above
outputs/RERUN/dualmap186            # HEAD, dualmap_osg_unified, 186 trials
outputs/RERUN/control_v4_fuse_cls   # HEAD, _v4_fuse_cls, the 107 dynamic
outputs/RERUN/tree62_xa             # 74e0eab, 00829 cross_anchor, 18 trials
outputs/RERUN/maps_tree62           # 00829's map as 74e0eab builds it today
```
