# Layout-conditions figure

`docs/figures/layout_conditions.{png,pdf,svg}`, rendered by
`python scripts/render_layout_conditions.py`.

**Caption.** *The four layout conditions, on the authored layouts of HM3D
`00873-bxsVRursffK`. A ring marks where the static layout puts an object —
the pose pass 1 maps and pass 2 begins by believing — and a disc where the
relocated layout puts it. **(a) static:** pass 2 sees the layout pass 1
mapped; this is the control that separates a stale map from a hard scene.
**(b) in-anchor:** the object stays on the same piece of furniture and moves
a few tens of centimetres, so the mapped pose is wrong but an arrival at it
still sees the object. **(c) cross-anchor:** the object moves to a different
piece of furniture on the same storey; the mapped pose is simply empty, and
nothing in the prior map points at the new one. **(d) multi-floor:** the
relocation crosses a storey, so the search has to give up the storey the map
points at before any surface on the right one can be scored. Distances and
anchors are read from the layout files; the panel examples are chosen by rule
(see below), not by hand.*

**Selection rule** (in `classify` / `pick`): an object is *in-anchor* when it
keeps the same anchor instance and moves less than 0.6 m, *multi-floor* when
its nearest storey changes, and *cross-anchor* otherwise. Panels (b) and (c)
show the largest qualifying move on the storey that has both, so the two
panels are comparable; (d) shows the smallest cross-storey move that fits
both rendered floors. The script prints what it picked.

**Provenance.** Object positions, anchor names and storey heights come from
`static_scene_config.json` and `dynamic_scene_config/{in_anchor,cross_anchor}/
layout_*.json` under `$OSG_YCB_MULTI_FLOOR_ROOT`; the floor plans are the
orthographic renders already used by the teaser and the floor-decision figure
(`hm3d_topdown_00873_*`).

⚠ **The layout set is being regenerated.** At the time of writing, every
scene's `static_scene_config.json` and `cross_anchor/layout_01.json` had been
rewritten (2026-09-16 02:53) while the `in_anchor` layouts still dated from
the previous evening. Against the new static layout, some objects in the
`in_anchor` files now read as cross-anchor or even cross-floor moves — which
is why this figure classifies every object itself instead of trusting the
directory name. Re-run the script after the authoring job finishes, and check
the printed picks before quoting the numbers in the paper.
