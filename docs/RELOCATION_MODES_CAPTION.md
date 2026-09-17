# Target-relocation patterns

Figure files: [PNG](figures/relocation_modes.png), [PDF](figures/relocation_modes.pdf), [SVG](figures/relocation_modes.svg). The source views in [`figures/ycb_layouts/`](figures/ycb_layouts/) are direct Habitat-Sim renders of the authored JSON layouts.

Regenerate the source renders and composed figure with:

```bash
docker exec docker-nav-1 /opt/conda/envs/habitat/bin/python \
  /workspace/scripts/render_ycb_layout_assets.py
python3 scripts/render_relocation_modes.py
```

Suggested LaTeX caption:

```latex
\caption{Authored target-relocation patterns rendered from the YCB layouts. After a static mapping pass, a target is moved (a) within its original support surface (in-anchor), (b) to another surface on the same floor (cross-anchor), or (c) to another floor (a cross-floor subset of cross-anchor). The search pass loads the prior map and searches the changed scene. Blue and green rings mark the target in the prior and relocated layouts, respectively.}
```

The backgrounds and target meshes are direct orthographic renders of scene `00873-bxsVRursffK`, layout index 1. The rings, arrows, and text are explanatory overlays. Cross-floor is a subset of the cross-anchor condition in the authored benchmark; it is shown separately to make the storey change legible.
