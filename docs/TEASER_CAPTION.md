# Teaser figure

**Caption.** A cup previously observed in an upstairs bedroom remains in the
scene graph after being moved to the downstairs kitchen. When the robot finds
the old location empty, it revises the bedroom–table–cup belief. Exploration
selects the staircase frontier, and the robot recovers the cup across floors;
the revised graph links floor, room, counter, and cup. The two tilted,
vertically separated floor backdrops are orthographic top-down renders of HM3D
scene 00873-bxsVRursffK. The staircase route follows a navmesh-checked
two-flight switchback, with an illustrative final approach toward the cup.
Robot pose, cup locations, and belief updates are explanatory overlays, not a
measured episode trajectory. The scene graphs on the right include floor,
room, container, and object layers.

Render the aligned floor assets, then compose the teaser:

```bash
docker exec docker-nav-1 /opt/conda/envs/habitat/bin/python /workspace/scripts/render_teaser_floor_assets.py
python3 scripts/render_teaser.py
```

The PDF and SVG keep labels and routes as vectors; the PNG is a preview. The
figure is a motivation teaser
above the paper's main text, while `F1_architecture` remains the method figure.
For a two-column LaTeX paper, place it near the beginning as a `figure*` with
`\includegraphics[width=\textwidth]{figures/teaser.pdf}` and use the caption above.
