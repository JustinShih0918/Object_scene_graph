# Scene-based teaser figure

**Caption.** A prior observation places the cup in an upstairs bedroom. When
the robot checks that location after a cross-floor change, it is empty and the
upstairs bedroom–bedside table–cup containment chain becomes outdated.
Exploration recognizes the staircase,
selects a floor-switch frontier, and guides search toward a downstairs kitchen
candidate. The belief is revised from the bedside table's cup to a cup on a kitchen
counter, preserving the floor → room → container → object hierarchy. The backdrop
is a real orthographic HM3D section of `00873-bxsVRursffK`; cup positions,
robot, and arrows are schematic annotations, not a measured episode trajectory.
The red cross marks the empty old site; the dashed green outline identifies the
visible stair flight, and the route runs through it toward the downstairs cup.

Render the section in `docker-nav-1` with
`/opt/conda/envs/habitat/bin/python scripts/render_scene_graph_multifloor.py --scene 00873-bxsVRursffK --maps-root outputs/maps_p1500_osg --side-res 2800 --section-frac 0.35`.
The resulting `outputs/figures/side_00873-bxsVRursffK_2800_0.35.npz` supplies
the image committed as `docs/figures/hm3d_section_00873.png` (black void converted
to transparency). Then run `python3 scripts/render_teaser_scene.py` for the PDF,
SVG, and PNG teaser. Place the PDF near the start of a two-column paper as a
`figure*` with `\includegraphics[width=\textwidth]{figures/teaser_scene.pdf}`.
