# Stair geometry figure

**Caption.** A real orthographic section of HM3D scene `00800-TEEsavR23oF`
shows the staircase and both landings (a). On the saved height layer, the
largest difference to neighbouring cells separates flat floor, riser-height
steps within the agent's 0.20 m climb limit, and wall-sized jumps (b). A local
steppable test needs a connected rise before its cells are trusted: the
per-cell test alone fired in 28/35 single-floor episodes. The saved stair mask
marks where the agent may step (d); as a traversability layer it is broader
than the detector's local riser band, and it gives neither a direction nor the
staircase mouth. A flight is recovered independently from contiguous cells
whose heights lie between the two storeys; its lowest and highest cells give a
foot and top (e). A nearby portal only indicates that the upper storey is
visible. After a completed upper-to-lower transition in this episode, the
graph writes a directed `StairEdge` between storey nodes with entry and exit
locations (c), preserving the containment hierarchy. The geometric flight is
the first climb target; a detected `stairs` track corroborates it, and a
portal is weaker evidence (f). Across the cited 100-episode run, such tracks
fired in only 15% of multi-floor episodes. The photo pointers
are explanatory; the map markers and edge values come from the saved map.

Render with `python3 scripts/render_stair_evidence.py`. It reads the real HM3D
section cached by `scripts/render_scene_graph_multifloor.py` at 2800 px and
section fraction 0.50, plus `outputs/maps_try_steps/00800-TEEsavR23oF.json`
and `.npz`. Output is `outputs/figures/F_stair_geometry.pdf` with PNG and SVG
variants.

---

# Stair anatomy figure (the two camera views)

**Caption.** *Height cues on one staircase. Colored dots mark the stair flight,
the set of connected intermediate-height cells recovered from the height
representation. The two circles denote its lowest and highest cells, the
candidate endpoints. **(a)** The lowest cell serves as the entry position
$\mathbf{x}_{\mathrm{entry}}$; the stair mask marks the steppable height change
at each riser, and the flat tread interiors are supplied by the flight.
**(b)** During descent, the same flight reduces to a sliver of tread at the edge
of the frame. The entry position therefore lies at the flight's highest cell.
A stair edge is added only after the traversal completes and certifies
traversability.*

Render with `python scripts/render_stair_anatomy.py`. The two views are
Habitat renders cached under `outputs/figures/`; the flight cells, their
heights and the lowest/highest markers are read from
`outputs/maps_try_steps/00800-TEEsavR23oF.{json,npz}` and projected into the
camera with a depth test, so nothing coloured is hand-placed — only the word
labels are annotations. The pose of (b) and the X come from the climb trace of
`00800-TEEsavR23oF__cross_anchor_01__50008__s0`
(`outputs/mf5_pass2_p1500/`), a different run in the same scene; say so if the
text could be read as one run. Output is
`outputs/figures/F_stair_anatomy.pdf` with PNG and SVG variants.
