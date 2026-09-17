# Presence-on-a-real-view figure

`docs/figures/presence_view.{png,pdf,svg}`, rendered by
`python scripts/render_presence_view.py` (`--survey` re-scores the poses).

**Caption.** *Presence belief on a view the agent really had. The camera pose
is one of the observations stored on a track of the saved map of HM3D
`00800-TEEsavR23oF`, re-rendered at the agent's own intrinsics; each outline
is that track's ellipsoid put through `Ellipsoid.project` — the same
dual-conic projection used to associate detections and to decide whether a
track should have been visible — and the number beside it is the track's
presence score $p=\sigma(L)$ at the end of the run. Colour runs from
disproved (red) to believed (green). The kitchen island's two tracks sit at
the $L=-6$ clamp after 14 misses in 16 and 21 misses in 25 exposed views; the
cabinet track on the right is at the $+3$ ceiling. A track whose mapped
geometry disagrees with the rendered depth by more than its own size is a
mapping error rather than a belief, and is not drawn.*

**What the numbers mean here.** This is a static mapping pass, so a low $p$ is
not an object that moved: it is a track the detector kept failing to confirm
from views where the stored ellipsoid was exposed — a mislocated or spurious
track being retired by the same arithmetic that retires a moved object.
That is the honest reading of this figure, and it is worth one clause in the
text; the moved-object case is what §C's schematic panel shows.

**Provenance.** Pose, intrinsics, ellipsoids, $L$, and the miss counters all
come from `outputs/maps_try_steps/00800-TEEsavR23oF.json`. The selection rule
is in the script: a track is drawn when its projected ellipse is between 11
and 110 px, its centre 0.9–6 m away and inside the frame, its evidence at
least 1.0, and its depth agrees with the rendered depth to within its own
extent; of those, up to six are kept, widest spread of belief first, none
within 95 px of another. The pose index is pinned in the script (`POSE`), so
a re-render reproduces the same figure. Only the word labels are annotation —
and they are the detector's own labels, mislabels included.
