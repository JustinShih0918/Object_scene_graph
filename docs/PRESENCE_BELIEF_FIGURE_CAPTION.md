# Presence-belief figure

**Caption.** The dashed cup marks an object's stored pose. (a) Valid depth
nearer than the ellipsoid's near boundary indicates occlusion; with no matched
detection, the frame is uninformative and leaves presence unchanged. (b) Depth
beyond the expected object exposes the background; if the view is otherwise
detectable and the detector is silent, the miss lowers presence. (c) A matched
sighting raises presence, including when the expectation test did not pass.
Only the near side gates a miss. (d) A keyframe detector miss, a VLM crop
judgment at arrival, or an expectation-gated detector-scan fallback contribute
likelihood steps to the same bounded log-odds state. The VLM and detector
fallback are **alternative readings for one arrival**, not two updates from
the same event. (e) An illustrative sequence starts at $L_i=1.50$: an occluded
view gives $0$, a clear detector miss gives $-0.87$, a VLM judgment that the cup
is absent gives $-2.08$, and a later matched sighting gives $+2.48$. The dashed arrival branch
shows the detector fallback's $-1.56$ in place of the VLM reading. These values
use the configured default $(r,q)$ and are **not an experiment trace**. Rays
and objects are schematic; the implementation tests a fraction of depth samples
inside the projection. The filter's configured clip is $[-6,+3]$; only its
positive cap appears within the example's plot range.

Render with `python3 scripts/render_presence_belief.py`. Source logic is
`src/osg/objects/presence.py`; outputs are
`docs/figures/presence_belief.{svg,pdf,png}`.
