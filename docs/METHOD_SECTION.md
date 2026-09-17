# Method

We address object-goal navigation from an **open-vocabulary** query: the agent
receives an RGB-D stream with pose and a target named in free text, and must
navigate to an instance of it and stop. Its only persistent representation is an
object-level 3D scene graph it builds itself, online, from that stream. Building
one poses three demands that a closed-set semantic map does not meet. First, the
query may name an object no fixed-vocabulary detector can label, so the front end
must be able to admit instances it cannot name. Second, an object's 3D extent is
not observable from a single view, so the representation must accumulate partial
and oblique observations into one geometric object rather than a point. Third,
the graph must record **what supports what** — that the mug is on *that* table —
because a room centroid is both too coarse to act on and too coarse to be wrong
about. The graph is maintained per storey, since in our benchmark the target is
frequently not on the storey the agent starts on.

The same graph is then used a second time, in a scene that has since changed. The
difficulty is no longer that the map is incomplete but that it is **wrong**: it
asserts positions the agent will walk to and find empty. A wrong map turns one
question into three — where objects are, whether they are still there, and where
to look once they are not — and the two halves of the method are not independent,
because the representation built for the first is chosen for what the third
demands. The projected ellipsoid that associates a detection to a track is the
same projection that decides whether the object *should* have been visible; the
support relation the hierarchy records is the relation the search posterior is
defined over.

## A. Overview

The system takes an RGB-D stream with pose, maintains a
persistent object-level 3D scene graph, and emits one discrete action per
simulation step. Each step begins by estimating which storey the agent is on and
updating that storey's occupancy costmap from depth (§B.3); the more expensive
perception runs only on **keyframes** (§B.1). A keyframe is passed to two
detection channels — one named, one class-agnostic — whose outputs are fused into
a single detection set; each segmentation mask is back-projected through depth to
initialise a dual-quadric ellipsoid, associated with the existing tracks, and
refined once enough views have accumulated (§B.2). Tracks carrying enough
evidence become object nodes of the scene graph, hanging beneath the rooms
segmented from the costmap and the containers qualified by category and geometry
together, with storeys joined by the stair edges the agent has actually traversed
(§B.3–B.7). The same keyframe also updates each track's **presence
belief**: a detection raises it, while detector silence lowers it only if the
object should have been visible had it remained there (§C). On the decision side,
each round
arbitrates two kinds of goal under one utility-per-cost index — mapped support
surfaces, which carry a semantic prior, and unexplored frontiers, which are
purely geometric — and hands the winner to the same drive state (§D); when the
belief mass lies on another storey, the same quantity is maximised once more at
the storey level (§E). The planner then produces a path and the state machine
emits an action. If a track clears the candidate gates the agent enters its
approach and stops on arrival; and an arrival that finds nothing is itself fed
back as negative evidence, so that navigation rewrites the map rather than merely
consuming it (§C.2).

Thresholds are described below as thresholds. Their values, and the measurements
that set them, are tabulated separately; none of the arguments in this section
turns on a particular one.

---

## B. Hierarchical 3D Scene Graph

§B.1 and §B.2 produce a set of located objects. The rest of this section is about
what is built *from* them: first the structure itself (§B.3), then how each of its
levels is produced — storeys (§B.4), rooms (§B.5) and support surfaces (§B.6) —
and finally the edges that join storeys to one another (§B.7).

### B.1 Perception front end

Running detection on every simulation step is wasteful, so it runs only on
**keyframes**, triggered once the pose has moved beyond a translation or a
rotation threshold; both are set to one discrete action, so nearly every step
that changes the view produces a keyframe.

Detection uses an open-vocabulary segmentation model, conditioned on a fixed
vocabulary of common indoor categories together with the query string, above a
score threshold. What matters for the rest of the method is that it returns a
**segmentation mask** rather than a box alone: an object's points are obtained by
back-projecting its mask through depth, and a rectangle would draw the wall
behind the object into the same object.

A detector cannot name what is not in its vocabulary, and the benchmark's targets
frequently are not. Each keyframe therefore also runs a **class-agnostic**
segmenter, whose regions are filtered by area, encoded with a vision-language
model, and compared with the query string; those above a similarity threshold
join the detection set. Because such a region carries no calibrated detection
score, it is recorded separately on the track as *proposal-only* and is held to
its own bar rather than to the detector-calibrated ones below.

Two thresholds separate what is seen from what is mapped. A detection must be
large enough in the image to be ingested at all, and a track must then accumulate
enough evidence before it can answer a query. Both are calibrated against
detector output, which is why the class-agnostic channel needs a bar of its own.

### B.2 Object construction

The object layer is inherited rather than proposed. Tracks are built, associated
and refined as in VOOM [cite]; we record here only the property of the
representation that the later sections depend on, and the two places we deviate.

Each object track is a dual-quadric ellipsoid. Projected into a frame it yields a
2D Gaussian $\mathcal{E}=(\boldsymbol{\mu},\Sigma)$, and that single projection
serves twice: as the association metric here, and as the visibility test of §C.1
on which the presence belief rests. Single-view initialisation, the Wasserstein
association, the multi-view refinement of the ellipsoid's pose and extent, and
the linking of same-label fragments that one ellipsoid cannot cover, all follow
VOOM unchanged.

Two of those defaults were set for SLAM sequences and break under ObjectNav. The
task retrieves a *category*, so a merge across labels corrupts a track's label and
starves the target's candidate set; we therefore require a detection's label to
match the track's, which VOOM does not. And the agent approaches an object
head-on rather than orbiting it, leaving an arc of viewpoints too narrow for a
parallax-limited reprojection objective to constrain depth; we therefore reject
any refinement that displaces the centre beyond a fixed radius.

### B.3 The hierarchy

The scene graph is a four-level containment tree,

$$
\text{floor}(\varphi) \;\to\; \text{room}(\varphi, r) \;\to\; \text{container}(o_c) \;\to\; \text{object}(o_i),
$$

read as: this object rests on that surface, which stands in that room, which is
on that storey. An object node is a view onto the object layer of §B.2 rather
than a copy of it, so the graph holds no object state of its own and a belief
updated there is immediately visible here.

Only the object level comes from perception directly. The three levels above it
are derived from a **per-storey occupancy grid** built from depth — one 2D grid
per storey on the ground plane (the world frame has $y$ up, so grids are indexed
on $(x,z)$), each cell unknown, free or occupied, with a companion layer holding
the lowest surface height seen in each cell. The grid is a means, not a level of
the graph; the sections below use it to find storeys, to cut rooms out of free
space, and to decide which surfaces a staircase runs between.

The whole graph, together with every track's presence state, the storey heights
and the connectivity, serialises to a snapshot that can be reloaded across
episodes.

### B.4 Storeys

Storeys are estimated from two signals. The primary one is the agent's own
standing height; the secondary one is the peak of a height histogram over
observed ground points, which may pre-register a storey that is visible but not
yet visited, and may never override the primary signal. A new
storey must be separated from every existing one by more than a threshold chosen
to sit above the height of a stair landing and below the spacing between storeys,
which rules out the failure mode of registering a landing as a floor. Floor ids
are **stable**: a storey discovered later takes a new id and existing ids are
never renumbered, because the per-storey costmaps, the room ids and the frontier
blacklist are all keyed on them.

### B.5 Rooms

On each storey's free space, connected regions are separated by morphological
erosion at a width corresponding to a doorway, and components above a minimum
size become room nodes. Rooms are therefore a partition of the space the agent
has actually seen, and they grow and split as it explores.

### B.6 Containers and the support relation

The container layer answers "which surface is this object resting on". Writing
$\mathcal{L}_{\mathrm{sup}}$ for the set of plausible support categories (table,
counter, shelf, bed, …), $h_o$ for the height of an object's top face and $A_o$
for its ground footprint, a track qualifies as a support surface iff

$$
o \in \mathcal{S}
\iff
\underbrace{\ell_o \in \mathcal{L}_{\mathrm{sup}}}_{\text{category}}
\ \wedge\
\underbrace{h_{\min} \le h_o \le h_{\max}\ \wedge\ A_o \ge A_{\min}}_{\text{geometry}} .
$$

The two gates are an **intersection**, and each rules out what the other cannot.
Category alone — or vision-language similarity alone — would promote any false
positive carrying a furniture label into a permanent anchor. Geometry alone would
admit any flat patch. Note also that the height gate is a *band*: a shelf whose
top sits above head height fails the upper bound, because nothing is ever put
down there, and a mis-segmented sliver labelled "table" fails the area.

**Merging duplicates.** A surface is frequently mapped as several tracks, so
duplicate nodes must be merged. Each container carries a ground **shadow**, the
2D Gaussian $(\boldsymbol{\mu},\Sigma)$ of its footprint, and two containers of
the same label are merged iff some pair of their shadows satisfies

$$
\min\!\big(\,\mathbf{d}^{\top}\Sigma_a^{-1}\mathbf{d},\ \
\mathbf{d}^{\top}\Sigma_b^{-1}\mathbf{d}\,\big) \;\le\; \sigma^{2},
\qquad \mathbf{d} = \boldsymbol{\mu}_b - \boldsymbol{\mu}_a ,
$$

and kept as separate nodes otherwise. The quantity is the Mahalanobis distance of
one shadow's centre measured in the other's own metric — *how many of its own
radii away the neighbour sits* — so the test **scales itself to the object**. Two
halves of one bed lie about one radius apart; two nightstands the same absolute
distance apart lie several. A single absolute radius cannot serve both, and that
is why an absolute radius is not used. The $\min$ makes the test directionally
minimal: it is enough that the *smaller* shadow reaches into the larger, because
a fragment of a bed is small and the bed is not.

This is also where large objects are reconciled. A sofa or table exceeding the
field of view is seen end by end, so its mask ellipses describe genuinely
different parts and association (§B.2) correctly declines to merge them, leaving
the object in the graph as several tracks; they are rejoined here, at the level
where the aggregate geometry is available. Against HM3D's own annotations, an
absolute-radius rule leaves scene 00829 with 16 `bed` nodes in a house containing
one bed; the shadow test collapses these to 7.

### B.7 Stairs and cross-floor connectivity

**Where a staircase belongs in the graph.** A staircase belongs to no single
space; it is the medium that joins two of them. It is therefore not a node of its
own, but a set of edges at the storey level,

$$
\mathrm{StairEdge} = \big(\varphi_{\mathrm{from}},\ \varphi_{\mathrm{to}},\
\mathbf{x}_{\mathrm{entry}},\ \mathbf{x}_{\mathrm{exit}}\big),
$$

written only once the robot has actually completed a storey transition.

**Three structures, read from the height layer.** All of them are geometric; no
semantic model is required for any of them.

- A **stair mask** marks the cells a robot may step on, found from the height
  difference between neighbouring cells and kept only where a connected region
  rises far enough to join two storeys.
- A **flight** is the staircase as an object: the connected region whose heights
  lie strictly between two storeys. It is found by a separate query, not read off
  the mask.
- A **portal** is a piece of another storey seen from this one: the landing above
  caught over a mezzanine rail, or the hallway below framed in a stairwell
  opening. It is a view through a hole in the building, so it fixes where that
  storey is and at what height — but a view is not a route, and nothing about it
  says the robot can get there.

**Finding the point at which to change storey.** The flight's extreme cells are
the two ends of the staircase, and which of them is the mouth on this storey
depends on the direction of travel.

*Going up*, the robot faces the treads and sees them directly. The mouth is the
flight's **lowest** cell, the mask confirms that those cells are steppable, and a
portal — an upper floor glimpsed over a mezzanine — adds nothing, because the
flight already supplies a route.

*Going down*, the same geometry is harder to read: what the robot sees through
the opening is mostly the floor below rather than the treads, so the mouth is the
flight's **highest** cell and the evidence for it is thinner. Two things
compensate. The height band is widened only where the stair mask already vouches
for the cells, so the floor below cannot merge into the staircase. And the robot
periodically pitches its camera down to gather stair evidence from directly
ahead — but only when it is already near mapped stair cells, since looking for a
staircase in the middle of a bedroom costs two steps and risks planting stair
evidence on furniture.

**Arbitration: geometry leads, semantics confirm.** The detector's `stairs` label
serves only as corroboration — measured, it fires in just 15% of multi-floor
episodes, which is too sparse to rely on. The order is therefore
**flight > `stairs` track > portal**, and §E is what consumes it.

---

## C. Presence Belief

A mapped pose can become stale without the graph knowing when the object moved.
We therefore maintain a **presence belief** for each object track: evidence that
the object is *still at its mapped pose*. The central distinction is whether a
frame did not see the pose or inspected it and found it empty (Fig. 4).

![Visibility-gated presence revision](figures/presence_belief.svg)

*Fig. 4. Visibility and multi-sensor presence revision. (a–c) Nearer depth
blocks a miss, exposed background makes it informative, and a matched sighting
raises belief. (d) Keyframe detector misses and one arrival observer feed the
same bounded update. (e) An illustrative trajectory uses configured sensor
reliabilities; the VLM and detector fallback are alternatives.*

### C.1 Visibility-gated evidence

Let $L_i$ be a track's bounded log-odds and $p_i=\sigma(L_i)$ its presence
score. $Z_i=1$ when a detection overlaps the track's projected extent; each
detection credits at most one track. $E_i=1$ when the frame *could have detected*
the track had it remained there. Given view-dependent detector recall $r_i$ and
false-alarm rate $q$, the update used by the agent is

$$
\Delta L_i =
\begin{cases}
\log(r_i/q), & Z_i=1 \quad \text{(sighting)},\\[1ex]
\log((1-r_i)/(1-q)), & Z_i=0,\ E_i=1 \quad \text{(informative miss)},\\[1ex]
0, & Z_i=0,\ E_i=0 \quad \text{(uninformative)}.
\end{cases}
$$

A sighting is positive evidence even when the expectation test fails; that test
gates **misses**, not detections. For a miss to be informative, the stored
ellipsoid must project into a readable, sufficiently large region at a range
where the detector could fire. The same projection used for track association
provides the expected depth band. Along the central viewing ray, let
$z_{\mathrm{near}}=z_c-e_i-\delta$ be the object's near boundary, where $z_c$
is its centre depth, $e_i$ its extent and $\delta$ a tolerance. If a sufficient
fraction of valid depths in the projected region lie **nearer** than this
boundary, another surface blocks the object and $E_i=0$. Depth **beyond** the
expected object remains eligible: the background is visible, so detector
silence can lower belief. This one-sided test prevents an occluded object from
being treated like a removed one. Recall $r_i$ depends on apparent size, range
and viewing angle; its calibration and gate values are implementation details.

### C.2 Bounded revision and arrival

The evidence is accumulated with an asymmetric bound,

$$
L_i \leftarrow \operatorname{clip}(L_i+\Delta L_i,\ L_{\min},L_{\max}),
\qquad |L_{\min}|>L_{\max}.
$$

The smaller positive ceiling limits confidence from old sightings in a world
that can change while unobserved. The deeper negative bound suppresses a
disproved pose, while leaving re-detection able to revive its track. Because
of this bound, $p_i$ is a decision score rather than a fully calibrated
posterior probability.

The same update accepts a detector's silence through an approach or a
vision-language answer with that reading's own $(r,q)$. When the agent reaches
an inspectable view of the mapped pose without finding the target, arrival
contributes a negative reading. An unreadable view gives no absence evidence;
a failed model call gives no VLM reading, but an expectation-gated detector scan
may supply the arrival reading. Fig. 4d–e shows how these observers contribute
different weights to one state. Thus navigation can revise the map it followed,
rather than repeatedly committing to an empty location.

---

## D. Search Posterior and Unified Selection

Mapped target tracks are tested first. Low presence removes a track's stale
pose from direct candidates; surviving named tracks are ranked by presence,
with accumulated track evidence breaking clamp ties. Presence does not
establish **identity**: a wrong target can still be physically present, so
repeated wrong-target arrivals are handled by a separate rejection count.
These candidate gates precede the choice between search goals.

When presence evidence disproves a mapped pose, the agent chooses between
**inspecting a mapped support surface** and **opening unexplored space**. Both
are locations to visit, priced by the same planner and executed by the same
drive state.

### D.1 Two kinds of search candidate

A frontier is a reachable boundary between free and unknown cells on the
current storey. Its relevance $P_j$ combines an exploration prior with
information gain and heading continuity; in the evaluated configuration these
terms are geometric and do not use the target category. A mapped surface $x$
comes from a container node of the scene graph and receives a relative score

$$
b(x)=a(x)\,
\alpha(\ell^\star,\ell_x)^\kappa\,
\exp\!\left(-\|\mathbf c_x-\mathbf c^{\mathrm{last}}\|/L\right).
$$

For within-storey selection, $\mathcal S$ contains surfaces on the current
storey; §E decides when to request another. Here $a(x)$ is a **binary geometric
affordance**: a surface that cannot support the target is not considered.
$\alpha$ is category affinity, softened by $\kappa<1$; proximity is measured
from the target's last mapped
position, not the agent, because travel appears separately in the cost.
$b(x)$ orders plausible surfaces but is not a calibrated probability. Before
comparing it with frontier relevance, the surface scores are peak-normalised
to a fixed mass $\bar b(x)$; inspection decay is applied **after** that
normalisation so prior scale and evidence from visits do not cancel each
other.

### D.2 Shared selection and persistent inspection

Let $c(g)$ be the planner's geodesic cost to candidate $g$, $d(x)$ the chance
that inspecting surface $x$ would detect the target, and $\lambda(x)$ its
persistent surviving belief after earlier inspections. The agent chooses

$$
g^\star=\arg\max_{g\in\mathcal F\cup\mathcal S} U(g),
\qquad
U(g)=
\begin{cases}
\beta P_j/c(f_j), & g=f_j\in\mathcal F,\\[1ex]
\bar b(x)\lambda(x)d(x)/c(x), & g=x\in\mathcal S.
\end{cases}
$$

The scalar $\beta$ states how much unexplored space is worth relative to a
known plausible surface. Cheap relevance ranking shortlists candidates
before geodesic planning; the winner becomes a navigation goal of either
kind. After an unsuccessful inspection, the visited surface is discounted
rather than erased:

$$
\lambda(x)\leftarrow\lambda(x)\bigl(1-d(x)\bigr).
$$

This factor persists across attempts. A glance need not rule out a surface,
while a close negative inspection sharply reduces its value; as mapped
surfaces are discounted, reachable frontiers remain available for recovery.

---

## E. Multi-Floor Navigation Policy

Search surfaces carry stable storey keys, so residual surface belief can also
guide a storey request. In the current configuration the storey score is the
**mean** over its eligible surfaces,

$$
M_\varphi=
\frac{1}{|\mathcal S_\varphi|}
\sum_{x\in\mathcal S_\varphi}\bar b(x)\lambda(x)d(x),
\qquad
\varphi^\star=\arg\max_{\varphi:|\mathcal S_\varphi|>0} M_\varphi.
$$

The mean avoids favouring a larger storey simply because it has more
furniture. A cross-storey request needs a margin over the current storey and
is held while a target-labelled track here is still believed and untested,
using the same presence gate as §D. Failed arrivals can disprove a storey;
a timing gate limits repeated or late switches. An accepted request is routed
toward a height-derived **flight** first, a detected stairs track second,
and a geometric portal last (§B.7). The graph records a StairEdge only
after the transition is completed.
