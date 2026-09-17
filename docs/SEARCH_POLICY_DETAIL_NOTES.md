# Search and storey policy: detailed working notes

This is the longer D–E source draft removed from the main Method section. Its measurements, parameter arguments, and open arbitration audit may be used in implementation details or Results. **Before reusing it:** the archived §C.6 candidate gate now appears in Method §D; the current unified experiment uses the *mean* floor-mass rule rather than the simple sum shown in the archived E equation; and surface priors are peak-normalised to a fixed mass *before* applying inspection decay. The present cross-anchor evidence does not measure both branches winning under a β sweep.

## D. Search Posterior and Unified Selection

Exploration and re-search are **one decision** in this system, sharing a single
utility form. §D.1–D.2 cover the frontier half, §D.3 the mapped-surface half, and
§D.4 unifies them; §E lifts the same quantity to the storey level.

### D.1 Frontier extraction

On the current storey's costmap, a frontier cell is an unknown cell adjoining
free space that is **reachable from the agent** and not adjoining an obstacle.
Connected frontier cells are clustered, clusters below a minimum size are
discarded, and each centroid is snapped to the nearest free cell; a centroid with
an obstacle inside its collision radius invalidates the frontier. Survivors are
deduplicated by proximity. Every frontier records its storey, without which a
frontier upstairs and one downstairs cannot be told apart by any test on distance.

### D.2 Frontier scoring and ranking

The base form is relevance divided by true path cost,

$$
j^\star = \arg\max_j \frac{P_j}{d_j},
$$

where $d_j$ is geodesic, taken from the planner rather than computed as a
straight line, and floored so that an adjacent frontier cannot win by having no
cost at all. $P_j$ is a prior multiplied by two geometric boosts:

$$
P_j = P_j^{0}\,\cdot\,
\underbrace{\Big(1+w_g \tfrac{g_j}{\max_{j'} g_{j'}}\Big)}_{\text{information gain}}
\cdot
\underbrace{\big(1+w_c\max(0,\hat{\mathbf{u}}^\top\hat{\mathbf{h}})\big)}_{\text{continuity}} .
$$

**Information gain** $g_j$ counts the unknown cells within a radius of the
frontier, normalised by the round's maximum. It is folded in *before* the
candidate list is truncated, so the agent commits to a frontier that opens a
large unknown region rather than to the nearest small one. **Continuity** is the
alignment between the frontier's direction and the agent's heading, which
suppresses a greedy argmax oscillating across the map — each traverse costing
roughly thirty steps. A third factor penalising frontiers in direct line of sight
is available and disabled in this configuration.

Frontiers are first ranked by $P_j$ and truncated; only the survivors are charged
the real planning cost, because planning is the expensive part.

**The ranking goal and the driving goal are separate.** $d_j$ is measured to the
snapped free centroid, while the drive target remains the frontier cell itself.
The two are a cell or two apart, which barely affects ranking but dominates the
planner's success rate: measuring to an unknown cell collapses the whole round of
selection — over 100 episodes, failed selections fell from 128 to 2 — while moving
the *drive* target to the free centroid as well leaves the agent stalled at the
edge of known space, taking single-floor success from 71.4% to 60.0%.

### D.3 The search posterior over mapped surfaces

Once the target has been disproved at its mapped pose, "where should I look" must
not fall back to blind exploration. We therefore admit **mapped support surfaces**
into the same candidate set, with a belief that is a product of three terms:

$$
b(x) \;=\; \underbrace{\mathbb{1}\big[h_{\min}\le h_x \le h_{\max} \wedge A_x \ge A_{\min}\big]}_{\text{affordance}}
\;\cdot\;
\underbrace{\alpha(\ell^\star, \ell_x)^{\,\kappa}}_{\text{affinity}}
\;\cdot\;
\underbrace{\exp\!\big(-\|\mathbf{c}_x - \mathbf{c}^{\mathrm{last}}\|/L\big)}_{\text{proximity}} .
$$

**Affordance is binary and comes first.** A shelf well above head height is not a
worse place to look for a bowl; it is not a place to look for a bowl.

**Affinity** $\alpha$ is read from a category co-occurrence table. Categories the
table does not list take a neutral value rather than a low one, because most of
the benchmark's own destinations are unlisted and the most common of them is a
bed — "unlisted" is not evidence against. The exponent $\kappa<1$ softens the term
into a tie-break: it should separate candidates that are otherwise even, not
overturn a metre of distance.

**Proximity** is measured from the object's **last believed position**, not from
the agent, whose distance is already carried by the cost term. It deliberately has
no floor. Flooring the exponential is not a mixture of two hypotheses but a
truncation, under which every sufficiently distant candidate scores identically
and the ordering degenerates — and that is precisely the range in which
cross-anchor displacements lie. Keeping the exponential untruncated preserves
ordering in the far field, on the principle that weak evidence is still evidence:
measured, it raised cross-anchor top-5 hits from $1/57$ to $7/57$ and in-anchor
from $19/57$ to $29/57$.

### D.4 The unified selection rule and the inspection log

Frontiers and surfaces share one index:

$$
x^\star = \arg\max_{x}\ \frac{b(x)\,\lambda(x)\;d(x)}{c(x)},
$$

with $d(x)$ the probability that one visit would detect the object if it were
there, and $c(x)$ the planner's geodesic cost. As with frontiers, candidates are
ranked cheaply first and only the survivors are charged planning cost.

**Arbitration.** The two halves do not take turns; they **compete every round**.
Both already return a utility per unit cost, so the two are directly comparable:

$$
\text{choice} =
\begin{cases}
f_{j^\star}, & \beta \cdot \dfrac{P_{j^\star}}{d_{j^\star}} \;\ge\; \dfrac{b(x^\star)\lambda(x^\star)d(x^\star)}{c(x^\star)}\\[2.2ex]
x^\star, & \text{otherwise.}
\end{cases}
$$

$\beta$ is the only purely judgemental parameter in the system, and its meaning
is: **what is a patch of unmapped space worth, relative to a known surface that
could be holding the target?** $\beta < 1$ expresses a default preference for
known surfaces; raising it makes the agent more willing to open up the map. The
entire trade-off between exploration and re-search is controlled by this one
scalar, and we set it below one.

Whichever half wins becomes the same kind of goal and enters the same drive
state. **The two halves do not carry equal semantics**, however, and this should
be held in mind when reading §D.2 alongside §D.3:

| | frontier half | surface half |
|---|---|---|
| source of relevance | a constant prior in this configuration | affordance $\times$ affinity $\times$ proximity |
| does it know the query? | no | yes |
| additional weighting | information gain, continuity (both geometric) | a bonus for the agent's own room |

In this configuration the frontier half is therefore **purely geometric**: it
knows where the unknown is, where the space opens up and what is on the way, but
not what is being looked for. All semantics are carried by the surface half.
Combined with $\beta<1$, the system's disposition is "go to a plausible surface if
there is one, and otherwise open up the map".

Once a candidate has been inspected, its belief decays multiplicatively, and the
decay is **persistent**:

$$
\lambda(x) \leftarrow \lambda(x)\cdot\big(1 - d(x)\big).
$$

This is the fundamental difference from an ignore list. An ignore list is the
degenerate case $b \leftarrow 0$, and it is discarded when the query ends;
$\lambda$ instead distinguishes a surface glanced at from across the room (still
fairly credible) from one inspected at close range (largely ruled out), and it
survives across attempts.

The frontier half additionally carries three responsibilities in the dynamic
setting, beyond mapping: it is **the endgame fallback**, since once every mapped
surface's $\lambda$ has decayed, unexplored space wins on its own — an option an
ignore-list design cannot express, its retry loop being able only to move to the
next anchor; it **carries cross-floor search**, frontiers being the only
candidates that exist on a storey with nothing mapped on it yet; and it **does the
work in the mapping pass**, where nothing has been moved and the surface half has
almost nothing to contest.

> **A measurement still owed.** On the current cross-anchor episodes every round
> of selection was settled by a surface or by a storey switch before the frontier
> branch was reached — the frontier half has never been committed to. This does
> not show the mechanism is broken, but it does mean the present results **do not
> measure $\beta$'s arbitration behaviour**. Claiming that exploration and
> re-search are unified under one index requires a set of episodes in which both
> halves visibly win some of the time; a sweep over $\beta$ is the cheapest
> available evidence.

---

## E. Multi-Floor Navigation Policy

Nothing in §D is storey-specific, and that is the point: every search candidate
carries a stable storey key (§B.4), so the same belief mass that ranks surfaces
within a storey can be summed to rank the storeys themselves. Choosing a storey
is therefore one argmax, three suppressors and a timing gate; the exits it drives
to are the stair structures of §B.7.

$$
\varphi^\star = \arg\max_{\varphi}\ \sum_{x:\ \varphi_x = \varphi} b(x)\lambda(x).
$$

Three suppressors are applied in order.

**Margin.** If the winning storey is not the current one but does not beat it by
a margin, the decision is treated as having no opinion.

**Anchor-untested hold.** If the current storey still holds a believed track of
the target category that the agent has never actually walked to, the cross-storey
request is held down. Leaving at that moment would mean abandoning the only
hypothesis the prior map supports in exchange for an argmax over which storey
happens to have more surfaces. This condition shares its belief threshold with
the candidate gate of §C.6, and the two must never reach opposite verdicts about
the same track.

**Disproval by failed arrival.** A failed attempt marks its storey, and that
conclusion takes precedence over a posterior which is neutral by design.

A request that survives all three still has to pass a timing gate: a storey switch
may not begin too late in the step budget, nor too soon after the previous switch,
nor before the agent has spent a minimum time on the storey it is leaving. Once
through, exits are sought in order — a flight extracted from the height layer,
whose lowest tread is the foot, takes precedence over a detected `stairs` track,
which in turn takes precedence over a geometric portal.

---
