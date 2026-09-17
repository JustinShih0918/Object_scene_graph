# Floor-score figure (the §E formula, computed)

`docs/figures/floor_score_formula.{png,pdf,svg}`, rendered by
`python scripts/render_floor_score_formula.py`. Companion to
`search_floor_decision.png`, same scene and same visual language.

**Caption.** *The floor score and the rule that acts on it, computed by the
pipeline's own code on the saved two-storey map of HM3D
`00873-bxsVRursffK` for the query "tin can", with the target's own mapped
track as the last believed position. **Left:** every eligible support surface
on each floor, sorted by $\bar b(x)\lambda(x)d(x)$; the dashed line is the
floor's score $M_\varphi$. Because $M_\varphi$ is a mean and not a sum, the
floor with 57 surfaces does not outscore the floor with 24 merely by being
larger. **Right:** the same two scores as the agent inspects the best
remaining surface on the floor it is standing on, each unsuccessful look
applying $\lambda(x)\leftarrow\lambda(x)(1-d(x))$. The scores cross after
eight inspections and the other floor clears the $1.15\times$ margin after
ten, at which point the switch request is accepted — the agent leaves because
what it has already searched is spent, not because it saw anything new.
**Bottom:** the order the decision is taken in.*

**Numbers in the figure** (printed by the script, so a re-render cannot drift
from the text): $|\mathcal S| = 57$ and $24$ surfaces, $M_\varphi = 0.104$ and
$0.049$ at the start, scores cross at 8 inspections, margin cleared at 10.

**What it must not be read as saying.** $M_\varphi$ is a mean over mapped
surfaces whose proximity term is a ground-plane distance, so it carries very
little information about *which storey* — measured on 00808's prior map, the
two storeys' means agree to within half a percent (`exploration/strategy.py`,
the comment above `floor_mass_rule`). That is why the rule is a mean plus a
margin rather than an argmax: the score is allowed to say "no opinion", and
the storey is then decided by the mechanisms that do carry floor information —
the stale anchor hold, and a storey disproved by a failed arrival.

**Provenance.** Bars, scores and the crossing are computed with
`build_container_candidates` and `InspectionLog` — the same functions
`ExplorationStrategy._select_surface` calls — at the shipped constants
(`search_detect_prob` 0.8, `search_surface_mass` 1.0,
`search_proximity_len_m` 1.0, `floor_mass_rule` mean, `floor_mass_margin`
1.15). The inspection *sequence* is the real decay rule applied to the best
remaining surface on the current floor; it is not replayed from one episode,
and the caption should not claim an episode outcome.
