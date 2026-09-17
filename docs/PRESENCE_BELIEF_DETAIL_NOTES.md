# Presence belief: detailed working notes

This is the longer source draft removed from the main Method section. Keep its
measurement and implementation arguments here for an appendix or ablation
discussion. The equation and wording below have been corrected to match the
implemented filter: expectation gates misses, while a sighting updates even
when the expectation test fails. Clipping makes the logistic value a bounded
decision score rather than a fully calibrated posterior probability.
The original §C.5–C.6 identity and candidate-use material now appears briefly
at the entrance to Method §D; the headings below preserve the archived draft.

## C. Presence Belief

Every mapped object carries one number: how far the map still believes it is
where it was put. That number is what lets a stale map be wrong, notice, and
recover, and the rest of this section is organised around it. §C.1 defines the
filter that maintains it. §C.2 settles what a single frame is worth — whether it
could have seen the object at all, and how much its answer counts. §C.3 settles
what many frames may add up to. §C.4 lets other sensors contribute to the same
number without any fusion machinery. §C.5 is the one question the belief
deliberately does not answer. §C.6 is what the agent finally does with it.

### C.1 The problem, and three channels

A conventional semantic map admits only two states for an object on it: present
(it was detected once) and absent (no track was ever created). That leaves "I
turned my head away" and "I am looking straight at it and it is not there"
indistinguishable, so a failure in a changing environment cannot be recorded at
all.

We maintain a bounded log-odds evidence filter for every track: $L_i$ records
evidence that track $i$ is still at its mapped pose, and its presence score is
$p_i = \sigma(L_i)$ with $\sigma$ the logistic function. Writing
$Z_i\in\{0,1\}$ for whether the track was detected in this frame, $E_i\in\{0,1\}$
for whether it *should* have been, $r_i$ for recall and $q$ for the false-alarm
rate (both §C.2), the update splits into three mutually exclusive
channels:

$$
\Delta L_i =
\begin{cases}
\log\dfrac{r_i}{q}, & Z_i = 1 \quad \text{(sighting)}\\[2ex]
\log\dfrac{1-r_i}{1-q}, & Z_i = 0,\ E_i = 1 \quad \text{(informative miss)}\\[2ex]
0, & Z_i = 0,\ E_i = 0 \quad \text{(uninformative)}
\end{cases}
$$

The third line is the point: **unobserved is not observed-absent**. Without the
$E$ channel, an object hidden behind a door and an object that has been carried
away receive the same negative update.

The test for $Z_i$ deliberately **ignores category**: an overlap counts as a
sighting whatever the detection's label. Association has already done the
category work, whereas presence asks whether there is still *something* at that
pose — if what is there now is a different object, that is not evidence the
original is still present, but it is even less evidence that it has been removed.

The overlap is, however, **exclusive**: each detection credits exactly one track,
the one it best matches. Crediting every track a detection merely overlaps would
let a neighbour keep a ghost alive, since an object displaced by less than a metre
still projects closely enough to be mistaken for the one the map believes in, and
the map would never receive the negative evidence it is standing in front of.

### C.2 What one frame is worth

A miss may only move belief on a frame that could have settled the question, so
$E_i = 1$ requires the track to project into the image with enough of it in
frame, at a range and an apparent size where the detector could reasonably fire,
and with enough valid depth samples inside the projection to read. The
apparent-size threshold has to be on the same scale as the ingestion gate of
§B.1, or every distant object would generate an artificial false negative. On
arrival at a goal the framing test relaxes, because an object filling the image
would otherwise fail the very view that sees it best.

The last condition is an occlusion test, and it is **one-sided**. The ellipsoid
gives the object's extent along the central ray, which defines a boundary just in
front of it; if enough depth samples fall nearer than that boundary, something is
in the way and $E_i=0$.

**That one-sidedness is the mechanism.** Only the near side is gated. A measured
depth lying **beyond** the expected band means the surface behind the object has
become visible — the most direct evidence there is that the object has been
removed, carrying the most valuable negative update, so it must never be gated
away. A measured depth lying **nearer** than the band means something is in
front, and that frame has nothing to say about that pose. One signed comparison
separates the frame that must update the belief from the frame that must not.

Given that the frame could have seen the object, how much its answer is worth is
set by the recall $r_i$ — a logistic function of the observing conditions, namely
the projected area, the range, and how oblique the view is — and by a constant
false-alarm rate $q$. Fitted from recorded runs, $r_i$ makes a miss from a close,
head-on view count for more than one from a distant, glancing view. Unfitted it
degenerates to a constant $r = 0.6$, which weights every negative update equally:
a default that is wrong but unbiased.

### C.3 What many frames may add up to

One frame is bounded by its own recall; the total is bounded explicitly, and
asymmetrically:

$$
L_i \leftarrow \mathrm{clip}\big(L_i + \Delta L_i,\ L_{\min},\ L_{\max}\big),
\qquad |L_{\min}| > L_{\max}.
$$

The two bounds are deliberately unequal. With $r=0.6$ and $q=0.05$ a sighting is
worth $+2.5$ and a miss only $-0.9$, so under a symmetric clamp three sightings
saturate the belief and **seven** clean misses would then be needed to overturn
it — which is exactly how an agent ends up standing in front of an empty shelf
while its map insists the object is there. Lowering the positive bound is an
admission that, for a fact which can change while you are not looking, no amount
of past evidence justifies near-certainty. The deeper negative floor lets
genuinely vanished objects vanish, but there is **no absorbing state**: one
re-detection can eventually restore the track's belief.

### C.4 Other sensors, and arrival as evidence

Any sensor can contribute one step with its own $(r,q)$, and two do. The detector
staying silent through an entire approach is a reliable but not certain observer;
a vision-language model answering "no" on a zoomed crop is more reliable still
but more prone to false alarms. Under their respective reliabilities one credible
model denial is worth roughly two detector misses, and **no fusion code is
required** to make it so — a direct dividend of writing presence as a filter
rather than as detector bookkeeping.

The most important of these readings is the arrival itself. When an approach ends
without the target ever having been seen, that arrival is submitted to the filter
as one observation — the agent went to the place the map named and the object was
not there. A failed call is deliberately **not** evidence: treating a network
error or a blocked line of sight as absence would silently delete objects behind
doors.

### C.5 What presence cannot answer

Presence cannot carry identity. A false positive is a **genuinely present**
object, so every observation that proves it is not the target simultaneously
re-detects it and restores its belief. Measured with this channel disabled, a
single episode committed to the same wrong track 251 times within 500 steps, its
belief pinned to the positive clamp throughout.

A separate rejection count is therefore kept, incremented by a model denial, an
unreachability verdict or a failed attempt. A track that accumulates more than a
small number of rejections leaves the candidate set whatever its presence belief.

### C.6 Using the belief: the candidate gate and its ordering

Writing $\ell_i$ and $\varphi_i$ for a track's label and the storey it is filed
on, $\ell^\star$ for the queried category and $\varphi_{\mathrm{cur}}$ for the
current storey, the candidate set is

$$
\mathcal{C} = \{\,o_i \;:\; \ell_i = \ell^\star,\ \varphi_i = \varphi_{\mathrm{cur}},\
p_i \ge p_{\min},\ \rho_i < \rho_{\max},\ \text{ingestion gates}\,\}.
$$

Candidates are only ever raised on the **current storey**; a target believed to
be elsewhere is handled by §E. Within the set, candidates are ranked by presence
belief, with accumulated evidence as a tie-break because the belief saturates at
the clamp.

Ranking by belief rather than by belief times detection confidence is a measured
choice: over 170 same-episode pairs in which one track was correct, one was wrong
and both were believed, the probability of ranking the correct one first was
$0.729$ for the product and $0.800$ for belief alone. Multiplying by detection
confidence actively hurts — a confident false positive is precisely some distant
object that really does look like the target, so a high score marks exactly where
it misleads. Proposal-only tracks, having no calibrated detection score, form
their own tier and always sort after named ones.

---
