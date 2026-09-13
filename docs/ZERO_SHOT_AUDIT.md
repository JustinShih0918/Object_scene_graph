# Zero-shot audit: what hand-authored, per-benchmark-class knowledge is in the pipeline

The claim we want to be able to make is that the agent is **open-vocabulary and
zero-shot**: given a novel target name it has never been tuned for, it still
maps, searches and commits. This audit lists every place the pipeline currently
carries knowledge that is specific to the benchmark's own classes, ranks each by
how much it threatens that claim and how load-bearing it is, and recommends what
to do. Nothing here is removed unilaterally — the load-bearing ones are proposed
as measured A/Bs.

The tell: the two per-class prior tables cover **exactly the seven DualMap
classes and none of the seven YCB-only classes.**

| table | classes listed | = DualMap targets | YCB-only targets covered |
|---|---|---|---|
| `CONTAINER_AFFINITY` | bowl, mug, red plate, cup, bottle, cracker box, tin can, pitcher, scissors, banana, book, pillow | 7/7 | 0/7 |
| `AFFORDANCE` | same 10 | 7/7 | 0/7 |

## The findings, worst first

### 1. `AFFORDANCE` — hand per-class, and a hard veto  (graph/priors.py)

`affords(target, top_h, area_m2)` returns **0.0** and `container_prior` then
returns 0.0 — the surface is removed from the search entirely. The bands
(`bowl: (0.4, 1.3, 0.06)` …) are hand-authored for the 7 benchmark classes.
Any novel target falls to `DEFAULT_AFFORDANCE = (0.15, 1.6, 0.03)`, a generous
catch-all. So the system already runs "zero-shot" here for the 7 YCB classes —
the per-class bands only sharpen the 7 DualMap ones.

* **Threat: high** — a hard, per-class veto is the strongest form of injected
  knowledge.
* **Load-bearing: unknown, measurable** — it only acts in the search posterior
  (Phase 3, "not here → look there"), which converts rarely on DualMap and is
  central on cross-floor. Replacing all classes with `DEFAULT_AFFORDANCE` is a
  one-line A/B.
* **Recommend:** measure `DEFAULT_AFFORDANCE`-for-all. If ≤ the ±3 noise floor
  on DualMap, drop the per-class bands — it strengthens the claim at no cost.

### 2. `CONTAINER_AFFINITY` — hand per-class, but already near-inert  (graph/priors.py)

The module's own measurement: affinity-alone ranks the true surface at median 27
against 25 for arbitrary order, so it is softened to a tie-breaker
(`AFFINITY_POWER = 0.5`) and absence from the list is not penalised
(`UNLISTED_AFFINITY = 0.5`). With `affinity_llm: true` (on in every arm), a class
absent from the table is ranked by the text LLM and cached — which is what the 7
YCB classes already use.

* **Threat: medium** — hand table, but weak, and the LLM path is the real
  zero-shot mechanism already in place.
* **Load-bearing: low.**
* **Recommend:** drop the static table for the benchmark classes and let the LLM
  affinity answer for all of them uniformly — same mechanism for every class,
  nothing special-cased. Measure; expected ≈ 0.

### 3. `region_proposal.commit_tau_by_class` {bowl, plate, red_plate, mug: 0.30}

The newest addition (the 59→62 step). Defensible **in principle**: it is set from
an offline probe of how often each class's *query text* admits on object-free
frames (bowl 42%, plate 26%, mug 16% — `docs/REGION_FALSE_ADMISSION.md`), which
is a property of the class name and the text encoder, not of the benchmark
scenes. But as implemented it is a hardcoded four-class dict.

* **Threat: medium** — reads as per-benchmark tuning even though the rule is
  general.
* **Load-bearing: +3** — a uniform 0.28 is 59, a uniform 0.30 is also ~59 by a
  different three trials; only the per-class split reaches 62.
* **Recommend:** to claim zero-shot, compute the bar from the text-only probe at
  load time for whatever the target is ("raise to 0.30 for any class whose name
  admits >15% object-free"), rather than shipping the dict. Same numbers, honest
  provenance. Lower-effort alternative: keep the dict but document it as a
  worked example of that rule.

### 4. `detector.class_conf` {blue plastic pitcher, tin can, banana, red plate: 0.20}

Per-class detection thresholds, lowered for the four weakest classes. Justified
by a measured FP/recall trade (`+8 TP for +32 FP globally, but a good trade for
exactly these four`). This is detector calibration rather than scene knowledge,
but the four keys are benchmark classes.

* **Threat: low-medium** — detector calibration is normally allowed, but the
  keys are class names.
* **Recommend:** keep; note it as detector calibration. If a fully uniform gate
  is wanted, `0.20` for all costs the cracker box 18 FP for 6 TP — a bad global
  trade, so this stays per-class on merit.

### 5. `YCB_TARGET_LABELS` — asset label substitution

"yellow bottle" for mustard, "blue plastic pitcher" for pitcher, "tin can" for
soup can. The open-vocab head scores "pitcher" 0.00 and "blue plastic pitcher"
0.71 on the actual asset.

* **Threat: low** — you must give the detector *some* query string; choosing one
  it can see is naming, not scene knowledge. Every open-vocab system does this.
* **Recommend:** keep. It is the query, not a prior.

### 6. `recall_model.json` — fitted P(detected | view)

A 4-parameter logistic (bias, log area, depth, incidence) fit self-supervised on
a logged run. No class or scene identity — pure view geometry. Falls back to a
constant 0.6 when absent.

* **Threat: low** — detector/view calibration, class-agnostic.
* **Recommend:** keep; note it was fit on benchmark frames and is class-free.

### 7. `DEFAULT_VOCABULARY` — 40 fixed indoor categories

The container/furniture ontology the detector answers to (not the targets).
"box" and "book" deliberately absent to stop class-competitive NMS stealing
specific detections.

* **Threat: none** — a fixed background vocabulary is standard open-vocab
  practice and is target-independent.
* **Recommend:** keep.

## Summary

| # | item | threat | load-bearing | recommend |
|---|---|---|---|---|
| 1 | `AFFORDANCE` per-class bands | high | measurable | measure default-for-all; drop if ≤ noise |
| 2 | `CONTAINER_AFFINITY` table | medium | low | route all classes through LLM affinity |
| 3 | `commit_tau_by_class` | medium | +3 | compute from text probe, don't hardcode |
| 4 | `detector.class_conf` | low-med | yes | keep (calibration) |
| 5 | `YCB_TARGET_LABELS` | low | yes | keep (it is the query) |
| 6 | `recall_model.json` | low | small | keep (class-free) |
| 7 | `DEFAULT_VOCABULARY` | none | yes | keep |

The clean zero-shot story after acting on 1–3: **no per-target-class table in
the pipeline.** Detection uses an open vocabulary plus the target's own name;
"where does it rest" is answered by an LLM for every class alike; the commit bar
is a text-probe rule, not a benchmark dict. Items 4–7 are calibration and
naming, which a zero-shot system is allowed.
