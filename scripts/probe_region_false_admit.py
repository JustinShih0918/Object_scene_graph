#!/usr/bin/env python3
"""How often does the proposal stage admit on a frame WITHOUT the object?

`probe_region_ranking.py` measured the stage where the object is present: among
a frame's regions, does the right one rank first. That is the question "can
appearance find it", and tau = 0.24 was chosen on it -- 40 correct admissions
against 7 wrong, 85% precision.

It is the wrong denominator. Those 7 wrong are wrong REGIONS on frames that
contained the object. The live agent runs this stage on every keyframe it is
allowed to, and on nearly all of them the target is nowhere in view. What
decides whether the stage is a fallback or a source of phantom tracks is the
admission rate THERE, which was never measured.

Positives: frames from the query's own episodes, where the ground-truth pixel
of the query object is known.
Negatives: frames from other-target episodes in the same scene.

Both go through `RegionProposer.propose` itself with tau lowered out of the
way, so what is swept here is the production path.
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_perception_gap import UV_RE, covers  # noqa: E402

YCB = ["scissors", "mug", "banana", "soup_can", "cracker_box", "bowl", "pitcher", "plate"]


def dump_index(dump: Path):
    """scene -> query -> [frame dicts]"""
    out = defaultdict(lambda: defaultdict(list))
    for d in sorted(dump.iterdir()):
        if not d.is_dir():
            continue
        query = d.name.rsplit("__", 1)[-1]
        if query not in YCB:
            continue
        scene = d.name.split("_")[0]
        for png in sorted((d / "raw").glob("*.png")):
            m = UV_RE.search(png.name)
            if not m:
                continue
            out[scene][query].append(
                {"path": png, "u": int(m.group("u")), "v": int(m.group("v"))}
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, default=Path("outputs/osg_dualmap_tightring/gt_dump"))
    ap.add_argument("--queries", nargs="+", default=YCB)
    ap.add_argument("--pos", type=int, default=60, help="positive frames per query")
    ap.add_argument("--neg", type=int, default=90, help="negative frames per query")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("docs/REGION_FALSE_ADMISSION.md"))
    ap.add_argument("--json", type=Path, default=Path("outputs/region_false_admit.json"))
    args = ap.parse_args()

    from osg.core.config.region_proposal import RegionProposalConfig
    from osg.perception.region_proposer import RegionProposer, build_region_proposer

    cfg = RegionProposalConfig()
    cfg.enabled = True
    cfg.device = args.device
    tau_live = float(cfg.tau)
    cfg.tau = -1.0                      # always rank; sweep the bar afterwards

    class _Holder:                      # build_region_proposer reads cfg.region_proposal
        region_proposal = cfg

    rp = build_region_proposer(_Holder())

    idx = dump_index(args.dump)
    rows = []
    rng = np.random.default_rng(0)

    for query in args.queries:
        target = query.replace("_", " ")
        rp.set_target(target)

        pos, neg = [], []
        for scene, byq in idx.items():
            pos += byq.get(query, [])
            for other, frames in byq.items():
                if other != query:
                    neg += frames
        if not pos:
            print(f"  {query}: no positive frames, skipped")
            continue
        pos = [pos[i] for i in rng.permutation(len(pos))[: args.pos]]
        neg = [neg[i] for i in rng.permutation(len(neg))[: args.neg]]

        for kind, items in (("pos", pos), ("neg", neg)):
            for it in items:
                rgb = np.asarray(Image.open(it["path"]).convert("RGB"))
                det = rp.propose(rgb)
                s = float(rp.last_score)
                if s < 0:                       # stage declined before ranking
                    rows.append({"query": query, "kind": kind, "score": None,
                                 "covers": False})
                    continue
                hit = bool(det is not None and kind == "pos"
                           and covers(det.bbox_xyxy, it["u"], it["v"]))
                rows.append({"query": query, "kind": kind, "score": s, "covers": hit})
        n_p = sum(1 for r in rows if r["query"] == query and r["kind"] == "pos")
        n_n = sum(1 for r in rows if r["query"] == query and r["kind"] == "neg")
        print(f"  {query}: {n_p} positive, {n_n} negative frames scored")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")

    taus = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36]
    L = ["# The proposal stage on frames without the object", "",
         f"positives: the query's own episodes;  negatives: other-target episodes, same scene",
         f"live tau = {tau_live}", "",
         "## Overall", "",
         "| tau | admits on NEGATIVE frames | admits on POSITIVE frames | ...and covers the object |",
         "|---:|---:|---:|---:|"]
    P = [r for r in rows if r["kind"] == "pos"]
    N = [r for r in rows if r["kind"] == "neg"]
    for t in taus:
        na = sum(1 for r in N if r["score"] is not None and r["score"] >= t)
        pa = sum(1 for r in P if r["score"] is not None and r["score"] >= t)
        pc = sum(1 for r in P if r["score"] is not None and r["score"] >= t and r["covers"])
        L.append(f"| {t:.2f} | {na}/{len(N)} ({100*na/max(1,len(N)):.0f}%) | "
                 f"{pa}/{len(P)} ({100*pa/max(1,len(P)):.0f}%) | "
                 f"{pc}/{len(P)} ({100*pc/max(1,len(P)):.0f}%) |")

    L += ["", "## Per query, at the live tau", "",
          "| query | negative admits | positive admits | positive & covers |",
          "|---|---:|---:|---:|"]
    for query in args.queries:
        p = [r for r in P if r["query"] == query]
        n = [r for r in N if r["query"] == query]
        if not p and not n:
            continue
        na = sum(1 for r in n if r["score"] is not None and r["score"] >= tau_live)
        pa = sum(1 for r in p if r["score"] is not None and r["score"] >= tau_live)
        pc = sum(1 for r in p if r["score"] is not None and r["score"] >= tau_live and r["covers"])
        L.append(f"| {query} | {na}/{len(n)} ({100*na/max(1,len(n)):.0f}%) | "
                 f"{pa}/{len(p)} ({100*pa/max(1,len(p)):.0f}%) | "
                 f"{pc}/{len(p)} ({100*pc/max(1,len(p)):.0f}%) |")

    sp = sorted(r["score"] for r in P if r["score"] is not None)
    sn = sorted(r["score"] for r in N if r["score"] is not None)
    def q(a, f): return a[int(f * (len(a) - 1))] if a else float("nan")
    L += ["", "## Score distributions", "",
          "| population | n | p10 | median | p90 | max |", "|---|---:|---:|---:|---:|---:|",
          f"| positive | {len(sp)} | {q(sp,.1):.3f} | {q(sp,.5):.3f} | {q(sp,.9):.3f} | {max(sp):.3f} |",
          f"| negative | {len(sn)} | {q(sn,.1):.3f} | {q(sn,.5):.3f} | {q(sn,.9):.3f} | {max(sn):.3f} |"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    print("\n".join(L[5:20]))


if __name__ == "__main__":
    main()
