#!/usr/bin/env python3
"""Paired report for the fused proposal arm against `_sensor_v2` and DualMap."""
from __future__ import annotations
import argparse, glob, json, collections
from pathlib import Path


def load(root):
    out = {}
    for f in glob.glob(f"{root}/**/episodes.jsonl", recursive=True):
        for line in open(f):
            if line.strip():
                d = json.loads(line); out[d["episode_id"]] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/osg_v4_fuse_full")
    ap.add_argument("--ref", default="outputs/osg_sensor_v2_full")
    ap.add_argument("--perception", default="data/splits/dualmap_perception.json")
    ap.add_argument("--out", type=Path, default=Path("docs/PROPOSAL_FUSION_RESULT.md"))
    a = ap.parse_args()
    run, ref = load(a.run), load(a.ref)
    perc = {t["trial_id"] for t in json.load(open(a.perception))["trials"]}
    sh = sorted(set(run) & set(ref))
    S = lambda src, ks: sum(int(src[k]["success"]) for k in ks)
    wins = [k for k in sh if run[k]["success"] > ref[k]["success"]]
    losses = [k for k in sh if run[k]["success"] < ref[k]["success"]]
    L = ["# The fused proposal arm on the full 107", "",
         f"run `{a.run}` vs reference `{a.ref}`, paired on {len(sh)} trials", "",
         "| condition | v2 | fuse | delta | +/- |", "|---|---:|---:|---:|---:|"]
    for cond in ("in_anchor", "cross_anchor"):
        ks = [k for k in sh if f"__{cond}__" in k]
        L.append(f"| {cond} | {S(ref,ks)}/{len(ks)} | {S(run,ks)}/{len(ks)} | {S(run,ks)-S(ref,ks):+d} | "
                 f"+{sum(k in wins for k in ks)}/-{sum(k in losses for k in ks)} |")
    L.append(f"| **total** | {S(ref,sh)}/{len(sh)} | {S(run,sh)}/{len(sh)} | {S(run,sh)-S(ref,sh):+d} | +{len(wins)}/-{len(losses)} |")
    ps = [k for k in sh if k in perc]
    L += ["", f"Perception subset ({len(ps)} trials the detector never names): v2 {S(ref,ps)} -> fuse {S(run,ps)}",
          f"Everything else ({len(sh)-len(ps)}): v2 {S(ref,sh)-S(ref,ps)} -> fuse {S(run,sh)-S(run,ps)}", ""]
    # proposal-only commit accounting
    pc = eps = att = sc = 0
    for k in sh:
        gl = run[k].get("goal_commit_log") or []
        po = [g for g in gl if g.get("proposal_only")]; pc += len(po); eps += bool(po)
        for at in run[k].get("attempt_log") or []:
            prev = [g for g in gl if g["step"] <= at["step"]]
            if prev and prev[-1].get("proposal_only"):
                att += 1; sc += int(bool(at["success"]))
    L += ["## Proposal-only commits", "",
          f"{pc} commits in {eps} episodes; {att} attempts spent on them, {sc} scored.", ""]
    L += ["## Trials that changed", "", "| trial | v2 | fuse | perception | proposal-only commit | fuse dist | attempts |", "|---|---:|---:|---|---|---:|---:|"]
    for k in wins + losses:
        gl = run[k].get("goal_commit_log") or []
        L.append(f"| {k} | {int(ref[k]['success'])} | {int(run[k]['success'])} | {'yes' if k in perc else ''} | "
                 f"{'yes' if any(g.get('proposal_only') for g in gl) else ''} | {run[k]['distance_to_goal']:.2f} | {run[k].get('attempts_used')} |")
    L += ["", "## By query", "", "| query | condition | v2 | fuse |", "|---|---|---:|---:|"]
    byq = collections.defaultdict(lambda: [0, 0, 0])
    for k in sh:
        q = k.rsplit("__", 1)[-1]; c = "in_anchor" if "__in_anchor__" in k else "cross_anchor"
        byq[(q, c)][0] += 1; byq[(q, c)][1] += int(ref[k]["success"]); byq[(q, c)][2] += int(run[k]["success"])
    for (q, c), (n, r, f) in sorted(byq.items()):
        mark = " **+**" if f > r else (" **-**" if f < r else "")
        L.append(f"| {q} | {c} | {r}/{n} | {f}/{n}{mark} |")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
