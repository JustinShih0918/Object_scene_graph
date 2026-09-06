"""Y (control) vs X (control + LLM room posterior), on the same episodes.

Reports the two numbers the campaign is actually about -- in_anchor and
cross_anchor separately -- and then the mechanism: how often the query was
answered at all, whether the room it named was the one holding the target, and
whether the episodes where it landed are the episodes that moved.

The last part matters more than the headline. The posterior is asked only once a
room has refused the agent, and the answer arrives from a reasoning model that
takes 130-200 s, so in any single run some episodes get a posterior and some do
not. Reading SR alone over a mixture of treated and untreated episodes measures
the endpoint's latency as much as the idea.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
from typing import Dict, List


# Tags are reused across campaigns -- `+run_tag=P` names a condition from
# 2026-08-27 as well as tonight's proximity arm -- so a loader that globs every
# dated directory silently merges two different experiments under one name, and
# the older one survives wherever the newer campaign has no episode with that id.
# Scope by date, always.
SINCE = "2026-09-03"


def load_runs(since: str = SINCE) -> Dict[str, Dict[str, dict]]:
    """{run_tag: {episode_id: record}} for tagged runs on or after `since`."""
    runs: Dict[str, Dict[str, dict]] = collections.defaultdict(dict)
    for jd in sorted(glob.glob("outputs/2026-*/*")):
        if jd.split("/")[1] < since:
            continue
        ov = os.path.join(jd, ".hydra", "overrides.yaml")
        if not os.path.exists(ov):
            continue
        m = re.search(r"\+run_tag=(\S+)", open(ov).read())
        if not m:
            continue
        ts = jd.split("/")[-2].replace("-", "") + "_" + jd.split("/")[-1].replace("-", "")
        path = f"outputs/{ts}/episodes.jsonl"
        if not os.path.exists(path):
            continue
        for line in open(path):
            e = json.loads(line)
            runs[m.group(1)][e["episode_id"]] = e
    return runs


def half(ep_id: str) -> str:
    return "cross_anchor" if "cross_anchor" in ep_id else "in_anchor"


def sr_table(runs, tags: List[str], common) -> None:
    print(f"{'tag':4s} {'in_anchor':>18s} {'cross_anchor':>18s} {'overall':>10s} {'SPL':>7s}")
    for t in tags:
        if t not in runs:
            continue
        eps = [runs[t][e] for e in common]
        by = collections.defaultdict(list)
        for e in eps:
            by[half(e["episode_id"])].append(e)
        i, c = by["in_anchor"], by["cross_anchor"]
        si, sc = sum(x["success"] for x in i), sum(x["success"] for x in c)
        # An unreachable goal records spl as nan; a single one poisons the mean.
        spls = [x["spl"] for x in eps if x["spl"] == x["spl"]]
        spl = sum(spls) / len(spls) if spls else float("nan")
        print(f"{t:4s} {si/len(i):8.3f} ({si:2.0f}/{len(i):2d}) "
              f"{sc/len(c):8.3f} ({sc:2.0f}/{len(c):2d}) "
              f"{(si+sc)/len(eps):10.3f} {spl:7.3f}")


def look_convert(runs, tags: List[str], common) -> None:
    print(f"\n{'tag':4s} {'half':13s} {'P(look)':>8s} {'P(succ|look)':>13s} {'SR':>7s}")
    for t in tags:
        if t not in runs:
            continue
        for h in ("in_anchor", "cross_anchor"):
            E = [runs[t][e] for e in common if half(e) == h]
            look = [x for x in E if x["gt_kf_in_view"] > 0]
            pc = sum(x["success"] for x in look) / len(look) if look else float("nan")
            print(f"{t:4s} {h:13s} {len(look)/len(E):8.3f} {pc:13.3f} "
                  f"{sum(x['success'] for x in E)/len(E):7.3f}")


def mechanism(runs, common) -> None:
    """Did the posterior actually fire, and did the episodes it fired on move?"""
    if "X" not in runs or "Y" not in runs:
        return
    print("\n--- mechanism (X only) ---")
    treated, untreated = [], []
    for e in common:
        x = runs["X"][e]
        st = x.get("agent_stats", {})
        (treated if st.get("room_prior_applied", 0) else untreated).append(e)
    print(f"queries requested : {sum(runs['X'][e].get('agent_stats', {}).get('room_prior_requests', 0) for e in common)}")
    print(f"answers applied   : {len(treated)} of {len(common)} episodes")
    for name, group in (("posterior applied", treated), ("never applied", untreated)):
        if not group:
            continue
        for h in ("in_anchor", "cross_anchor"):
            g = [e for e in group if half(e) == h]
            if not g:
                continue
            xs = sum(runs["X"][e]["success"] for e in g)
            ys = sum(runs["Y"][e]["success"] for e in g)
            print(f"  {name:18s} {h:13s} n={len(g):3d}  X {xs/len(g):.3f} ({xs:.0f})"
                  f"   Y {ys/len(g):.3f} ({ys:.0f})   delta {(xs-ys)/len(g):+.3f}")


def paired(runs, common) -> None:
    """Which episodes flipped, in each direction."""
    if "X" not in runs or "Y" not in runs:
        return
    print("\n--- paired flips (X vs Y) ---")
    won = [e for e in common if runs["X"][e]["success"] > runs["Y"][e]["success"]]
    lost = [e for e in common if runs["X"][e]["success"] < runs["Y"][e]["success"]]
    for name, group in (("X won", won), ("X lost", lost)):
        byh = collections.Counter(half(e) for e in group)
        print(f"{name}: {len(group)}  ({dict(byh)})")
        for e in sorted(group)[:12]:
            st = runs["X"][e].get("agent_stats", {})
            print(f"    {e:52s} applied={st.get('room_prior_applied', 0)} "
                  f"top_room={st.get('room_prior_top')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="Y,X")
    args = ap.parse_args()
    tags = args.tags.split(",")
    runs = load_runs()
    have = [t for t in tags if t in runs]
    missing = [t for t in tags if t not in runs]
    if missing:
        print(f"(not present yet: {missing})")
    if not have:
        return
    common = sorted(set.intersection(*[set(runs[t]) for t in have]))
    print(f"episodes common to {have}: {len(common)}\n")
    sr_table(runs, have, common)
    look_convert(runs, have, common)
    mechanism(runs, common)
    paired(runs, common)


if __name__ == "__main__":
    main()
