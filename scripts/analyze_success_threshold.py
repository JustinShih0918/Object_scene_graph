#!/usr/bin/env python3
"""What would loosening the success radius do -- to us AND to DualMap?

The benchmark scores a stop within 1.0 m of the object, in up to three attempts.
Some objects have no navmesh point within 1.0 m for our agent, and the agent is
physically confined to that navmesh (docs/UNREACHABLE_SET_EXPERIMENT.md), so
those trials are unwinnable for us at any amount of search or perception skill.
DualMap is not confined: its released runner places the agent with
`set_agent_state` along its own occupancy grid.

That makes "loosen the rule" a fair question and a dangerous one. Fair, because
the rule as written is partly a statement about floor plans. Dangerous, because
a threshold change helps whoever is sitting just outside it, and that is not
necessarily us -- so it must be recomputed for BOTH systems from their own
recorded stops, which is what this does.

A trial scores at radius tau if ANY of its attempts stopped within tau.

Usage:
  python scripts/analyze_success_threshold.py --out outputs/SUCCESS_THRESHOLD.md
"""
from __future__ import annotations

import argparse, glob, json
from pathlib import Path
from typing import Dict, List, Optional

TAUS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]


def load_osg(root: str) -> Dict[str, dict]:
    out = {}
    for p in glob.glob(f"{root}/**/episodes.jsonl", recursive=True):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["authored_layout"]["dualmap"]["trial_id"]] = r
    return out


def load_dm(root: str) -> Dict[str, dict]:
    out = {}
    for p in glob.glob(f"{root}/trials/*/result.json"):
        r = json.load(open(p, encoding="utf-8"))
        out[str(r["trial_id"])] = r
    return out


def best_stop_osg(r: dict) -> Optional[float]:
    a = r["authored_layout"]["dualmap"].get("attempts") or []
    return min((float(x["distance_horizontal_m"]) for x in a), default=None)


def best_stop_dm(r: dict) -> Optional[float]:
    a = r.get("attempts") or []
    return min((float(x["distance_horizontal_m"]) for x in a), default=None)


def curve(best: Dict[str, Optional[float]], ids: List[str]) -> Dict[float, int]:
    return {t: sum(1 for i in ids if best.get(i) is not None and best[i] <= t) for t in TAUS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--navmesh", default="outputs/osg_released_maps/flat_anchor_v2_island_close")
    ap.add_argument("--sensor", default="outputs/osg_sensor_r10")
    ap.add_argument("--dualmap", default="outputs/dualmap_official_bench/seed12")
    ap.add_argument("--reachability", default="outputs/osg_released_maps/reachability.json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    nav, sen, dm = load_osg(args.navmesh), load_osg(args.sensor), load_dm(args.dualmap)
    reach = {r["trial_id"]: r for r in json.load(open(args.reachability, encoding="utf-8"))}
    ids = sorted(set(nav) & set(dm))

    L: List[str] = []
    def say(s: str = "") -> None:
        print(s); L.append(s)

    say("# Loosening the success radius, for both systems")
    say()
    say(f"{len(ids)} trials. A trial scores at radius tau if ANY attempt stopped within tau.")
    say("Sanity check against the recorded rule is the tau = 1.00 column.")
    say()

    bn = {i: best_stop_osg(nav[i]) for i in ids}
    bd = {i: best_stop_dm(dm[i]) for i in ids}
    cn, cd = curve(bn, ids), curve(bd, ids)
    say("| tau (m) | DualMap | ours, navmesh | gap |")
    say("|---:|---:|---:|---:|")
    for t in TAUS:
        say(f"| {t:.2f} | {cd[t]}/{len(ids)} | {cn[t]}/{len(ids)} | {cn[t]-cd[t]:+d} |")
    say()
    say(f"(recorded at the real rule: DualMap {sum(dm[i]['success'] for i in ids)}, "
        f"ours {sum(nav[i]['success'] for i in ids):.0f})")
    say()

    sids = sorted(set(sen) & set(dm) & set(nav))
    if sids:
        bs = {i: best_stop_osg(sen[i]) for i in sids}
        cs, cn2, cd2 = curve(bs, sids), curve(bn, sids), curve(bd, sids)
        say(f"## The sensor arm, on the {len(sids)} trials it completed")
        say()
        say("| tau (m) | DualMap | ours, navmesh | ours, sensor |")
        say("|---:|---:|---:|---:|")
        for t in TAUS:
            say(f"| {t:.2f} | {cd2[t]}/{len(sids)} | {cn2[t]}/{len(sids)} | {cs[t]}/{len(sids)} |")
        say()

    say("## What radius the floor plan actually demands")
    say()
    say("The smallest tau at which every object has SOME navmesh point within it,")
    say("i.e. the rule at which no trial is unwinnable by construction:")
    say()
    opt18 = sorted(reach[i]["ours_m"] for i in ids if i in reach)
    opt_ship = sorted(reach[i]["shipped_m"] for i in ids if i in reach)
    say("| body | trials unreachable at 1.0 m | tau needed for all | tau for 95% |")
    say("|---|---:|---:|---:|")
    say(f"| radius 0.18 (ours) | {sum(1 for x in opt18 if x > 1.0)} | {opt18[-1]:.2f} m | "
        f"{opt18[int(0.95*len(opt18))]:.2f} m |")
    say(f"| radius 0.10 (habitat default) | (see UNREACHABLE_SET_EXPERIMENT: 9 of 15 remain) | - | - |")
    say(f"| shipped HM3D mesh | {sum(1 for x in opt_ship if x > 1.0)} | {opt_ship[-1]:.2f} m | "
        f"{opt_ship[int(0.95*len(opt_ship))]:.2f} m |")
    say()
    say("## Reading it")
    say()
    say("Two things have to be true for a looser rule to be worth proposing:")
    say("it has to be justified by geometry rather than by our score, and it has")
    say("to be applied to DualMap from ITS recorded stops, not from its published")
    say("number. Both columns above are recomputed that way. Compare the gap")
    say("column across rows: if it does not widen as tau grows, loosening is not")
    say("buying us an advantage, it is only removing unwinnable trials -- which")
    say("is the honest reason to do it, and the only one worth writing down.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
