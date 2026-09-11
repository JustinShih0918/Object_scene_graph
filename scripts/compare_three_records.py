#!/usr/bin/env python3
"""DualMap, our navmesh record, and our sensor-only record, on the same 107.

Three columns, one row per trial, same starts, same 1 m / 3-attempt rule. The
point of keeping all three is that they are not three attempts at one number:

  DualMap          its own released runner, which places the agent with
                   `set_agent_state` along its own occupancy grid and is not
                   held to habitat's navmesh at all.
  ours (navmesh)   habitat's ShortestPathFollower on the ground-truth navmesh,
                   plus the `is_reachable` island oracle and a closing walk
                   that picked its goal with a pathfinder query. Privileged.
  ours (sensor)    a frozen PointNav policy on depth and pose, no navmesh
                   query anywhere, on a 0.10 m body.

Read SR beside the reachability ceiling: the agent is physically confined to
the navmesh habitat-lab recomputes for its radius, so some objects cannot be
approached to within 1 m at all (docs/UNREACHABLE_SET_EXPERIMENT.md). That
ceiling is 92/107 at radius 0.18 and 98/107 at 0.10, and DualMap is not subject
to it.
"""
from __future__ import annotations

import argparse, glob, json, math, statistics as st
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


def load_osg(root: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for p in glob.glob(f"{root}/**/episodes.jsonl", recursive=True):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["authored_layout"]["dualmap"]["trial_id"]] = r
    return out


def load_dualmap(root: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for p in glob.glob(f"{root}/trials/*/result.json"):
        r = json.load(open(p, encoding="utf-8"))
        out[str(r["trial_id"])] = r
    return out


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n))


def cond_of(rec: dict) -> str:
    return rec["authored_layout"]["dualmap"]["condition"]


def query_of(rec: dict) -> str:
    return rec["authored_layout"]["dualmap"]["query"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--navmesh", default="outputs/osg_released_maps/flat_anchor_v2_island_close")
    ap.add_argument("--sensor", default="outputs/osg_sensor_r10/sensor_r10")
    ap.add_argument("--dualmap-seeds", nargs="+",
                    default=["outputs/dualmap_official_bench/seed12",
                             "outputs/dualmap_official_bench/seed13",
                             "outputs/dualmap_official_bench/seed14"])
    ap.add_argument("--reachability", default="outputs/osg_released_maps/reachability.json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    nav, sen = load_osg(args.navmesh), load_osg(args.sensor)
    dms = [load_dualmap(s) for s in args.dualmap_seeds]
    dm0 = dms[0]
    reach = {r["trial_id"]: r for r in json.load(open(args.reachability, encoding="utf-8"))}
    ids = sorted(set(nav) & set(sen) & set(dm0))

    L: List[str] = []
    def say(s: str = "") -> None:
        print(s); L.append(s)

    say("# DualMap, our navmesh record, and our sensor-only record")
    say()
    say(f"{len(ids)} trials shared by all three. Same starts, same 1 m / 3-attempt rule.")
    say()
    say(f"- DualMap  `{args.dualmap_seeds[0]}` (+{len(dms)-1} more seeds for the spread)")
    say(f"- navmesh  `{args.navmesh}`")
    say(f"- sensor   `{args.sensor}`")
    say()
    say("## Success rate")
    say()
    say("| condition | DualMap | ours, navmesh | ours, sensor-only r=0.10 |")
    say("|---|---:|---:|---:|")
    for cond in ("in_anchor", "cross_anchor"):
        sel = [i for i in ids if cond_of(nav[i]) == cond]
        d = [sum(m[i]["success"] for i in sel) for m in dms]
        a = sum(nav[i]["success"] for i in sel)
        b = sum(sen[i]["success"] for i in sel)
        spread = f" (seeds {'/'.join(str(x) for x in d)})" if len(d) > 1 else ""
        say(f"| {cond} | {d[0]}/{len(sel)}{spread} | {a:.0f}/{len(sel)} | {b:.0f}/{len(sel)} |")
    d = [sum(m[i]["success"] for i in ids) for m in dms]
    a = sum(nav[i]["success"] for i in ids)
    b = sum(sen[i]["success"] for i in ids)
    say(f"| **total** | {d[0]}/{len(ids)} | {a:.0f}/{len(ids)} | {b:.0f}/{len(ids)} |")
    say()
    say("## Paired, sensor-only against each")
    say()
    say("| condition | vs navmesh | sign test | vs DualMap seed 12 | sign test |")
    say("|---|---:|---:|---:|---:|")
    for cond in ("in_anchor", "cross_anchor", "all"):
        sel = [i for i in ids if cond == "all" or cond_of(nav[i]) == cond]
        w1 = sum(1 for i in sel if sen[i]["success"] > nav[i]["success"])
        l1 = sum(1 for i in sel if sen[i]["success"] < nav[i]["success"])
        w2 = sum(1 for i in sel if sen[i]["success"] > dm0[i]["success"])
        l2 = sum(1 for i in sel if sen[i]["success"] < dm0[i]["success"])
        say(f"| {cond} | +{w1}/-{l1} | {sign_test(w1,l1):.3f} | +{w2}/-{l2} | {sign_test(w2,l2):.3f} |")
    say()
    say("## Against the reachability ceiling")
    say()
    say("The agent cannot stand within 1 m of some objects at all. That bound moves")
    say("with the body, and DualMap is not subject to it.")
    say()
    say("| set | n | DualMap | ours, navmesh (r=0.18) | ours, sensor (r=0.10) |")
    say("|---|---:|---:|---:|---:|")
    for name, pred in (("reachable at r=0.18", lambda i: reach[i]["ours_m"] <= 1.0),
                       ("NOT reachable at r=0.18", lambda i: reach[i]["ours_m"] > 1.0)):
        sel = [i for i in ids if i in reach and pred(i)]
        if not sel:
            continue
        say(f"| {name} | {len(sel)} | {sum(dm0[i]['success'] for i in sel)} | "
            f"{sum(nav[i]['success'] for i in sel):.0f} | {sum(sen[i]['success'] for i in sel):.0f} |")
    say()
    say("## Did the agent actually find the object?")
    say()
    say("A success within 1 m that never named the target is credit for standing in")
    say("the right place. Localised = the target was named on some keyframe.")
    say()
    say("| | DualMap | ours, navmesh | ours, sensor |")
    say("|---|---:|---:|---:|")
    ln = sum(1 for i in ids if nav[i]["success"] and int(nav[i].get("gt_kf_detected") or 0) > 0)
    ls = sum(1 for i in ids if sen[i]["success"] and int(sen[i].get("gt_kf_detected") or 0) > 0)
    say(f"| localised successes | (not comparable) | {ln}/{a:.0f} | {ls}/{b:.0f} |")
    say()
    say("## Distance and cost")
    say()
    say("| | DualMap | ours, navmesh | ours, sensor |")
    say("|---|---:|---:|---:|")
    fd = [dm0[i]["final_distance_horizontal_m"] for i in ids]
    fn = [nav[i]["distance_to_goal"] for i in ids]
    fs = [sen[i]["distance_to_goal"] for i in ids]
    say(f"| median final distance | {st.median(fd):.2f} m | {st.median(fn):.2f} m | {st.median(fs):.2f} m |")
    say(f"| final within 2 m | {sum(1 for x in fd if x<=2)} | {sum(1 for x in fn if x<=2)} | {sum(1 for x in fs if x<=2)} |")
    say(f"| median SPL | {st.median([dm0[i]['spl'] for i in ids]):.3f} | "
        f"{st.median([nav[i]['spl'] for i in ids]):.3f} | {st.median([sen[i]['spl'] for i in ids]):.3f} |")
    say(f"| median steps | - | {st.median([nav[i]['steps'] for i in ids]):.0f} | "
        f"{st.median([sen[i]['steps'] for i in ids]):.0f} |")
    say()
    say("## By query")
    say()
    say("| query | condition | DualMap | navmesh | sensor |")
    say("|---|---|---:|---:|---:|")
    for q in sorted({query_of(nav[i]) for i in ids}):
        for cond in ("in_anchor", "cross_anchor"):
            sel = [i for i in ids if query_of(nav[i]) == q and cond_of(nav[i]) == cond]
            if not sel:
                continue
            x, y, z = (sum(dm0[i]["success"] for i in sel),
                       sum(nav[i]["success"] for i in sel), sum(sen[i]["success"] for i in sel))
            mark = "" if z == y else ("  **+**" if z > y else "  **-**")
            say(f"| {q} | {cond} | {x}/{len(sel)} | {y:.0f}/{len(sel)} | {z:.0f}/{len(sel)}{mark} |")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
