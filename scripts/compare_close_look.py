#!/usr/bin/env python3
"""The close-look A/B: four arms, paired on trial id, mechanism first.

Arms (docs/SR_PROPOSAL_CLOSE_LOOK.md): the tight-ring baseline; the look
before absence; the opportunistic look; both. The report is ordered the way
the proposal says it must be read -- did the knob move, did the population it
targets change, what did it cost -- and only then SR, with the sign test.

    python scripts/compare_close_look.py \
      --arm baseline=outputs/osg_dualmap_tightring \
      --arm inanchor=outputs/osg_closelook/inanchor \
      --arm opportunistic=outputs/osg_closelook/opportunistic \
      --arm both=outputs/osg_closelook/both
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osg.eval import dualmap_release as release  # noqa: E402

CONDITIONS = ("in_anchor", "cross_anchor")
CONTAINER_CATEGORIES = {
    "table", "desk", "counter", "shelf", "cabinet", "dresser", "nightstand", "bed",
    "sofa", "stool", "bench", "oven", "washing machine", "refrigerator",
}


def horizontal(a, b) -> float:
    ax, az = (a[0], a[2]) if len(a) == 3 else (a[0], a[1])
    bx, bz = (b[0], b[2]) if len(b) == 3 else (b[0], b[1])
    return float(np.hypot(ax - bx, az - bz))


def load(root: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    for path in sorted(root.glob("**/episodes.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            block = (r.get("authored_layout") or {}).get("dualmap") or {}
            if block.get("condition") in CONDITIONS:
                out[block["trial_id"]] = r
    return out


def sign_test(gained: int, lost: int) -> float:
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def pct(n: int, d: int) -> str:
    return f"{n}/{d} = {100.0 * n / d:.1f}%" if d else "-"


def med(values) -> str:
    values = [v for v in values if v is not None]
    return f"{statistics.median(values):.1f}" if values else "-"


def table(title: str, header: List[str], rows: List[List[str]], note: str = "") -> List[str]:
    out = ["", f"## {title}", ""]
    if note:
        out += [note, ""]
    out += ["| " + " | ".join(header) + " |",
            "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


class Maps:
    def __init__(self, root: Path):
        self.root, self.cache = root, {}

    def containers(self, scene: str) -> Dict[int, Dict[str, Any]]:
        if scene not in self.cache:
            path = self.root / scene / f"{scene}.json"
            self.cache[scene] = {} if not path.exists() else {
                int(t["id"]): t for t in json.loads(path.read_text(encoding="utf-8"))["tracks"]
                if t.get("label") in CONTAINER_CATEGORIES
            }
        return self.cache[scene]


def own_surface(r: Dict[str, Any], maps: Maps) -> Optional[Dict[str, Any]]:
    """What happened to the container the object actually sat on."""
    tracks = maps.containers(str(r["scene"]))
    if not tracks:
        return None
    new = r["authored_layout"]["target_position"]
    dist, near = min((horizontal(t["center"], new), t) for t in tracks.values())
    cid = int(near["id"])
    events = r.get("search_log_events") or []
    looks = [e for e in (r.get("close_look_log") or []) if int(e.get("container_id", -10**9)) == cid]
    glance = (r.get("glance_ranges") or {}).get(str(cid))
    return {
        "cid": cid, "label": str(near["label"]), "dist": dist,
        "selected": any(e.get("container_id") == cid and "utility" in e for e in events),
        "arrived": any(e.get("container_id") == cid and e.get("arrived") for e in events),
        "close_looked": bool(looks),
        "look_detected": any(e.get("detected") for e in looks),
        "glance_min_m": None if glance is None else float(glance),
    }


def stat(r: Dict[str, Any], key: str) -> int:
    return int((r.get("agent_stats") or {}).get(key, 0) or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True,
                        help="name=path, repeatable; the first is the baseline")
    parser.add_argument("--maps", type=Path, default=Path("outputs/maps_v5"))
    parser.add_argument("--out", type=Path, default=Path("outputs/osg_closelook/CLOSE_LOOK_AB.md"))
    args = parser.parse_args()

    arms: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        arms[name] = load(Path(path))
    base_name = next(iter(arms))
    common = set.intersection(*(set(a) for a in arms.values()))
    maps = Maps(args.maps)
    lines = ["# The close look, four arms", "",
             " · ".join(f"`{n}`: {len(a)} dynamic trials" for n, a in arms.items())
             + f" · paired on {len(common)} trial ids."]

    # 1. Did the knob move?
    rows = []
    for name, recs in arms.items():
        rs = [recs[t] for t in sorted(common)]
        started = sum(stat(r, "close_look_started") for r in rs)
        rows.append([name, str(started),
                     str(sum(stat(r, "close_look_absence") for r in rs)),
                     str(sum(stat(r, "close_look_explore") for r in rs)),
                     str(sum(stat(r, "close_look_detected") for r in rs)),
                     str(sum(stat(r, "close_look_reapproach") for r in rs)),
                     str(sum(stat(r, "close_look_goto_timeout") for r in rs)),
                     med([stat(r, "close_look_steps") / max(1, stat(r, "close_look_started"))
                          for r in rs if stat(r, "close_look_started")]),
                     str(sum(stat(r, "absence_abandon") for r in rs)),
                     str(sum(stat(r, "glance_updates") for r in rs))])
    lines += table("1. Did the knob move?",
                   ["arm", "looks started", "before absence", "opportunistic", "detected during a look",
                    "re-approached", "drive timed out", "median steps per look", "absence_abandon", "glance updates"],
                   rows, "If `looks started` is 0 in an arm meant to look, nothing below is evidence of anything.")

    # 2. The object's own surface, cross-anchor failures without a detection in the BASELINE
    base = arms[base_name]
    target_set = [t for t in sorted(common)
                  if base[t]["authored_layout"]["dualmap"]["condition"] == "cross_anchor"
                  and not base[t]["authored_layout"]["dualmap"]["success"]
                  and not int(base[t].get("gt_kf_detected") or 0)]
    rows = []
    for name, recs in arms.items():
        fates = [own_surface(recs[t], maps) for t in target_set]
        fates = [f for f in fates if f is not None]
        rows.append([name, str(len(fates)),
                     str(sum(f["selected"] for f in fates)), str(sum(f["arrived"] for f in fates)),
                     str(sum(f["close_looked"] for f in fates)), str(sum(f["look_detected"] for f in fates)),
                     str(sum(1 for f in fates if f["glance_min_m"] is not None and f["glance_min_m"] <= 2.5)),
                     str(sum(int(recs[t].get("gt_kf_close_centred") or 0) >= 3 for t in target_set)),
                     str(sum(int(recs[t].get("gt_kf_detected") or 0) > 0 for t in target_set)),
                     str(sum(int(recs[t]["authored_layout"]["dualmap"]["success"]) for t in target_set))])
    lines += table("2. The object's own surface, on the baseline's cross-anchor no-detection failures",
                   ["arm", "trials", "own surface selected", "arrived", "close-looked", "look found it",
                    "glanced from <= 2.5 m", "target had >= 3 close+centred kf", "target ever named", "scored"],
                   rows, f"The {len(target_set)} trials are fixed by the baseline, so an arm is read on the same "
                         "population. This is the table the proposal says must move before SR is read.")

    # 3. In-anchor arrive-and-leave, on the baseline's first-commit-correct in-anchor episodes
    inanchor = [t for t in sorted(common)
                if base[t]["authored_layout"]["dualmap"]["condition"] == "in_anchor"
                and (base[t].get("goal_commit_log") or [])
                and horizontal(base[t]["goal_commit_log"][0]["center"],
                               base[t]["authored_layout"]["target_position"]) <= 1.0]
    rows = []
    for name, recs in arms.items():
        rs = [recs[t] for t in inanchor]
        far = [r for r in rs if not r["authored_layout"]["dualmap"]["success"]
               and float(r["authored_layout"]["dualmap"]["final_distance_horizontal_m"]) > 3.0]
        rows.append([name, str(len(rs)),
                     str(sum(int(r["authored_layout"]["dualmap"]["success"]) for r in rs)),
                     str(len(far)), str(sum(stat(r, "absence_abandon") > 0 for r in far)),
                     str(sum(stat(r, "close_look_absence") for r in rs)),
                     str(sum(stat(r, "close_look_reapproach") for r in rs))])
    lines += table("3. In-anchor: the episodes whose first commit was already within 1 m of the object",
                   ["arm", "episodes", "scored", "ended > 3 m away", "... after absence_abandon",
                    "looks before absence", "re-approached"], rows)

    # 4. Cost
    rows = []
    for name, recs in arms.items():
        rs = [recs[t] for t in sorted(common)]
        rows.append([name, med([int(r.get("steps") or 0) for r in rs]),
                     str(sum(int(r.get("steps") or 0) >= 500 for r in rs)),
                     med([float(r["authored_layout"]["dualmap"].get("travelled_m") or 0) for r in rs]),
                     str(sum(stat(r, "close_look_steps") for r in rs))])
    lines += table("4. What it cost", ["arm", "median steps", "at budget", "median travelled m", "steps spent looking"], rows)

    # 5. Funnel
    def stage(r) -> str:
        block = r["authored_layout"]["dualmap"]
        if block["success"]:
            return "success"
        if not int(r.get("gt_kf_in_view") or 0):
            return "never in view"
        if not int(r.get("gt_kf_detected") or 0):
            return "seen, never named"
        if not int(r.get("gt_kf_admitted") or 0):
            return "named, never admitted"
        if r.get("target_obj_xy") is None:
            return "admitted, never committed"
        trial = next(t for t in release.protocol() if t["trial_id"] == block["trial_id"])
        if not release.position_correct(r["target_obj_xy"], release.trial_targets(trial)):
            return "committed elsewhere"
        return "committed, never arrived"
    stages = ["success", "never in view", "seen, never named", "named, never admitted",
              "admitted, never committed", "committed elsewhere", "committed, never arrived"]
    for c in CONDITIONS:
        rows = []
        counts = {name: collections.Counter(stage(recs[t]) for t in sorted(common)
                                            if recs[t]["authored_layout"]["dualmap"]["condition"] == c)
                  for name, recs in arms.items()}
        for s in stages:
            rows.append([s] + [str(counts[name][s]) for name in arms])
        lines += table(f"5. Funnel, {c}", ["stage"] + list(arms), rows)

    # 6. SR, paired
    rows = []
    for c in CONDITIONS:
        ids = [t for t in sorted(common) if base[t]["authored_layout"]["dualmap"]["condition"] == c]
        for name, recs in arms.items():
            if name == base_name:
                continue
            gained = sum(1 for t in ids if recs[t]["authored_layout"]["dualmap"]["success"]
                         and not base[t]["authored_layout"]["dualmap"]["success"])
            lost = sum(1 for t in ids if base[t]["authored_layout"]["dualmap"]["success"]
                       and not recs[t]["authored_layout"]["dualmap"]["success"])
            rows.append([c, name,
                         pct(sum(int(base[t]["authored_layout"]["dualmap"]["success"]) for t in ids), len(ids)),
                         pct(sum(int(recs[t]["authored_layout"]["dualmap"]["success"]) for t in ids), len(ids)),
                         str(gained), str(lost), f"{sign_test(gained, lost):.3f}"])
    lines += table("6. SR, paired against the baseline", ["split", "arm", "SR base", "SR arm", "gained", "lost", "p (sign)"],
                   rows, "107 dynamic trials cannot make a ten-point change significant; read sections 1-3 first.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
