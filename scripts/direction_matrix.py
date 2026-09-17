#!/usr/bin/env python3
"""Per-episode: which way must the agent cross, and did it succeed?

The arm measured 0/11 on ascents and 5/15 on descents, and a target's direction
is decided entirely by the authoring -- the agent starts on the storey the
PRIOR puts the object on (`ycb.start_on_prior_floor`), so the demand is
`sign(cross_anchor_floor - static_floor)`. This prints the matrix that decides
which targets are worth flipping.

    python scripts/direction_matrix.py
    python scripts/direction_matrix.py --run outputs/mf5_pass2_reauth --layout-root <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter
from pathlib import Path

TOL_M = 0.5


def directions(root: Path, index: int = 1) -> dict:
    out = {}
    for scene in sorted(p.name for p in Path(root).iterdir() if p.is_dir()):
        sp = Path(root) / scene / "static_scene_config.json"
        cp = (Path(root) / scene / "dynamic_scene_config" / "cross_anchor"
              / f"layout_{index:02d}.json")
        if not sp.is_file() or not cp.is_file():
            continue
        st = {int(o["semantic_id"]): o for o in json.loads(sp.read_text())["objects"]}
        ca = {int(o["semantic_id"]): o for o in json.loads(cp.read_text())["objects"]}
        handles = {int(k): v for k, v in json.loads(sp.read_text())["id_handle_mapping"].items()}
        for sid in sorted(set(st) & set(ca)):
            sy = float(st[sid]["anchor"]["floor_height"])
            cy = float(ca[sid]["anchor"]["floor_height"])
            d = cy - sy
            out[(scene, sid)] = {
                "handle": handles.get(sid, "?"),
                "dir": "same" if abs(d) <= TOL_M else ("UP" if d > 0 else "DOWN"),
                "from": sy, "to": cy,
            }
    return out


def results(run: str) -> dict:
    out = {}
    for p in glob.glob(f"{run}/*/episodes.jsonl"):
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            m = re.match(r"(.+?)__cross_anchor_\d+__(\d+)__", r["episode_id"])
            if m:
                out[(m.group(1), int(m.group(2)))] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout-root", default="/habitat-data-collector/outputs/"
                                             "dualmap_multifloor.bak_20260915_113748")
    ap.add_argument("--run", default="outputs/mf5_pass2_fixed")
    ap.add_argument("--layout-root-b")
    ap.add_argument("--run-b")
    args = ap.parse_args()

    dirs, res = directions(Path(args.layout_root)), results(args.run)
    dirs_b = directions(Path(args.layout_root_b)) if args.layout_root_b else {}
    res_b = results(args.run_b) if args.run_b else {}

    hdr = f"{'scene':<22} {'ep':<6} {'target':<21} {'dir':<5} {'ok':<3} {'reach':<6} {'dy':>5}"
    if res_b:
        hdr += f" | {'dir':<5} {'ok':<3} {'reach':<6} {'dy':>5}"
    print(hdr)
    print("-" * len(hdr))
    tally, tally_b = Counter(), Counter()
    for (scene, sid), info in sorted(dirs.items()):
        r = res.get((scene, sid))
        if r is None and (scene, sid) not in res_b:
            continue                      # same-floor target: the arm runs no episode
        row = f"{scene:<22} {sid:<6} {info['handle'][:21]:<21} {info['dir']:<5} "
        if r:
            ok = bool(r.get("success"))
            row += (f"{'YES' if ok else '.':<3} "
                    f"{str(bool(r.get('goal_floor_reached')))[:5]:<6} "
                    f"{round(r.get('traj_y_range') or 0, 1):>5}")
            tally[(info["dir"], "n")] += 1
            tally[(info["dir"], "ok")] += ok
            tally[(info["dir"], "reach")] += bool(r.get("goal_floor_reached"))
        else:
            row += f"{'-':<3} {'-':<6} {'-':>5}"
        if res_b:
            rb, ib = res_b.get((scene, sid)), dirs_b.get((scene, sid))
            if rb and ib:
                ok = bool(rb.get("success"))
                row += (f" | {ib['dir']:<5} {'YES' if ok else '.':<3} "
                        f"{str(bool(rb.get('goal_floor_reached')))[:5]:<6} "
                        f"{round(rb.get('traj_y_range') or 0, 1):>5}")
                tally_b[(ib["dir"], "n")] += 1
                tally_b[(ib["dir"], "ok")] += ok
                tally_b[(ib["dir"], "reach")] += bool(rb.get("goal_floor_reached"))
            else:
                row += f" | {'-':<5} {'-':<3} {'-':<6} {'-':>5}"
        print(row)

    for name, t in (("A  " + args.run, tally), ("B  " + (args.run_b or ""), tally_b)):
        if not t:
            continue
        print(f"\n{name}")
        print(f"   {'direction':<10} {'n':>3} {'success':>8} {'reached goal storey':>21}")
        for k in ("UP", "DOWN", "same"):
            if t[(k, "n")]:
                print(f"   {k:<10} {t[(k,'n')]:>3} {t[(k,'ok')]:>8} {t[(k,'reach')]:>21}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
