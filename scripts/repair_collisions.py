#!/usr/bin/env python3
"""Re-place only the targets the all-down swap left overlapping.

The swap exchanged each flipped target's static and cross_anchor poses
wholesale. Each pose was valid in ITS OWN layout, but mixing the two sets puts
targets inside each other's 0.35 m keep-out and onto anchors their neighbours
already hold -- measured, 13 pairs from 0.02 to 0.33 m. The cost is visible in
the results: four episodes that were baseline SUCCESSES at 0.00-0.15 m, in the
same direction on the same map, now stop 2.7-4.4 m short.

Every replacement is constrained to the object's OWN storey, so the direction
the swap established is preserved -- that is the whole point of repairing
rather than reverting.

    python scripts/repair_collisions.py --dry-run
    python scripts/repair_collisions.py --scene 00821-eF36g7L6Z9M
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COLLECTOR = Path("/habitat-data-collector")
ROOT = COLLECTOR / "outputs" / "dualmap_multifloor"
SEP_M = 0.35
STOREY_M = 0.5


def files(scene: str, index: int = 1):
    d = ROOT / scene
    return {"static": d / "static_scene_config.json",
            "cross_anchor": d / "dynamic_scene_config" / "cross_anchor" / f"layout_{index:02d}.json"}


def successes(run: str) -> set:
    """(scene, semantic_id) for every episode that SUCCEEDED in `run`.

    These poses are never moved. Four of the five baseline successes were lost
    to this very repair problem, so re-placing a target that currently works
    would trade one regression for another -- the repair is only worth doing to
    the ones that are already failing.
    """
    import glob, re
    out = set()
    for path in glob.glob(f"{run}/*/episodes.jsonl"):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("success"):
                continue
            m = re.match(r"(.+?)__cross_anchor_\d+__(\d+)__", r["episode_id"])
            if m:
                out.add((m.group(1), int(m.group(2))))
    return out


def problems(scene: str, protect: set) -> list:
    """(layout, semantic_id, floor_y) for one target of each overlapping pair.

    Never a target in `protect` -- those episodes already succeed. Among the
    movable members the higher semantic id wins, purely so the choice is
    deterministic and re-running converges instead of ping-ponging the pair.
    """
    out = []
    for layout, path in files(scene).items():
        objs = json.loads(path.read_text())["objects"]
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                if math.dist(objs[i]["translation"], objs[j]["translation"]) < SEP_M:
                    movable = [o for o in (objs[i], objs[j])
                               if (scene, int(o["semantic_id"])) not in protect]
                    if not movable:
                        continue          # both already succeed: leave them be
                    pick = max(movable, key=lambda o: int(o["semantic_id"]))
                    out.append((layout, int(pick["semantic_id"]),
                                float(pick["anchor"]["floor_height"])))
    return out


def exclude_off_storey(scene: str, floor_y: float, stairs: dict) -> list:
    """Every anchor NOT on this target's storey -- so direction is preserved."""
    anchors = stairs["scenes"].get(scene, {}).get("anchors") or []
    return [a["object_id"] for a in anchors
            if a.get("mouth_height") is None
            or abs(float(a["mouth_height"]) - floor_y) > STOREY_M]


def replace(scene: str, layout: str, handle: str, exclude: list, seed: int,
            index: int, dry_run: bool):
    cmd = ["xvfb-run", "-a", "python", "scripts/replace_placement.py",
           "--root", str(ROOT), "--scene", scene, "--layout", layout,
           "--target", handle, "--seed", str(seed), "--attempts", "120"]
    if layout != "static":
        cmd += ["--index", str(index)]
    if exclude:
        cmd += ["--exclude-anchor", *exclude]
    if dry_run:
        cmd += ["--dry-run"]
    p = subprocess.run(cmd, cwd=COLLECTOR, capture_output=True, text=True)
    noise = ("Gym has been", "Please upgrade", "See the migration")
    def keep(stream):
        return [l for l in (stream or "").splitlines()
                if l.strip() and not l.startswith(noise)]
    err, out = keep(p.stderr), keep(p.stdout)
    if p.returncode != 0:
        return False, (err[-1] if err else (out[-1] if out else "?"))
    return True, (out[-1] if out else "")


def repair_scene(scene: str, stairs: dict, args, protect: set) -> list:
    log = []
    for attempt in range(args.rounds):
        todo = problems(scene, protect)
        if not todo:
            break
        handles = {int(k): v for k, v in
                   json.loads(files(scene)["static"].read_text())["id_handle_mapping"].items()}
        for layout, sid, floor_y in todo:
            ok, msg = replace(scene, layout, handles[sid],
                              exclude_off_storey(scene, floor_y, stairs),
                              args.seed + attempt, args.index, args.dry_run)
            log.append({"scene": scene, "layout": layout, "sid": sid,
                        "floor_y": round(floor_y, 2), "ok": ok, "msg": msg[:110]})
            if args.dry_run:
                return log        # nothing is written, so the list cannot shrink
    left = problems(scene, protect)
    log.append({"scene": scene, "layout": "-", "sid": -1, "floor_y": 0.0,
                "ok": not left, "msg": f"{len(left)} overlapping pair(s) remain"})
    return log


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", action="append", default=[])
    ap.add_argument("--stairs", type=Path, default=Path("outputs/audit/stair_anchors.json"))
    ap.add_argument("--index", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--protect-run", default="outputs/mf5_pass2_alldown",
                    help="never move a target whose episode succeeded here")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stairs = json.loads(args.stairs.read_text())
    protect = successes(args.protect_run)
    print(f"  protecting {len(protect)} succeeding target(s): "
          f"{sorted((s[0][:5], s[1]) for s in protect)}")
    scenes = args.scene or sorted(stairs["scenes"])
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(lambda s: repair_scene(s, stairs, args, protect), scenes))
    bad = 0
    for group in rows:
        for r in group:
            mark = "ok  " if r["ok"] else "FAIL"
            if not r["ok"]:
                bad += 1
            print(f"  {mark} {r['scene'][:5]} {r['layout']:<12} {r['sid']:<6} "
                  f"y={r['floor_y']:+6.2f}  {r['msg']}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
