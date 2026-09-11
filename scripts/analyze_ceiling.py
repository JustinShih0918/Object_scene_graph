#!/usr/bin/env python3
"""Where the benchmark's remaining failures actually are, paired against a
reference run.

Written for one question: after the prior maps were rebuilt in the world the
episodes are scored in (`outputs/maps_released`, see
docs/FEATURE_MEMORY_PROBE.md section 1), how much of the in-anchor gap is left,
and how much of what is left is the documented wall rather than something a
search change could move.

Every bucket is decided from the record's own fields, never from a narrative:

  perception   the target was in view on a keyframe and never named, and the
               closest it ever came was beyond a metre -- the band both
               detectors score near zero on (docs/ASSET_SUBSTITUTION_EVIDENCE.md)
  unreachable  the agent struck a candidate off as unreachable, or the follower
               gave up, and it never got within a metre of anything
  never_in_view the target never entered a keyframe at all: coverage, not
               perception, and the movable half of the funnel
  committed_elsewhere  it committed to something and ended far away
  near_miss    it ended between 1.0 and 1.6 m out, the costmap-ring band

Usage:
  python scripts/analyze_ceiling.py \\
      --run outputs/osg_released_maps/flat_anchor_v2_island_close \\
      --reference outputs/osg_close_full/flat_anchor_v2_island_close
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def load(root: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for path in glob.glob(str(Path(root) / "**" / "episodes.jsonl"), recursive=True):
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            record = json.loads(line)
            block = (record.get("authored_layout") or {}).get("dualmap") or {}
            if block.get("trial_id"):
                out[str(block["trial_id"])] = record
    return out


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial on the discordant pairs only."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def bucket(record: dict) -> str:
    if float(record.get("success", 0.0)) > 0.0:
        return "success"
    distance = float(record.get("distance_to_goal", -1.0))
    in_view = int(record.get("gt_kf_in_view") or 0)
    named = int(record.get("gt_kf_detected") or 0)
    closest = record.get("gt_min_range_m")
    closest = float(closest) if closest is not None else None
    stats = record.get("agent_stats") or {}

    if in_view == 0:
        return "never_in_view"
    if named == 0:
        # Seen and never named. Beyond a metre this is the detector wall both
        # systems share; inside a metre it is a real miss worth chasing.
        if closest is None or closest > 1.0:
            return "perception_wall"
        return "perception_close"
    if 1.0 <= distance <= 1.6:
        return "near_miss"
    if int(stats.get("unreachable_skip", 0)) > 0 and distance > 1.6:
        return "unreachable"
    return "committed_elsewhere"


def table(rows: List[str], title: str) -> None:
    print(f"\n{title}")
    counts = Counter(rows)
    width = max((len(k) for k in counts), default=10)
    for name, n in counts.most_common():
        print(f"   {name:<{width}}  {n:3d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--goal-in-anchor", type=int, default=38)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run, ref = load(args.run), load(args.reference)
    ids = sorted(set(run) & set(ref))
    lines: List[str] = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say(f"# Ceiling after the map rebuild")
    say()
    say(f"run       `{args.run}`  ({len(run)} trials)")
    say(f"reference `{args.reference}`  ({len(ref)} trials)")
    say(f"paired on {len(ids)} shared trial ids")
    say()
    say("| condition | reference | run | delta | +/- | sign test |")
    say("|---|---:|---:|---:|---:|---:|")
    for cond in ("in_anchor", "cross_anchor"):
        sel = [i for i in ids
               if ref[i]["authored_layout"]["dualmap"]["condition"] == cond]
        a = sum(ref[i]["success"] for i in sel)
        b = sum(run[i]["success"] for i in sel)
        wins = sum(1 for i in sel if run[i]["success"] > ref[i]["success"])
        losses = sum(1 for i in sel if run[i]["success"] < ref[i]["success"])
        say(f"| {cond} | {a:.0f}/{len(sel)} | {b:.0f}/{len(sel)} | {b - a:+.0f} | "
            f"+{wins}/-{losses} | {sign_test(wins, losses):.3f} |")
    a = sum(ref[i]["success"] for i in ids)
    b = sum(run[i]["success"] for i in ids)
    wins = sum(1 for i in ids if run[i]["success"] > ref[i]["success"])
    losses = sum(1 for i in ids if run[i]["success"] < ref[i]["success"])
    say(f"| **total** | {a:.0f}/{len(ids)} | {b:.0f}/{len(ids)} | {b - a:+.0f} | "
        f"+{wins}/-{losses} | {sign_test(wins, losses):.3f} |")

    say()
    say("## By query")
    say()
    say("| query | condition | reference | run |")
    say("|---|---|---:|---:|")
    queries = sorted({ref[i]["authored_layout"]["dualmap"]["query"] for i in ids})
    for query in queries:
        for cond in ("in_anchor", "cross_anchor"):
            sel = [i for i in ids
                   if ref[i]["authored_layout"]["dualmap"]["query"] == query
                   and ref[i]["authored_layout"]["dualmap"]["condition"] == cond]
            if not sel:
                continue
            a = sum(ref[i]["success"] for i in sel)
            b = sum(run[i]["success"] for i in sel)
            mark = "" if a == b else ("  **+**" if b > a else "  **-**")
            say(f"| {query} | {cond} | {a:.0f}/{len(sel)} | {b:.0f}/{len(sel)}{mark} |")

    say()
    say("## What the remaining failures are")
    say()
    for cond in ("in_anchor", "cross_anchor"):
        sel = [i for i in ids
               if run[i]["authored_layout"]["dualmap"]["condition"] == cond]
        buckets = Counter(bucket(run[i]) for i in sel)
        failures = sum(n for k, n in buckets.items() if k != "success")
        say(f"**{cond}**: {buckets.get('success', 0)}/{len(sel)} scored, {failures} failed")
        say()
        say("| bucket | n | movable? |")
        say("|---|---:|---|")
        movable = {
            "perception_wall": "no -- both detectors score ~0 beyond a metre",
            "perception_close": "yes -- seen inside a metre and still unnamed",
            "never_in_view": "yes -- coverage",
            "near_miss": "yes -- the 1.0-1.6 m ring",
            "unreachable": "no -- DualMap fails these too",
            "committed_elsewhere": "yes -- search order",
        }
        for name, n in buckets.most_common():
            if name == "success":
                continue
            say(f"| {name} | {n} | {movable.get(name, '')} |")
        say()
        if cond == "in_anchor":
            hard = buckets.get("perception_wall", 0) + buckets.get("unreachable", 0)
            scored = buckets.get("success", 0)
            say(f"Ceiling with those held fixed: {scored + failures - hard}/{len(sel)} "
                f"= {100.0 * (scored + failures - hard) / len(sel):.1f}%  "
                f"(goal {args.goal_in_anchor}/{len(sel)})")
            say()

    say("## Which objects the residue belongs to")
    say()
    say("The question the ceiling turns on: how much of what is left is the two")
    say("assets neither detector can see, and how much is everything else.")
    say()
    all_buckets = ["perception_wall", "perception_close", "never_in_view",
                   "near_miss", "committed_elsewhere", "unreachable"]
    present = [b for b in all_buckets
               if any(bucket(run[i]) == b for i in ids)]
    say("| query | " + " | ".join(present) + " | failed |")
    say("|---|" + "---:|" * (len(present) + 1))
    hard_assets = 0
    for query in queries:
        sel = [i for i in ids
               if run[i]["authored_layout"]["dualmap"]["query"] == query]
        row = Counter(bucket(run[i]) for i in sel)
        failed = sum(n for k, n in row.items() if k != "success")
        if not failed:
            continue
        if query in ("scissors", "mug"):
            hard_assets += failed
        say(f"| {query} | " + " | ".join(str(row.get(b, 0)) for b in present)
            + f" | {failed} |")
    total_failed = sum(1 for i in ids if bucket(run[i]) != "success")
    say()
    say(f"Scissors and mug account for {hard_assets} of {total_failed} remaining "
        f"failures ({100.0 * hard_assets / max(total_failed, 1):.0f}%).")
    say()
    say("## Scissors, specifically")
    say()
    say("The rebuild's whole point: 00829 now holds a real `scissors` track 0.01 m")
    say("from the object, where maps_v5 held a bleach cleanser.")
    say()
    say("| trial | ref | run | final distance | target named | closest approach |")
    say("|---|---:|---:|---:|---:|---:|")
    for i in ids:
        block = ref[i]["authored_layout"]["dualmap"]
        if block["query"] != "scissors":
            continue
        r = run[i]
        closest = r.get("gt_min_range_m")
        say(f"| {i.split('__', 1)[1]} | {ref[i]['success']:.0f} | {r['success']:.0f} | "
            f"{r['distance_to_goal']:.2f} m | {int(r.get('gt_kf_detected') or 0)} | "
            f"{'-' if closest is None else f'{float(closest):.2f} m'} |")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
