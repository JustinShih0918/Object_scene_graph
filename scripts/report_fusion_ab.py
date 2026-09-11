#!/usr/bin/env python3
"""Compare the fused arm against the baseline, episode for episode.

The arms are paired: same code, same manifests, same prior maps, same starts.
So the comparison is per (scene, layout, target, start) and a difference is a
difference in behaviour, not in sampling.

Mechanism counters come first and the success rate last, deliberately. On this
benchmark the noise floor is about three trials, so a +-3 swing in SR says
nothing on its own; what says something is whether the mechanism fired at all
and whether the thing it was supposed to change actually changed.

  python scripts/report_fusion_ab.py --run outputs/fusion_ab
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

ARMS = ("base", "fused")
COUNTERS = [
    "cross_floor_request_held",
    "cross_floor_search_requests",
    "directed_floor_switch_attempts",
    "floor_switch_attempts",
    "absence_checks",
    "absence_abandon",
    "stale_anchor_stop",
    "search_surface",
    "portals_seen",
]


def load(run: str, arm: str) -> dict:
    out = {}
    for path in sorted(glob.glob(os.path.join(run, arm, "*", "episodes.jsonl"))):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                episode = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[episode.get("episode_id")] = episode
    return out


def layout_of(episode: dict) -> str:
    authored = episode.get("authored_layout")
    if isinstance(authored, dict):
        return str(authored.get("layout_type") or "?")
    return str(episode.get("layout_type") or "?")


def first_step(episode: dict, key: str):
    """Step of the first search-log event carrying `key`."""
    for event in episode.get("search_log_events") or []:
        if key in event:
            return int(event.get("step", -1))
    return None


def first_absence(episode: dict):
    events = episode.get("presence_events") or []
    steps = [int(e.get("step", 10 ** 9)) for e in events if "absent_on_arrival" in str(e.get("cause", ""))]
    return min(steps) if steps else None


def summarise(episodes: dict, keys) -> dict:
    rows = [episodes[k] for k in keys if k in episodes]
    total = len(rows)
    if not total:
        return {}
    counters = Counter()
    fired = Counter()
    for episode in rows:
        stats = episode.get("agent_stats") or {}
        for name in COUNTERS:
            value = int(stats.get(name, 0) or 0)
            counters[name] += value
            fired[name] += int(value > 0)
    # Did a storey request come before the anchor was ever tested?
    premature = 0
    for episode in rows:
        req = first_step(episode, "selected_floor")
        absent = first_absence(episode)
        if req is not None and (absent is None or req < absent):
            premature += 1
    return {
        "episodes": total,
        "success": sum(int(bool(e.get("success"))) for e in rows),
        "spl": sum(float(e.get("spl") or 0.0) for e in rows) / total,
        "steps": sorted(int(e.get("steps") or 0) for e in rows)[total // 2],
        "at_budget": sum(int((e.get("steps") or 0) >= 500) for e in rows),
        "goal_floor_reached": sum(int(bool(e.get("goal_floor_reached"))) for e in rows),
        "premature_floor_request": premature,
        "counters": counters,
        "fired": fired,
    }


def pct(n, d):
    return f"{n}/{d} = {100.0 * n / d:.1f}%" if d else "-"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="outputs/fusion_ab")
    args = parser.parse_args()

    arms = {arm: load(args.run, arm) for arm in ARMS}
    shared = sorted(set(arms["base"]) & set(arms["fused"]))
    only = {arm: sorted(set(arms[arm]) - set(shared)) for arm in ARMS}
    print(f"# Fused pipeline A/B -- `{args.run}`\n")
    print(f"Paired episodes: {len(shared)}"
          + (f" (base-only {len(only['base'])}, fused-only {len(only['fused'])})"
             if only["base"] or only["fused"] else ""))
    if not shared:
        print("\nNo paired episodes yet.")
        return

    groups = defaultdict(list)
    for key in shared:
        episode = arms["base"][key]
        groups[(episode.get("floor_class") or "?", layout_of(episode))].append(key)
    groups["all", ""] = shared

    print("\n## 1. Outcome, paired\n")
    print("| split | episodes | base SR | fused SR | base SPL | fused SPL | base reached goal floor | fused |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in sorted(groups, key=lambda g: (g[0] == "all", g)):
        keys = groups[name]
        b, f = summarise(arms["base"], keys), summarise(arms["fused"], keys)
        if not b or not f:
            continue
        label = " ".join(x for x in name if x) or "all"
        print(f"| {label} | {b['episodes']} | {pct(b['success'], b['episodes'])} | "
              f"{pct(f['success'], f['episodes'])} | {b['spl']:.3f} | {f['spl']:.3f} | "
              f"{b['goal_floor_reached']} | {f['goal_floor_reached']} |")

    print("\n## 2. Mechanism counters (sum over paired episodes; episodes>0 in brackets)\n")
    b, f = summarise(arms["base"], shared), summarise(arms["fused"], shared)
    print("| counter | base | fused |")
    print("|---|---:|---:|")
    for name in COUNTERS:
        print(f"| {name} | {b['counters'][name]} [{b['fired'][name]}] | "
              f"{f['counters'][name]} [{f['fired'][name]}] |")
    print(f"| **storey request before any absence test** | {b['premature_floor_request']} | "
          f"{f['premature_floor_request']} |")
    print(f"| median steps | {b['steps']} | {f['steps']} |")
    print(f"| episodes at budget | {b['at_budget']} | {f['at_budget']} |")

    print("\n## 3. Episodes whose outcome changed\n")
    flipped = [(k, int(bool(arms['base'][k].get('success'))),
                int(bool(arms['fused'][k].get('success')))) for k in shared]
    flipped = [x for x in flipped if x[1] != x[2]]
    if not flipped:
        print("None.")
    else:
        print("| episode | floor class | base | fused |")
        print("|---|---|---:|---:|")
        for key, base_ok, fused_ok in flipped:
            print(f"| {key} | {arms['base'][key].get('floor_class')} | {base_ok} | {fused_ok} |")
        gained = sum(1 for x in flipped if x[2] > x[1])
        print(f"\nGained {gained}, lost {len(flipped) - gained}.")


if __name__ == "__main__":
    main()
