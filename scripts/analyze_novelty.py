#!/usr/bin/env python3
"""The observation-novelty arm: did the knob move, and did it help?

Two questions, in this order, because the second is meaningless without the
first. This codebase has shipped arms that returned confident nulls while
testing nothing, so the report leads with the mechanism counters and refuses to
call a difference a result when `novelty_reordered` is zero.

    python scripts/analyze_novelty.py CONTROL=outputs/<ts> TREATMENT=outputs/<ts>

With a single run and no treatment it reports the calibration instead: the
distribution of `novelty_n_max`, which is the scale sigma has to be set against.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def load(path: Path) -> list:
    f = path / "episodes.jsonl"
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


def stat(eps: list, key: str) -> list:
    return [e.get("agent_stats", {}).get(key, 0) or 0 for e in eps]


def half(eps: list, kind: str) -> list:
    return [e for e in eps if kind in str(e.get("episode_id", ""))]


def sr(eps: list) -> float:
    return sum(1 for e in eps if e.get("success")) / len(eps) if eps else float("nan")


def close_rate(eps: list, thresh: float = 1.25) -> float:
    """P(the agent got within `thresh` of the target) -- the measurement the
    campaign settled on: cross_anchor SR is this quantity and almost nothing
    else (P(success|close) 0.895 vs 0.121 beyond, corr 0.963 over 9 conditions).
    """
    seen = [e for e in eps if e.get("gt_min_range_any_m") is not None]
    if not seen:
        return float("nan")
    return sum(1 for e in seen if e["gt_min_range_any_m"] <= thresh) / len(seen)


def calibration(eps: list) -> None:
    n = sorted(x for x in stat(eps, "novelty_n_max") if x)
    print(f"  episodes with a measured scale : {len(n)}/{len(eps)}")
    if not n:
        print("  novelty_n_max is zero everywhere -- the surface queue never ran,")
        print("  so this run cannot calibrate sigma.")
        return
    q = lambda p: n[min(len(n) - 1, int(p * len(n)))]
    print(f"  novelty_n_max  min {n[0]:.1f}  q1 {q(.25):.1f}  med {q(.5):.1f} "
          f" q3 {q(.75):.1f}  p90 {q(.9):.1f}  max {n[-1]:.1f}")
    print(f"  -> sigma at the median puts a typical worked-over room at 1/e;")
    print(f"     sigma = {q(.5):.0f} is the calibrated choice, "
          f"floor binds past {q(.5) * 1.386:.0f} observations at floor 0.25.")


def report(name: str, eps: list) -> None:
    ia, ca = half(eps, "in_anchor"), half(eps, "cross_anchor")
    print(f"\n{name}: {len(eps)} episodes")
    print(f"  SR overall {sr(eps):.3f}   in_anchor {sr(ia):.3f} ({len(ia)})"
          f"   cross_anchor {sr(ca):.3f} ({len(ca)})")
    print(f"  P(close<=1.25m) overall {close_rate(eps):.3f}"
          f"   cross_anchor {close_rate(ca):.3f}")
    applied = stat(eps, "novelty_applied")
    reord = stat(eps, "novelty_reordered")
    fired = sum(1 for x in reord if x)
    print(f"  MECHANISM: reordered the queue in {fired}/{len(eps)} episodes"
          f"  (median {statistics.median(reord):.0f} rounds, "
          f"total applications {sum(applied)})")
    # `stat` reads a missing counter as 0; a run that predates the
    # instrumentation must not be reported as having applied a weight of zero.
    mins = [x for x in stat(eps, "novelty_min") if 0.0 < x < 1.0]
    if mins:
        print(f"             weakest weight applied: {min(mins):.3f}"
              f"  (median of those that bit: {statistics.median(mins):.3f})")
    calibration(eps)


def main() -> int:
    args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    if not args:
        print(__doc__)
        return 2
    runs = {k: load(Path(v)) for k, v in args.items()}
    for k, eps in runs.items():
        report(k, eps)

    if "CONTROL" in runs and "TREATMENT" in runs:
        c, t = runs["CONTROL"], runs["TREATMENT"]
        fired = sum(1 for x in stat(t, "novelty_reordered") if x)
        print("\n" + "=" * 68)
        if fired == 0:
            print("VOID, not a null: the weight never changed a ranking decision.")
            print("Whatever the SR difference is, this run did not test the idea.")
            return 0
        for label, sel in (("overall", lambda e: e),
                           ("in_anchor", lambda e: half(e, "in_anchor")),
                           ("cross_anchor", lambda e: half(e, "cross_anchor"))):
            d = sr(sel(t)) - sr(sel(c))
            n = len(sel(c))
            print(f"  {label:13s} {sr(sel(c)):.3f} -> {sr(sel(t)):.3f} "
                  f"({d:+.3f}, {d * n:+.1f} episodes of {n})")
        d = close_rate(half(t, "cross_anchor")) - close_rate(half(c, "cross_anchor"))
        print(f"  P(close) cross {close_rate(half(c, 'cross_anchor')):.3f} -> "
              f"{close_rate(half(t, 'cross_anchor')):.3f} ({d:+.3f})")
        print("  Read P(close) first: it is what cross_anchor SR is made of.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
