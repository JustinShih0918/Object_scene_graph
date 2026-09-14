"""Draw the stored ASCENT map with the episode's target marked on it.

The cross-anchor failures are geometric, and a table of coordinates does not
settle a geometric question. This renders what the agent was actually planning
over -- the occupancy and stair mask pasted out of `ycb.obstacle_map_in` -- and
puts on it the four points that decide the episode:

    *  the TARGET the episode is scored against (`target_obj_xy`)
    *  where the agent finished (`final_xy`)
    *  the flight foot it drove to (`portal_log`)
    *  the staircase ASCENT itself recorded (`_up_stair_start`/`_end`)

If the flight foot and ASCENT's staircase disagree, the ramp is wrong; if they
agree and there is no staircase there on screen, the paste is wrong.

    python scripts/plot_cross_anchor_map.py \\
        --maps outputs/maps_mf5_ascent --scene 00800-TEEsavR23oF \\
        --run outputs/mf5_pass2_v4 --out /tmp/cross_anchor.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from navigation.mapping.map_store import (_anchor_from, _px_to_world,
                                          apply_to_costmap, load_obstacle_maps)
from osg.mapping.costmap import FREE, OCCUPIED, UNKNOWN, Costmap2D


def _render(costmap):
    """Grid -> RGB. White free, black occupied, grey unknown, orange stairs."""
    g = costmap.grid
    img = np.full(g.shape + (3,), 0.72, dtype=float)      # unknown
    img[g == FREE] = (1.0, 1.0, 1.0)
    img[g == OCCUPIED] = (0.15, 0.15, 0.18)
    mask = getattr(costmap, "stair_mask", None)
    if mask is not None:
        img[mask] = (0.95, 0.55, 0.10)                    # stairs
    return img


def _extent(costmap):
    rows, cols = costmap.grid.shape
    x0, z0 = float(costmap.origin[0]), float(costmap.origin[1])
    return x0, x0 + rows * costmap.resolution, z0, z0 + cols * costmap.resolution


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heights", default="", help="comma-separated storey heights, bottom-up")
    args = ap.parse_args()

    blob = load_obstacle_maps(str(Path(args.maps) / f"{args.scene}.json"))
    npz = np.load(Path(args.maps) / f"{args.scene}.npz")
    anchor = _anchor_from(blob)
    episodes = [json.loads(l) for l in
                (Path(args.run) / "episodes.jsonl").read_text().splitlines() if l.strip()]

    stored = [f for f in blob["floors"] if f.get("explored_cells", 0) > 0]
    # Storey heights: from the run's own floor_log unless overridden.
    if args.heights:
        heights = [float(v) for v in args.heights.split(",")]
    else:
        seen = sorted({round(float(r[2]), 3) for e in episodes for r in (e.get("floor_log") or [])})
        heights = seen or [0.0]
    print(f"storeys stored: {[f['index'] for f in stored]}   heights used: {heights}")

    fig, axes = plt.subplots(1, len(stored), figsize=(9 * len(stored), 9))
    axes = np.atleast_1d(axes)
    for ax, record in zip(axes, stored):
        i = int(record["index"])
        lo = heights[min(len(heights) - 1, stored.index(record))]
        nxt = heights[stored.index(record) + 1] if stored.index(record) + 1 < len(heights) \
            else (heights[stored.index(record) - 1] if len(heights) > 1 else None)
        cm = Costmap2D(resolution=float(record["resolution"]), size_m=60.0)
        cm.height = np.full(cm.grid.shape, np.nan, dtype=np.float32)
        cells = apply_to_costmap(blob, i, cm, floor_y=lo, next_floor_y=nxt)
        ax.imshow(np.transpose(_render(cm), (1, 0, 2)), origin="lower",
                  extent=_extent(cm), interpolation="nearest")

        size = int(record["size"])
        epo = np.asarray(record["episode_pixel_origin"], float)
        ppm = float(record["pixels_per_meter"])
        for name, colour in (("_up_stair_start", "#1b9e77"), ("_up_stair_end", "#d95f02"),
                             ("_down_stair_start", "#d95f02"), ("_down_stair_end", "#1b9e77")):
            key = record["prefix"] + name
            if key not in npz.files:
                continue
            px = np.asarray(npz[key], float).ravel()
            if px.size < 2:
                continue
            w = _px_to_world(px[1], px[0], size, epo, ppm, anchor)   # stored (col, row)
            ax.plot(*w, marker="P", ms=15, mfc=colour, mec="k", mew=1.4, ls="none",
                    label=f"ASCENT {name.strip('_').replace('_', ' ')}")

        for e in episodes:
            # A marker belongs on the storey it is ON, and the two numbering
            # schemes disagree: the run's `prior_obstacle_map.matched_floors`
            # is what says which ASCENT floor is which OSG floor. Drawing every
            # episode on every panel makes a target look reachable when it is
            # one storey away, which is the whole failure here.
            m = {int(r["osg_floor"]): int(r["ascent_floor"])
                 for r in (e.get("prior_obstacle_map") or {}).get("matched_floors", [])}
            goal_panel = m.get(int(e.get("goal_floor", -1)))
            here_panel = m.get(int(e.get("start_floor", -1)))
            tgt = e.get("target_obj_xy")
            if tgt and goal_panel == i:
                ax.plot(tgt[0], tgt[1], marker="*", ms=30, mfc="#e41a1c", mec="k",
                        mew=1.6, ls="none", label=f"TARGET {e['target']}")
                ax.annotate(f"   {e['target']}", (tgt[0], tgt[1]), color="#e41a1c",
                            fontsize=13, weight="bold")
            fin = e.get("final_xy")
            if fin and here_panel == i:
                ax.plot(fin[0], fin[1], marker="X", ms=16, mfc="#377eb8", mec="k",
                        mew=1.4, ls="none", label=f"agent ended ({e['target']})")
                ax.annotate(f"   ended: {e['target']}", (fin[0], fin[1]),
                            color="#377eb8", fontsize=10)
            for step, foot, kind, _n in (e.get("portal_log") or []):
                if here_panel != i:
                    continue
                ax.plot(foot[0], foot[1], marker="o", ms=15, mfc="none", mec="#984ea3",
                        mew=3.2, ls="none", label=f"flight {kind} @{step}")

        ax.set_title(f"ASCENT floor {i}  (storey y={lo})   {cells} cells pasted",
                     fontsize=13)
        ax.set_xlabel("world x (m)"); ax.set_ylabel("world z (m)")
        ax.grid(alpha=0.25, ls=":")
        handles, labels = ax.get_legend_handles_labels()
        seen_l, keep = set(), []
        for h, l in zip(handles, labels):
            if l not in seen_l:
                seen_l.add(l); keep.append((h, l))
        ax.legend(*zip(*keep), loc="upper left", fontsize=9, framealpha=0.92)
        # Crop to what was actually pasted, with a margin.
        known = np.argwhere(cm.grid != UNKNOWN)
        if known.size:
            xs = cm.origin[0] + known[:, 0] * cm.resolution
            zs = cm.origin[1] + known[:, 1] * cm.resolution
            ax.set_xlim(xs.min() - 2, xs.max() + 2)
            ax.set_ylim(zs.min() - 2, zs.max() + 2)

    fig.suptitle(
        "orange = stairs carried from ASCENT   white = free   black = obstacle   grey = never mapped",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=115, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
