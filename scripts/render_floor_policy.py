#!/usr/bin/env python3
"""The floor-switch policy, end to end: when it is asked, whether it says yes,
which portal it picks, and what keeps the agent driving to it.

Same visual language and the same words as `render_architecture.py`. Every
threshold printed is the resolved value of the shipped configuration, not a
default read off the dataclass.

    python scripts/render_floor_policy.py
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

WORKSPACE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#16202a"
MUTED = "#5d6b76"
BAND_A = "#eef4f7"
BAND_B = "#f4f1fa"
BAND_C = "#fdf4ec"
C_FLOOR = "#0d7c8c"
C_CONT = "#b8471f"
C_FRONT = "#15637f"
C_GO = "#1f7a34"
C_STAY = "#a12b22"


def box(ax, x, y, w, h, text, *, fc="white", ec=INK, ls="--", lw=1.1, fs=9.0,
        weight="normal", tc=None, align="center"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.010",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=3))
    tx = x + w / 2 if align == "center" else x + 0.010
    ax.text(tx, y + h / 2, text, ha=align, va="center", fontsize=fs,
            color=tc or INK, zorder=4, linespacing=1.35, fontweight=weight)
    return (x, y, w, h)


def arrow(ax, a, b, *, color=MUTED, lw=1.2, rad=0.0, side="e2w"):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    pts = {
        "e2w": ((ax1 + aw, ay1 + ah / 2), (bx1, by1 + bh / 2)),
        "s2n": ((ax1 + aw / 2, ay1), (bx1 + bw / 2, by1 + bh)),
        "n2s": ((ax1 + aw / 2, ay1 + ah), (bx1 + bw / 2, by1)),
        "e2n": ((ax1 + aw, ay1 + ah / 2), (bx1 + bw / 2, by1 + bh)),
    }
    p, q = pts[side]
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=lw, shrinkA=1.5, shrinkB=2.5,
                                 connectionstyle=f"arc3,rad={rad}", zorder=2))


def chip(ax, x, y, text, color):
    ax.add_patch(FancyBboxPatch((x, y - 0.014), 0.052, 0.028,
                                boxstyle="round,pad=0.002,rounding_size=0.012",
                                facecolor=color, edgecolor="none", zorder=5))
    ax.text(x + 0.026, y, text, ha="center", va="center", fontsize=7.4,
            color="white", fontweight="bold", zorder=6)


def rule(ax, y, cond, thresh, verdict, color, *, x=0.055, w=0.62):
    ax.add_patch(FancyBboxPatch((x, y - 0.024), w, 0.048,
                                boxstyle="round,pad=0.002,rounding_size=0.008",
                                facecolor="white", edgecolor="#d8dfe4",
                                linewidth=0.9, zorder=3))
    ax.text(x + 0.014, y, cond, ha="left", va="center", fontsize=8.6,
            color=INK, zorder=4)
    ax.text(x + w - 0.075, y, thresh, ha="right", va="center", fontsize=8.0,
            color=MUTED, family="monospace", zorder=4)
    chip(ax, x + w - 0.062, y, verdict, color)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section-scene", default="00808-y9hTuugGdiq")
    ap.add_argument("--out", default=str(WORKSPACE / "outputs/figures/floor_policy.pdf"))
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    fig = plt.figure(figsize=(13.6, 8.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_axis_off()

    # ------------------------------------------------- A: when it is asked
    ax.add_patch(FancyBboxPatch((0.012, 0.796), 0.976, 0.190,
                                boxstyle="round,pad=0.004,rounding_size=0.010",
                                facecolor=BAND_A, edgecolor="none", zorder=0))
    ax.text(0.5, 0.962, "1 - When the policy is asked", ha="center", fontsize=12,
            fontweight="bold", style="italic", color=INK, zorder=4)

    t1 = box(ax, 0.036, 0.836, 0.238, 0.092,
             "Posterior names another storey\n"
             r"$\mathrm{argmax}_k\ \sum b \cdot d$  is not this floor",
             ec=C_CONT, tc=C_CONT, fs=8.6)
    t2 = box(ax, 0.306, 0.836, 0.238, 0.092,
             "Best frontier here is far\nfloor_switch(best.path_cost)",
             ec=C_FRONT, tc=C_FRONT, fs=8.6)
    t3 = box(ax, 0.576, 0.836, 0.238, 0.092,
             "No frontier at all on this floor\nfloor_switch(None)",
             ec=C_FRONT, tc=C_FRONT, fs=8.6)
    ax.text(0.155, 0.818, "DIRECTED  (target floor known)", ha="center",
            fontsize=7.4, color=C_CONT, fontweight="bold", zorder=4)
    ax.text(0.560, 0.818, "UNDIRECTED  (only 'nothing near left here')",
            ha="center", fontsize=7.4, color=C_FRONT, fontweight="bold", zorder=4)

    ax.add_patch(FancyArrowPatch((0.155, 0.836), (0.300, 0.716), arrowstyle="-|>",
                                 mutation_scale=11, color=C_CONT, lw=1.3,
                                 connectionstyle="arc3,rad=0.12", zorder=2))
    ax.text(0.208, 0.766, "skips the gate", fontsize=7.8, color=C_CONT,
            style="italic", zorder=4,
            path_effects=[matplotlib.patheffects.withStroke(linewidth=2.4,
                                                            foreground="white")])
    for x in (0.425, 0.695):
        ax.add_patch(FancyArrowPatch((x, 0.836), (0.500, 0.762), arrowstyle="-|>",
                                     mutation_scale=11, color=C_FRONT, lw=1.2,
                                     connectionstyle="arc3,rad=-0.06", zorder=2))

    # ------------------------------------------------- B: the gate
    ax.add_patch(FancyBboxPatch((0.012, 0.288), 0.700, 0.474,
                                boxstyle="round,pad=0.004,rounding_size=0.010",
                                facecolor=BAND_B, edgecolor="none", zorder=0))
    ax.text(0.362, 0.738, "2 - Whether to leave  ·  SwitchPolicy.may_switch",
            ha="center", fontsize=12, fontweight="bold", style="italic",
            color=INK, zorder=4)
    ax.text(0.362, 0.712, "read top to bottom; the first row that fires decides",
            ha="center", fontsize=7.8, color=MUTED, style="italic", zorder=4)

    rows = [
        ("Too late in the episode", "step > 350  (0.7 x 500)", "STAY", C_STAY),
        ("Switched too recently", "step - last < 50", "STAY", C_STAY),
        ("The target itself is mapped here", "evidence >= 10", "STAY", C_STAY),
        ("Looked here, and it shows no sign\nof the target's usual companions",
         "n_obj >= 8, step >= 30, evidence < 2", "GO", C_GO),
        ("Looked here, looks right, but a long\nsearch has turned up nothing",
         "steps_on_floor >= 120", "GO", C_GO),
        ("Looks like the target's kind of floor", "evidence >= 2, not stale", "STAY", C_STAY),
        ("Too early", "step < 50", "STAY", C_STAY),
        ("Nothing near left on this floor", "cost is None or > 4.0 m", "GO", C_GO),
    ]
    y = 0.676
    for cond, thresh, verdict, color in rows:
        h = 0.048 if "\n" in cond else 0.036
        rule(ax, y - h / 2, cond, thresh, verdict, color)
        y -= h + 0.005

    ax.text(0.055, 0.302, "evidence = distinct context categories of the target mapped "
            "on this floor, +10 if the target category itself is here "
            "(graph/priors.floor_target_evidence)",
            fontsize=7.4, color=MUTED, style="italic", zorder=4, ha="left")

    # ------------------------------------------------- C: portal + pursuit
    ax.add_patch(FancyBboxPatch((0.728, 0.288), 0.260, 0.474,
                                boxstyle="round,pad=0.004,rounding_size=0.010",
                                facecolor=BAND_C, edgecolor="none", zorder=0))
    ax.text(0.858, 0.738, "3 - Which portal", ha="center", fontsize=12,
            fontweight="bold", style="italic", color=INK, zorder=4)

    p1 = box(ax, 0.748, 0.640, 0.220, 0.070,
             "find_portals\nheight deltas in the costmap", ec=C_FLOOR, tc=C_FLOOR, fs=8.2)
    p2 = box(ax, 0.748, 0.536, 0.220, 0.082,
             "DIRECTED   keep the ones going\nthe right way, nearest target height\n"
             "UNDIRECTED   prefer a storey never\nvisited, then nearest", fs=7.6)
    p3 = box(ax, 0.748, 0.446, 0.220, 0.066, "Reachability check\non the chosen portal", fs=8.2)
    p4 = box(ax, 0.748, 0.334, 0.220, 0.074,
             "PortalGoal\ngoal_xy, target_y, deadline 120", ls="-", lw=1.5,
             ec=C_FLOOR, tc=C_FLOOR, fs=8.4, weight="bold")
    arrow(ax, p1, p2, side="s2n"); arrow(ax, p2, p3, side="s2n")
    arrow(ax, p3, p4, side="s2n")
    ax.text(0.858, 0.300, "no portal, or unreachable  ->  stay", ha="center",
            fontsize=7.4, color=MUTED, style="italic", zorder=4)

    # ------------------------------------------------- D: pursuit
    ax.add_patch(FancyBboxPatch((0.012, 0.014), 0.976, 0.252,
                                boxstyle="round,pad=0.004,rounding_size=0.010",
                                facecolor=BAND_C, edgecolor="none", zorder=0))
    ax.text(0.5, 0.244, "4 - Getting there  ·  pursuit_ok, every step",
            ha="center", fontsize=12, fontweight="bold", style="italic",
            color=INK, zorder=4)

    d1 = box(ax, 0.036, 0.100, 0.148, 0.086,
             "state = GOTO_FRONTIER\ndeadline = step + 120", fs=8.2)
    d2 = box(ax, 0.212, 0.100, 0.166, 0.086,
             "Vertical progress?\n|cam_y - start_y| >= 0.25 m\nOR on_stairs", fs=8.2)
    d3 = box(ax, 0.406, 0.100, 0.148, 0.086, "Keep driving\nEXPLORE is skipped",
             ec=C_GO, tc=C_GO, fs=8.4)
    d4 = box(ax, 0.408, 0.020, 0.144, 0.058,
             "grace 100 steps,\nthen give up", ec=C_STAY, tc=C_STAY, fs=8.2)
    d5 = box(ax, 0.582, 0.100, 0.162, 0.086,
             "On stairs\ncostmap update OFF\nstair cells -> traversable", ec=C_FLOOR,
             tc=C_FLOOR, fs=8.2)
    d6 = box(ax, 0.772, 0.100, 0.190, 0.086,
             "FloorPolicy.observe\nnew floor id, new floor_y\nFloorStack swaps the layer",
             ls="-", lw=1.5, ec=C_FLOOR, tc=C_FLOOR, fs=8.2, weight="bold")
    arrow(ax, d1, d2); arrow(ax, d2, d3); arrow(ax, d3, d5); arrow(ax, d5, d6)
    arrow(ax, d2, d4, side="e2n", color=C_STAY, rad=-0.2)
    ax.text(0.300, 0.076, "no", fontsize=7.6, color=C_STAY, zorder=5, ha="center",
            path_effects=[matplotlib.patheffects.withStroke(linewidth=2.2,
                                                            foreground="white")])
    ax.text(0.782, 0.046,
            "horizontal displacement is NOT the test: on a switchback staircase\n"
            "15 steps of good climbing can move (x, z) barely at all",
            ha="center", fontsize=7.4, color=MUTED, style="italic", zorder=4)

    ax.add_patch(FancyArrowPatch((0.858, 0.334), (0.110, 0.186), arrowstyle="-|>",
                                 mutation_scale=12, color=C_FLOOR, lw=1.3,
                                 linestyle=(0, (5, 3)),
                                 connectionstyle="arc3,rad=0.16", zorder=2))
    ax.text(0.500, 0.286, "commit", fontsize=8.0, color=C_FLOOR, style="italic",
            ha="center", zorder=5,
            path_effects=[matplotlib.patheffects.withStroke(linewidth=2.6,
                                                            foreground="white")])

    ax.text(0.012, 0.004,
            "Shipped DualMap-protocol arm runs with floor.cross_floor = false; "
            "the released benchmark is single-storey.",
            fontsize=7.6, color=MUTED, style="italic", zorder=4, ha="left")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out}\nwrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
