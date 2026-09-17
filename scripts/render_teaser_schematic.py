#!/usr/bin/env python3
"""Render the paper teaser: stale upstairs evidence and cross-floor recovery.

This is a schematic motivation figure, not an experiment trace or a system
diagram. The same house and cup are shown at three moments so the changed
object location, failed observation, and revised search are easy to compare.

    python3 scripts/render_teaser_schematic.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
INK = "#1e2933"
MUTED = "#64727c"
LINE = "#b9c6cb"
NAVY = "#20607d"
NAVY_LIGHT = "#e5f1f5"
RED = "#be4c42"
RED_LIGHT = "#faeeeb"
GREEN = "#267a59"
GREEN_LIGHT = "#e8f5eb"
CUP = "#da8650"
UP = "#eaf0f4"
DOWN = "#f5f0e8"


def rounded(ax, xy, width, height, *, face="white", edge="none", radius=0.03,
            lw=1.0, z=1):
    patch = FancyBboxPatch(xy, width, height,
                           boxstyle=f"round,pad=0,rounding_size={radius}",
                           facecolor=face, edgecolor=edge, linewidth=lw, zorder=z)
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, *, color=NAVY, lw=2.2, scale=12, rad=0, z=8,
          linestyle="solid"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                 mutation_scale=scale, color=color, lw=lw,
                                 linestyle=linestyle,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z))


def cup(ax, x, y, *, color=CUP, ghost=False, size=1.0):
    """A recognizable cup icon with a handle; x,y are its bottom centre."""
    w, h = 0.060 * size, 0.070 * size
    fill = "none" if ghost else color
    stroke = RED if ghost else "#8e5135"
    ax.add_patch(Polygon([(x - w * 0.48, y + h), (x + w * 0.48, y + h),
                          (x + w * 0.37, y), (x - w * 0.37, y)],
                         closed=True, facecolor=fill, edgecolor=stroke,
                         linewidth=1.4, linestyle="--" if ghost else "-", zorder=7))
    ax.add_patch(Ellipse((x, y + h), w, h * 0.22, facecolor=fill,
                         edgecolor=stroke, linewidth=1.3, zorder=8))
    ax.add_patch(Arc((x + w * 0.49, y + h * 0.55), w * 0.46, h * 0.52,
                     theta1=-90, theta2=95, color=stroke, lw=1.5,
                     linestyle="--" if ghost else "-", zorder=7))


def robot(ax, x, y, *, faded=False):
    color = "#90a7b1" if faded else NAVY
    ax.add_patch(Circle((x, y), 0.033, facecolor=color, edgecolor="white",
                        linewidth=1.2, zorder=9))
    ax.add_patch(Circle((x + 0.012, y + 0.007), 0.006,
                        facecolor="white", edgecolor="none", zorder=10))
    ax.plot([x - 0.008, x - 0.018], [y + 0.030, y + 0.049],
            color=color, lw=1.7, zorder=8)
    ax.add_patch(Circle((x - 0.019, y + 0.050), 0.0045,
                        facecolor=color, zorder=9))


def cross(ax, x, y, *, color=RED, size=0.019, lw=2.4, z=10):
    ax.plot([x - size, x + size], [y - size, y + size], color=color, lw=lw, zorder=z)
    ax.plot([x - size, x + size], [y + size, y - size], color=color, lw=lw, zorder=z)


def house(ax):
    """Shared two-storey cutaway with bedroom, kitchen, and visible staircase."""
    rounded(ax, (0.065, 0.280), 0.870, 0.555, face="white", edge=LINE,
            radius=0.020, lw=1.2)
    ax.add_patch(Rectangle((0.071, 0.580), 0.858, 0.247,
                           facecolor=UP, edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((0.071, 0.288), 0.858, 0.286,
                           facecolor=DOWN, edgecolor="none", zorder=2))
    # A thick slab separates the floors; the open centre is a stairwell.
    ax.plot([0.072, 0.565], [0.578, 0.578], color="#718590", lw=3.0, zorder=5)
    ax.plot([0.810, 0.928], [0.578, 0.578], color="#718590", lw=3.0, zorder=5)
    ax.plot([0.565, 0.565], [0.580, 0.802], color="#c3ced2", lw=1.2, zorder=4)
    ax.plot([0.810, 0.810], [0.305, 0.798], color="#c3ced2", lw=1.2, zorder=4)
    # Bed and pillow upstairs.
    rounded(ax, (0.200, 0.630), 0.285, 0.045, face="#aab9c2",
            radius=0.012, z=5)
    rounded(ax, (0.212, 0.664), 0.095, 0.026, face="white", edge="#97aab5",
            radius=0.008, lw=0.9, z=6)
    ax.plot([0.221, 0.221], [0.630, 0.611], color="#8095a0", lw=2, zorder=5)
    ax.plot([0.465, 0.465], [0.630, 0.611], color="#8095a0", lw=2, zorder=5)
    # Kitchen worktop downstairs, with cabinet doors.
    rounded(ax, (0.182, 0.382), 0.316, 0.062, face="#ceb89c",
            radius=0.006, z=5)
    ax.plot([0.182, 0.498], [0.444, 0.444], color="#9f876e", lw=2.0, zorder=6)
    ax.plot([0.339, 0.339], [0.387, 0.438], color="#ae9a80", lw=1.1, zorder=6)
    for knob_x in (0.280, 0.401):
        ax.add_patch(Circle((knob_x, 0.412), 0.0035,
                            facecolor="#7c6b58", zorder=6))
    # Steps run from the upstairs landing to the lower floor.
    for i in range(5):
        xx = 0.588 + i * 0.039
        yy = 0.558 - i * 0.049
        ax.plot([xx, xx + 0.043], [yy, yy], color="#82939a", lw=2.0, zorder=5)
        ax.plot([xx + 0.043, xx + 0.043], [yy, yy - 0.049],
                color="#82939a", lw=1.6, zorder=5)
    ax.text(0.128, 0.775, "UPSTAIRS  ·  BEDROOM", fontsize=8.3,
            fontweight="bold", color="#405761", zorder=7)
    ax.text(0.128, 0.550, "DOWNSTAIRS  ·  KITCHEN", fontsize=8.3,
            fontweight="bold", color="#776550", zorder=7)
    ax.text(0.684, 0.315, "stairs", fontsize=7.5, color=MUTED,
            ha="center", zorder=7)


def graph_card(ax, left, relation, right, *, state="valid", title="3D scene graph"):
    color = RED if state == "stale" else GREEN if state == "updated" else NAVY
    face = RED_LIGHT if state == "stale" else GREEN_LIGHT if state == "updated" else NAVY_LIGHT
    rounded(ax, (0.067, 0.064), 0.865, 0.175, face=face,
            edge="#d8e1e3", radius=0.021, lw=0.9, z=2)
    ax.text(0.101, 0.209, title.upper(), fontsize=7.4,
            fontweight="bold", color=color, va="center", zorder=4)
    node_y = 0.129
    for cx, label, width in ((0.240, left, 0.240), (0.759, right, 0.175)):
        rounded(ax, (cx - width / 2, node_y - 0.030), width, 0.059,
                face="white", edge=color, radius=0.018, lw=1.0, z=4)
        ax.text(cx, node_y, label, ha="center", va="center", fontsize=8.6,
                color=INK, fontweight="bold", zorder=5)
    ax.plot([0.361, 0.670], [node_y, node_y], color=color, lw=2.0,
            linestyle="--" if state == "stale" else "-", zorder=3)
    ax.text(0.514, node_y + 0.018, relation, ha="center", fontsize=7.8,
            color=color, fontweight="bold", zorder=5,
            bbox=dict(facecolor=face, edgecolor="none", pad=1.5))
    if state == "stale":
        cross(ax, 0.520, node_y - 0.010, size=0.018, z=7)


def title(ax, number, heading, detail, color):
    ax.add_patch(Circle((0.089, 0.926), 0.033, facecolor=color, zorder=5))
    ax.text(0.089, 0.926, str(number), ha="center", va="center",
            fontsize=10, color="white", fontweight="bold", zorder=6)
    ax.text(0.140, 0.940, heading, fontsize=12.8, fontweight="bold",
            color=INK, va="center", zorder=5)
    ax.text(0.140, 0.890, detail, fontsize=8.6, color=MUTED,
            va="center", zorder=5)


def prior(ax):
    title(ax, 1, "Prior knowledge", "Cup observed in the upstairs bedroom", NAVY)
    house(ax)
    cup(ax, 0.390, 0.692)
    robot(ax, 0.510, 0.714)
    ax.plot([0.480, 0.431], [0.719, 0.731], color=NAVY,
            lw=1.3, alpha=0.70, zorder=7)
    ax.plot([0.479, 0.430], [0.697, 0.709], color=NAVY,
            lw=1.3, alpha=0.70, zorder=7)
    graph_card(ax, "bedroom", "contains", "cup")


def change(ax):
    title(ax, 2, "Dynamic cross-floor change",
          "The cup moves; the old location is empty", RED)
    house(ax)
    cup(ax, 0.390, 0.692, ghost=True)
    cross(ax, 0.391, 0.734, size=0.030)
    cup(ax, 0.390, 0.449)
    robot(ax, 0.505, 0.712)
    # A static or single-floor policy keeps searching the upstairs room.
    arrow(ax, (0.507, 0.760), (0.283, 0.756), color=RED,
          lw=2.1, scale=13, rad=0.40)
    ax.text(0.618, 0.754, "empty on arrival", fontsize=8.0,
            color=RED, fontweight="bold", zorder=9)
    rounded(ax, (0.070, 0.248), 0.865, 0.041, face=RED_LIGHT,
            radius=0.012, z=6)
    ax.text(0.502, 0.269, "STATIC / SINGLE-FLOOR SEARCH: continues upstairs",
            ha="center", va="center", fontsize=7.9, color=RED,
            fontweight="bold", zorder=7)
    graph_card(ax, "bedroom", "contains", "cup", state="stale",
               title="Outdated map")


def recovery(ax):
    title(ax, 3, "Recovery", "Explore, revise the belief, then change floors", GREEN)
    house(ax)
    cup(ax, 0.390, 0.692, ghost=True)
    cross(ax, 0.390, 0.731, color=GREEN, size=0.027)
    ax.text(0.617, 0.778, "invalidate old location", fontsize=7.8,
            color=GREEN, fontweight="bold", zorder=9)
    cup(ax, 0.390, 0.449)
    ax.add_patch(Circle((0.390, 0.460), 0.065, facecolor="none",
                        edgecolor=GREEN, linewidth=2.0, zorder=8))
    ax.text(0.390, 0.310, "kitchen candidate", ha="center", fontsize=8.0,
            color=GREEN, fontweight="bold", zorder=9)
    robot(ax, 0.527, 0.712)
    # Explicit route: across the upstairs landing, down the stairwell,
    # then back to the candidate on the lower storey.
    route = [(0.555, 0.700), (0.620, 0.637), (0.645, 0.564),
             (0.684, 0.515), (0.721, 0.466), (0.747, 0.411),
             (0.595, 0.350), (0.425, 0.393)]
    ax.plot([p[0] for p in route], [p[1] for p in route], color=GREEN,
            lw=3.0, solid_capstyle="round", zorder=8)
    arrow(ax, route[-2], route[-1], color=GREEN, lw=2.5, scale=16, z=9)
    rounded(ax, (0.070, 0.248), 0.865, 0.041, face=GREEN_LIGHT,
            radius=0.012, z=6)
    ax.text(0.502, 0.269, "BELIEF REVISION  ·  CROSS-FLOOR RECOVERY AFTER EXPLORATION",
            ha="center", va="center", fontsize=7.45, color=GREEN,
            fontweight="bold", zorder=7)
    graph_card(ax, "kitchen", "contains", "cup", state="updated",
               title="Updated scene graph")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "docs/figures/teaser_schematic.pdf")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.35),
                             gridspec_kw={"wspace": 0.035})
    for ax, draw, tint in zip(axes, (prior, change, recovery),
                              ("#f6fafb", "#fff9f7", "#f7fbf8")):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_axis_off()
        rounded(ax, (0.005, 0.018), 0.990, 0.970, face=tint,
                edge="#d9e1e2", radius=0.028, lw=1.0, z=0)
        draw(ax)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.008, right=0.992, top=0.995, bottom=0.005)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".svg", ".png"):
        target = args.out.with_suffix(ext)
        fig.savefig(target, dpi=args.dpi, facecolor="white")
        print(f"wrote {target}")
    plt.close(fig)


if __name__ == "__main__":
    main()
