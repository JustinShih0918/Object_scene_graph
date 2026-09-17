#!/usr/bin/env python3
"""How a staircase is recognised -- a schematic, drawn rather than measured.

The argument in the method is geometric, not empirical: it is about where the
height step between neighbouring cells falls relative to flat floor and to a
wall, and about why the cells you may STEP on are not the same set as the cells
that tell you where the staircase RUNS. A cross-section says that in one look.

Nothing here is data. It is a diagram of the criterion, and is labelled as one.

    python scripts/render_stair_schematic.py
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

INK = "#16202a"
MUTED = "#5d6b76"
SLAB = "#d8dee3"
SLAB_EDGE = "#9aa4ad"
C_STEP = "#e8952f"      # the steppable band / the stair mask
C_FLIGHT = "#15637f"    # the flight
C_FOOT = "#c02a22"
C_WALL = "#7a8791"

RISER, TREAD, N = 0.25, 0.30, 12          # 12 x 0.25 m = 3.0 m between storeys
RISE = RISER * N
RUN = TREAD * N


def stair_profile():
    """(x, y) of the stepped surface, bottom landing to top landing."""
    xs, ys = [-1.8], [0.0]
    for k in range(N):
        x = k * TREAD
        xs += [x, x]
        ys += [k * RISER, (k + 1) * RISER]
    xs += [RUN, RUN + 1.8]
    ys += [RISE, RISE]
    return np.array(xs), np.array(ys)


def label(ax, xy, text, *, color=INK, fs=8.4, weight="normal", ha="center", va="center"):
    ax.annotate(text, xy, ha=ha, va=va, fontsize=fs, color=color, fontweight=weight,
                zorder=9, path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])


def draw_section(ax):
    xs, ys = stair_profile()

    # The solid body under the stepped surface, and the two landings.
    ax.fill_between(xs, ys, -0.9, color=SLAB, edgecolor="none", zorder=1)
    ax.plot(xs, ys, color=SLAB_EDGE, lw=1.6, zorder=3)

    # A wall at the far end, so the "wall" regime has something to point at.
    ax.add_patch(Rectangle((RUN + 1.5, RISE), 0.30, 2.1, facecolor=C_WALL,
                           edgecolor="none", zorder=2))

    # Two storey levels.
    for y, name in ((0.0, "storey $\\varphi$"), (RISE, "storey $\\varphi'$")):
        ax.axhline(y, color=INK, lw=1.0, ls=(0, (6, 4)), alpha=0.55, zorder=2)
        label(ax, (-1.72, y + 0.17), name, fs=8.6, weight="bold", ha="left")

    # The height layer: one sample per cell, on whatever surface was seen.
    step = TREAD / 2.0
    samples = []
    for x in np.arange(-1.7, -0.05, step):
        samples.append((x, 0.0))
    for k in range(N):
        for j in range(2):
            samples.append((k * TREAD + step * (j + 0.5) * 0.9, (k + 1) * RISER))
    for x in np.arange(RUN + 0.1, RUN + 1.5, step):
        samples.append((x, RISE))
    sx, sy = np.array(samples).T
    ax.plot(sx, sy, "o", ms=3.2, color=INK, alpha=0.75, zorder=5,
            markeredgecolor="white", markeredgewidth=0.5)
    label(ax, (-1.0, -0.55), "one height sample per cell", fs=7.8, color=MUTED)

    # --- the three regimes of the neighbour height step -------------------
    def dh_arrow(x, y0, y1, text, color):
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="<|-|>",
                                     mutation_scale=9, color=color, lw=1.7, zorder=8))
        label(ax, (x + 0.12, (y0 + y1) / 2), text, color=color, fs=8.0,
              weight="bold", ha="left")

    dh_arrow(-0.75, 0.0, 0.0, "", MUTED)
    ax.plot([-1.05, -0.45], [0, 0], color="#2e7d32", lw=2.6, zorder=8)
    label(ax, (-0.75, -0.28), "flat floor\n$\\Delta h \\approx 0$", color="#2e7d32", fs=8.0,
          weight="bold")

    k = 5
    dh_arrow(k * TREAD, k * RISER, (k + 1) * RISER,
             "stairs\n$\\Delta h =$ riser", C_STEP)

    dh_arrow(RUN + 1.5, RISE, RISE + 2.1, "wall\n$\\Delta h =$ metres", C_WALL)

    # --- total rise -------------------------------------------------------
    ax.add_patch(FancyArrowPatch((RUN + 0.75, 0.0), (RUN + 0.75, RISE),
                                 arrowstyle="<|-|>", mutation_scale=10,
                                 color=C_FLIGHT, lw=1.7, zorder=8))
    label(ax, (RUN + 0.62, RISE / 2), "total rise\nmust reach\nthe next storey",
          color=C_FLIGHT, fs=8.0, weight="bold", ha="right")

    # --- the flight: cells strictly between the two storeys ---------------
    ax.add_patch(Rectangle((-0.02, 0.001), RUN + 0.02, RISE - 0.002,
                           facecolor=C_FLIGHT, alpha=0.10, edgecolor=C_FLIGHT,
                           lw=1.2, ls=(0, (4, 3)), zorder=0))
    label(ax, (RUN / 2, RISE + 0.42),
          "the flight: every cell whose height lies\nstrictly between the two storeys",
          color=C_FLIGHT, fs=8.6, weight="bold")

    for xy, txt, col in (((0.0, 0.0), "foot", C_FOOT), ((RUN, RISE), "top", C_FLIGHT)):
        ax.plot(*xy, "o", ms=9, color=col, markeredgecolor="white",
                markeredgewidth=1.5, zorder=10)
        label(ax, (xy[0] + 0.1, xy[1] - 0.30), txt, color=col, fs=9.0, weight="bold",
              ha="left")

    ax.set_xlim(-1.95, RUN + 2.05)
    ax.set_ylim(-0.95, RISE + 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a)  cross-section: what the height layer sees",
                 fontsize=11, color=INK, pad=4, fontweight="bold", loc="left")


def draw_band(ax):
    """The one-dimensional test, as a number line."""
    ax.axhline(0, color=MUTED, lw=1.2)
    ax.add_patch(Rectangle((0.06, -0.16), 0.24, 0.32, facecolor=C_STEP,
                           alpha=0.30, edgecolor=C_STEP, lw=1.4))
    for x, txt, col in ((0.0, "flat\nfloor", "#2e7d32"),
                        (0.18, "stairs", C_STEP),
                        (0.75, "wall", C_WALL)):
        ax.plot([x], [0], "o", ms=7, color=col, markeredgecolor="white",
                markeredgewidth=1.2, zorder=5)
        label(ax, (x, -0.42), txt, color=col, fs=8.4, weight="bold")
    label(ax, (0.06, 0.40), "noise floor", fs=7.6, color=MUTED)
    label(ax, (0.30, 0.40), "climb limit\nof the agent", fs=7.6, color=MUTED)
    ax.set_xlim(-0.08, 0.9)
    ax.set_ylim(-0.75, 0.75)
    ax.axis("off")
    ax.set_title("(b)  a cell is steppable if $\\Delta h$ falls in the band",
                 fontsize=11, color=INK, pad=2, fontweight="bold", loc="left")


def draw_topdown(ax):
    """Why the mask is not the flight, seen from above."""
    w = 1.2
    ax.add_patch(Rectangle((0, 0), w, RUN, facecolor="#f2f2f0",
                           edgecolor=SLAB_EDGE, lw=1.2))
    # The mask: only the riser edges clear the noise floor; tread interiors are flat.
    for k in range(N):
        ax.add_patch(Rectangle((0, k * TREAD - 0.045), w, 0.09,
                               facecolor=C_STEP, edgecolor="none"))
    label(ax, (w + 0.12, RUN / 2),
          "stair mask\n\nonly the risers clear the\nnoise floor, so a staircase\n"
          "arrives as a row of stripes",
          color=C_STEP, fs=8.2, weight="bold", ha="left")

    ax.add_patch(Rectangle((w + 1.85, 0), w, RUN, facecolor=C_FLIGHT, alpha=0.22,
                           edgecolor=C_FLIGHT, lw=1.4))
    ax.plot([w + 1.85 + w / 2], [0.0], "o", ms=8, color=C_FOOT,
            markeredgecolor="white", markeredgewidth=1.3, zorder=5)
    ax.plot([w + 1.85 + w / 2], [RUN], "o", ms=8, color=C_FLIGHT,
            markeredgecolor="white", markeredgewidth=1.3, zorder=5)
    label(ax, (w + 1.85 + w / 2, -0.30), "foot", color=C_FOOT, fs=8.6, weight="bold")
    label(ax, (w + 1.85 + w / 2, RUN + 0.28), "top", color=C_FLIGHT, fs=8.6, weight="bold")
    label(ax, (w + 1.85 + w + 0.12, RUN / 2),
          "flight\n\ntreads touch, so their\nin-between heights form\none connected run",
          color=C_FLIGHT, fs=8.2, weight="bold", ha="left")

    ax.set_xlim(-0.25, 7.4)
    ax.set_ylim(-0.7, RUN + 0.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(c)  from above: the mask says where to step, the flight says where it goes",
                 fontsize=11, color=INK, pad=4, fontweight="bold", loc="left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/figures/stair_schematic.png")
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    fig = plt.figure(figsize=(12.6, 8.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 0.72, 1.45], hspace=0.30)
    draw_section(fig.add_subplot(gs[0]))
    draw_band(fig.add_subplot(gs[1]))
    draw_topdown(fig.add_subplot(gs[2]))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
