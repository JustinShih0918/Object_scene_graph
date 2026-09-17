#!/usr/bin/env python3
"""Compose the three-panel teaser from real HM3D top-down floor renders.

The floor textures are real scene geometry. Cup poses, robot markers and paths
are explanatory overlays, not a measured episode trajectory.

    python3 scripts/render_teaser_topdown.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon
import matplotlib.patheffects as pe

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs/figures"
INK = "#26313a"
BLUE = "#286ca4"
RED = "#ba463c"
GREEN = "#21805b"
GOLD = "#d99b18"
MUTED = "#60717c"

UP_BOX = (0.172, 0.585, 0.655, 0.400)
DOWN_BOX = (0.172, 0.180, 0.655, 0.400)


def card(ax, xy, width, height, *, face, edge="none", radius=0.025,
         lw=1.1, z=1):
    patch = FancyBboxPatch(xy, width, height,
                           boxstyle=f"round,pad=0,rounding_size={radius}",
                           facecolor=face, edgecolor=edge, linewidth=lw,
                           zorder=z)
    ax.add_patch(patch)
    return patch


def image_point(meta, px800, py800, box):
    """Map a point in the 800px inspection render to a cropped figure image."""
    scale = meta["resolution"] / 800.0
    left, top, right, bottom = meta["crop"]
    x0, y0, w, h = box
    x = x0 + w * ((px800 * scale - left) / (right - left))
    y = y0 + h * (1.0 - (py800 * scale - top) / (bottom - top))
    return x, y


def route(ax, points, *, color, lw=3.2, dashed=False, arrow_end=True, z=10):
    xx, yy = zip(*points)
    line, = ax.plot(xx, yy, color=color, lw=lw, zorder=z,
                    linestyle=(0, (5, 3)) if dashed else "-",
                    solid_capstyle="round", solid_joinstyle="round")
    line.set_path_effects([pe.Stroke(linewidth=lw + 3.3,
                                     foreground="white"), pe.Normal()])
    if arrow_end:
        arrow = FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                mutation_scale=18, lw=lw - 0.3,
                                color=color, zorder=z + 1)
        arrow.set_path_effects([pe.Stroke(linewidth=lw + 3.1,
                                          foreground="white"), pe.Normal()])
        ax.add_patch(arrow)


def cross(ax, x, y, *, color, size=0.024, lw=3.5, z=14):
    for a, b in (((-1, -1), (1, 1)), ((-1, 1), (1, -1))):
        line, = ax.plot([x + a[0] * size, x + b[0] * size],
                        [y + a[1] * size, y + b[1] * size],
                        color=color, lw=lw, solid_capstyle="round", zorder=z)
        line.set_path_effects([pe.Stroke(linewidth=lw + 3.2,
                                         foreground="white"), pe.Normal()])


def cup(ax, x, y, *, ghost=False, z=13):
    width, height = 0.030, 0.041
    color = RED
    fill = "white" if ghost else "#d53f35"
    ax.add_patch(Polygon([(x - width / 2, y + height / 2),
                          (x + width / 2, y + height / 2),
                          (x + width * 0.38, y - height / 2),
                          (x - width * 0.38, y - height / 2)],
                         closed=True, facecolor=fill, edgecolor=color,
                         linewidth=2.0, linestyle="--" if ghost else "-",
                         zorder=z))
    ax.add_patch(Ellipse((x, y + height / 2), width, height * 0.22,
                         facecolor=fill, edgecolor=color, lw=1.7, zorder=z + 1))
    ax.add_patch(Arc((x + width * 0.47, y), width * 0.56,
                     height * 0.57, theta1=-90, theta2=90,
                     color=color, lw=2.0, zorder=z))


def robot(ax, x, y):
    ax.add_patch(Circle((x, y), 0.026, facecolor=BLUE, edgecolor="white",
                        linewidth=2.4, zorder=12))
    ax.add_patch(Circle((x + 0.007, y + 0.006), 0.0055,
                        facecolor="white", zorder=13))


def ring(ax, x, y, *, color, r=0.045, lw=3.2, z=11):
    ax.add_patch(Ellipse((x, y), 2 * r, 2 * r * 0.90,
                         facecolor="none", edgecolor="white",
                         linewidth=lw + 4, zorder=z - 1))
    ax.add_patch(Ellipse((x, y), 2 * r, 2 * r * 0.90,
                         facecolor="none", edgecolor=color,
                         linewidth=lw, zorder=z))


def floor_tag(ax, y, floor, room):
    card(ax, (0.018, y - 0.039), 0.126, 0.080,
         face="white", edge="#d5dde0", radius=0.014, lw=1.0, z=7)
    ax.text(0.081, y + 0.010, floor, ha="center", va="center",
            color=INK, fontsize=14, fontweight="bold", zorder=8)
    ax.text(0.081, y - 0.020, room, ha="center", va="center",
            color=MUTED, fontsize=10.5, zorder=8)


def note(ax, x, y, text, *, color, size=12.5):
    ax.text(x, y, text, ha="center", va="center", fontsize=size,
            color=color, fontweight="bold", zorder=17,
            bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.20",
                      facecolor="white", edgecolor="#dce5e6", lw=0.9))


def graph(ax, *, state):
    color = BLUE if state == "prior" else RED if state == "stale" else GREEN
    tint = "#edf6fa" if state == "prior" else "#fff0ee" if state == "stale" else "#ecf8f1"
    heading = {"prior": "PRIOR BELIEF", "stale": "OUTDATED MAP",
               "updated": "BELIEF REVISED"}[state]
    nodes = ("F2", "bedroom", "table", "cup") if state != "updated" else (
        "F1", "kitchen", "counter", "cup")
    centers = (0.130, 0.360, 0.620, 0.870)
    widths = (0.150, 0.200, 0.210, 0.130)
    y = 0.056
    card(ax, (0.035, 0.012), 0.930, 0.158, face=tint,
         edge="#d9e4e5", radius=0.022, lw=1.0, z=3)
    ax.text(0.080, 0.145, heading, va="center", fontsize=13.5,
            color=color, fontweight="bold", zorder=6)
    for a, b, wa, wb in zip(centers, centers[1:], widths, widths[1:]):
        line, = ax.plot([a + wa / 2, b - wb / 2], [y + 0.011, y + 0.011],
                        color=color, lw=2.8, zorder=4,
                        linestyle="--" if state == "stale" else "-")
    for cx, width, label in zip(centers, widths, nodes):
        card(ax, (cx - width / 2, y - 0.018), width, 0.057,
             face="white", edge=color, radius=0.012, lw=1.7, z=5)
        ax.text(cx, y + 0.011, label, ha="center", va="center",
                fontsize=12.3 if len(label) > 8 else 13.3,
                color=INK, fontweight="bold", zorder=6)
    if state == "stale":
        cross(ax, 0.766, y + 0.011, color=RED, size=0.013,
              lw=2.8, z=9)


def scene(ax, upper, lower, meta):
    ax.imshow(upper, extent=(UP_BOX[0], UP_BOX[0] + UP_BOX[2],
                             UP_BOX[1], UP_BOX[1] + UP_BOX[3]),
              origin="upper", interpolation="bilinear", aspect="auto", zorder=2)
    ax.imshow(lower, extent=(DOWN_BOX[0], DOWN_BOX[0] + DOWN_BOX[2],
                             DOWN_BOX[1], DOWN_BOX[1] + DOWN_BOX[3]),
              origin="upper", interpolation="bilinear", aspect="auto", zorder=2)
    floor_tag(ax, 0.785, "F2", "bedroom")
    floor_tag(ax, 0.384, "F1", "kitchen")
    return {
        "old": image_point(meta, 250, 250, UP_BOX),
        "robot": image_point(meta, 340, 340, UP_BOX),
        "stair_up": image_point(meta, 420, 270, UP_BOX),
        "stair_down": image_point(meta, 420, 270, DOWN_BOX),
        "target": image_point(meta, 210, 300, DOWN_BOX),
        "up_corridor": image_point(meta, 405, 350, UP_BOX),
        "down_corridor": image_point(meta, 400, 365, DOWN_BOX),
        "kitchen_approach": image_point(meta, 300, 350, DOWN_BOX),
    }


def stage(ax, n, *, color):
    ax.add_patch(Circle((0.073, 0.946), 0.031, facecolor=color,
                        edgecolor="white", lw=2.0, zorder=18))
    ax.text(0.073, 0.946, str(n), color="white", ha="center",
            va="center", fontsize=15, fontweight="bold", zorder=19)


def prior(ax, p):
    stage(ax, 1, color=BLUE)
    robot(ax, *p["robot"])
    cup(ax, *p["old"])
    route(ax, [p["robot"], p["old"]], color=BLUE, lw=2.5,
          arrow_end=False)
    graph(ax, state="prior")


def change(ax, p):
    stage(ax, 2, color=RED)
    robot(ax, *p["robot"])
    cup(ax, *p["old"], ghost=True)
    cross(ax, *p["old"], color=RED)
    cup(ax, *p["target"])
    route(ax, [p["robot"], (p["robot"][0] - 0.085, p["robot"][1] - 0.050),
               (p["old"][0] - 0.105, p["old"][1] - 0.040)],
          color=RED, dashed=True, lw=2.6)
    note(ax, 0.895, 0.755, "searches\nupstairs", color=RED)
    graph(ax, state="stale")


def recovery(ax, p):
    stage(ax, 3, color=GREEN)
    robot(ax, *p["robot"])
    cup(ax, *p["old"], ghost=True)
    cross(ax, *p["old"], color=GREEN)
    cup(ax, *p["target"])
    ring(ax, *p["target"], color=GREEN, r=0.047, lw=3.0)
    ring(ax, *p["stair_up"], color=GOLD, r=0.038, lw=3.0)
    ring(ax, *p["stair_down"], color=GOLD, r=0.038, lw=3.0)
    route(ax, [p["robot"], p["up_corridor"], p["stair_up"],
               p["stair_down"], p["down_corridor"],
               p["kitchen_approach"], p["target"]],
          color=GREEN, lw=3.8)
    note(ax, 0.889, 0.765, "frontier →\nstairs", color=GREEN,
         size=11.8)
    arrow = FancyArrowPatch((0.805, 0.785), p["stair_up"],
                            arrowstyle="-|>", mutation_scale=15,
                            lw=2.1, color=GOLD, zorder=15)
    arrow.set_path_effects([pe.Stroke(linewidth=4.5,
                                      foreground="white"), pe.Normal()])
    ax.add_patch(arrow)
    graph(ax, state="updated")


def check_text_layout(fig, axes):
    """Catch accidental text collisions or labels spilling beyond a panel."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for panel, ax in enumerate(axes, start=1):
        texts = [artist for artist in ax.texts if artist.get_visible()]
        boxes = [(artist.get_text(), artist.get_window_extent(renderer))
                 for artist in texts]
        for i, (name, box) in enumerate(boxes):
            if box.x0 < ax.bbox.x0 - 2 or box.x1 > ax.bbox.x1 + 2:
                raise RuntimeError(f"panel {panel}: label spills out: {name!r}")
            for other, other_box in boxes[i + 1:]:
                overlap_x = min(box.x1, other_box.x1) - max(box.x0, other_box.x0)
                overlap_y = min(box.y1, other_box.y1) - max(box.y0, other_box.y0)
                if overlap_x > 4 and overlap_y > 4:
                    raise RuntimeError(
                        f"panel {panel}: overlapping labels {name!r} / {other!r}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIGURES / "teaser.png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    paths = {name: FIGURES / f"hm3d_topdown_00873_{name}.png"
             for name in ("upper", "lower")}
    meta_path = FIGURES / "hm3d_topdown_00873_meta.json"
    for path in (*paths.values(), meta_path):
        if not path.exists():
            raise SystemExit(f"missing scene asset {path}; render floor assets first")
    upper, lower = (plt.imread(paths[name]) for name in ("upper", "lower"))
    meta = json.loads(meta_path.read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 6.2),
                             gridspec_kw={"wspace": 0.022})
    for ax, draw, tint in zip(axes, (prior, change, recovery),
                              ("#f9fcfd", "#fffafa", "#f9fdfb")):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        card(ax, (0.006, 0.005), 0.988, 0.990, face=tint,
             edge="#d9e3e5", radius=0.029, lw=1.2, z=0)
        p = scene(ax, upper, lower, meta)
        draw(ax, p)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.006, right=0.994, top=0.994, bottom=0.006)
    check_text_layout(fig, axes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf", ".svg"):
        target = args.out.with_suffix(ext)
        fig.savefig(target, dpi=args.dpi, facecolor="white")
        print(f"wrote {target}")
    plt.close(fig)


if __name__ == "__main__":
    main()
