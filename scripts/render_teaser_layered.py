#!/usr/bin/env python3
"""Render the layered two-floor teaser with scene graphs on the right.

The HM3D floor renders are true geometry; the cup move and robot are
illustrative. The stair route follows a geometry-checked navmesh guide through
the scene's two-flight switchback, but is not an episode trajectory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon
import matplotlib.patheffects as pe
from matplotlib.transforms import Affine2D

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs/figures"
INK = "#26343d"
MUTED = "#617079"
BLUE = "#296da7"
RED = "#bf4640"
GREEN = "#1e805a"
GOLD = "#d79a1b"
CARD_LINE = "#dce5e6"
ASPECT = 16.8 / 6.2

UPPER = (0.064, 0.522, 0.602, 0.445)
LOWER = (0.040, 0.068, 0.614, 0.430)


def card(ax, x, y, w, h, *, face="white", edge=CARD_LINE,
         radius=0.014, lw=1.1, z=1):
    item = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          facecolor=face, edgecolor=edge, linewidth=lw,
                          zorder=z)
    ax.add_patch(item)
    return item


def floor_transform(box, angle):
    x, y, w, h = box
    return Affine2D().rotate_deg_around(x + w / 2, y + h / 2, angle)


def floor_point(meta, u, v, box, affine):
    left, top, right, bottom = meta["crop"]
    scale = meta["resolution"] / 800.0
    x, y, w, h = box
    base = (x + w * (u * scale - left) / (right - left),
            y + h * (1 - (v * scale - top) / (bottom - top)))
    return tuple(affine.transform(base))


def line(ax, points, *, color, lw=3.6, dashed=False, arrow_end=False, z=10):
    xs, ys = zip(*points)
    artist, = ax.plot(xs, ys, color=color, lw=lw, zorder=z,
                      linestyle=(0, (5, 3)) if dashed else "-",
                      solid_capstyle="round", solid_joinstyle="round")
    artist.set_path_effects([pe.Stroke(linewidth=lw + 3.5,
                                       foreground="white"), pe.Normal()])
    if arrow_end:
        arrow = FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                mutation_scale=19, color=color, lw=lw - 0.2,
                                zorder=z + 1)
        arrow.set_path_effects([pe.Stroke(linewidth=lw + 3.4,
                                          foreground="white"), pe.Normal()])
        ax.add_patch(arrow)


def cross(ax, x, y, *, color, sx=0.012, sy=0.029, lw=3.4, z=16):
    for reverse in (False, True):
        ya, yb = (y - sy, y + sy) if not reverse else (y + sy, y - sy)
        artist, = ax.plot([x - sx, x + sx], [ya, yb], color=color,
                          lw=lw, solid_capstyle="round", zorder=z)
        artist.set_path_effects([pe.Stroke(linewidth=lw + 3.2,
                                           foreground="white"), pe.Normal()])


def ring(ax, x, y, *, color, radius=0.021, lw=3.0, z=14):
    # Data coordinates are wide; this y scale yields a physical circle.
    yr = radius * ASPECT
    ax.add_patch(Ellipse((x, y), 2 * radius, 2 * yr,
                         facecolor="none", edgecolor="white",
                         linewidth=lw + 4.0, zorder=z - 1))
    ax.add_patch(Ellipse((x, y), 2 * radius, 2 * yr,
                         facecolor="none", edgecolor=color,
                         linewidth=lw, zorder=z))


def robot(ax, x, y):
    rx = 0.015
    ax.add_patch(Ellipse((x, y), 2 * rx, 2 * rx * ASPECT,
                         facecolor=BLUE, edgecolor="white",
                         linewidth=2.4, zorder=14))
    ax.add_patch(Ellipse((x + 0.0037, y + 0.008), 0.004, 0.011,
                         facecolor="white", edgecolor="none", zorder=15))


def cup(ax, x, y, *, ghost=False):
    w, h = 0.017, 0.047
    fill = "white" if ghost else "#d83d35"
    ax.add_patch(Polygon([(x - w / 2, y + h / 2),
                          (x + w / 2, y + h / 2),
                          (x + w * 0.38, y - h / 2),
                          (x - w * 0.38, y - h / 2)],
                         facecolor=fill, edgecolor=RED, linewidth=2.0,
                         linestyle="--" if ghost else "-", zorder=14))
    ax.add_patch(Ellipse((x, y + h / 2), w, 0.010,
                         facecolor=fill, edgecolor=RED, linewidth=1.5,
                         zorder=15))
    ax.add_patch(Arc((x + w * 0.50, y), 0.010, 0.029,
                     theta1=-90, theta2=90, color=RED, lw=1.8, zorder=14))


def floor_tag(ax, x, y, text):
    card(ax, x, y - 0.035, 0.091, 0.070, face="#35424a",
         edge="white", radius=0.009, lw=1.8, z=13)
    ax.text(x + 0.0455, y, text, ha="center", va="center",
            fontsize=14.5, fontweight="bold", color="white", zorder=14)


def note(ax, x, y, text, *, color, fontsize=12.0):
    return ax.text(x, y, text, color=color, fontsize=fontsize,
                   fontweight="bold", ha="center", va="center", zorder=18,
                   bbox=dict(boxstyle="round,pad=0.30,rounding_size=0.20",
                             facecolor="white", edgecolor=CARD_LINE,
                             linewidth=0.9))


def graph_panel(ax, y, *, stale):
    color = RED if stale else GREEN
    face = "#fff3f2" if stale else "#effaf2"
    title = "Outdated belief" if stale else "Belief revised"
    labels = ("F2", "bedroom", "table", "cup") if stale else (
        "F1", "kitchen", "counter", "cup")
    card(ax, 0.723, y, 0.257, 0.390, face=face,
         edge="#e4c7c7" if stale else "#bfe1c9",
         radius=0.020, lw=1.3, z=3)
    ax.text(0.8515, y + 0.341, title, ha="center", va="center",
            color=color, fontsize=17.0, fontweight="bold", zorder=7)
    node_y = [y + 0.270, y + 0.202, y + 0.134, y + 0.066]
    for i, (level, label) in enumerate(zip(node_y, labels)):
        card(ax, 0.772, level - 0.023, 0.159, 0.046,
             face="white", edge=color, radius=0.010, lw=1.6, z=6)
        ax.text(0.8515, level, label, ha="center", va="center",
                fontsize=12.7 if len(label) > 6 else 13.4,
                fontweight="bold", color=INK, zorder=7)
        if i < 3:
            style = "--" if stale and i == 2 else "-"
            ax.plot([0.8515, 0.8515], [level - 0.023, node_y[i + 1] + 0.023],
                    color=color, lw=2.3, linestyle=style, zorder=5)
    if stale:
        cross(ax, 0.8515, (node_y[2] + node_y[3]) / 2,
              color=RED, sx=0.008, sy=0.018, lw=2.7, z=11)


def check_text_layout(fig, ax):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(text.get_text(), text.get_window_extent(renderer))
             for text in ax.texts if text.get_visible()]
    for i, (name, box) in enumerate(texts):
        if box.x0 < ax.bbox.x0 - 2 or box.x1 > ax.bbox.x1 + 2:
            raise RuntimeError(f"label outside figure: {name!r}")
        for other, other_box in texts[i + 1:]:
            dx = min(box.x1, other_box.x1) - max(box.x0, other_box.x0)
            dy = min(box.y1, other_box.y1) - max(box.y0, other_box.y0)
            if dx > 4 and dy > 4:
                raise RuntimeError(f"overlapping labels: {name!r} / {other!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIGURES / "teaser.png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    prefix = FIGURES / "hm3d_topdown_00873"
    upper_path = Path(f"{prefix}_upper.png")
    lower_path = Path(f"{prefix}_lower.png")
    meta_path = Path(f"{prefix}_meta.json")
    guide_path = Path(f"{prefix}_stair_path.json")
    for path in (upper_path, lower_path, meta_path, guide_path):
        if not path.exists():
            raise SystemExit(f"missing {path}; run render_teaser_floor_assets.py")
    upper, lower = plt.imread(upper_path), plt.imread(lower_path)
    meta = json.loads(meta_path.read_text())
    guide = json.loads(guide_path.read_text())["points_uvy"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    fig = plt.figure(figsize=(16.8, 6.2))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    upper_aff = floor_transform(UPPER, -2.8)
    lower_aff = floor_transform(LOWER, 2.6)
    for image, box, affine, z in ((lower, LOWER, lower_aff, 2),
                                  (upper, UPPER, upper_aff, 3)):
        x, y, w, h = box
        ax.imshow(image, extent=(x, x + w, y, y + h),
                  origin="upper", interpolation="bilinear", aspect="auto",
                  transform=affine + ax.transData, zorder=z)
    floor_tag(ax, 0.014, 0.777, "Floor 2")
    floor_tag(ax, 0.014, 0.278, "Floor 1")

    up = lambda u, v: floor_point(meta, u, v, UPPER, upper_aff)
    down = lambda u, v: floor_point(meta, u, v, LOWER, lower_aff)
    old, target, start = up(250, 250), down(210, 300), up(340, 340)
    cup(ax, *old, ghost=True)
    cross(ax, *old, color=RED)
    robot(ax, *start)
    # The short prior route stops before the old cup marker.
    line(ax, [start, up(285, 290)], color=BLUE,
         lw=2.9, arrow_end=True, z=10)

    # The guide is the actual shortest navmesh path through both flights of
    # this scene's switchback staircase. Split it by storey for the two renders.
    upper_guide = [up(u, v) for u, v, h in guide if h > -1.5]
    lower_guide = [down(u, v) for u, v, h in guide if h <= -1.5]
    if len(upper_guide) < 3 or len(lower_guide) < 3:
        raise RuntimeError("stair path does not traverse two storeys")
    line(ax, upper_guide, color=GREEN, lw=4.0, z=11)
    line(ax, [upper_guide[-1], lower_guide[0]], color=GREEN,
         lw=4.0, arrow_end=True, z=11)
    # Follow the navmesh through the stair exit, then turn toward the kitchen
    # counter. The arrow stops beside the cup rather than crossing its marker.
    approach = down(250, 300)
    line(ax, [*lower_guide[:-1], down(330, 360), approach], color=GREEN,
         lw=4.0, arrow_end=True, z=11)
    cup(ax, *target)
    ring(ax, *target, color=GREEN, radius=0.024, lw=3.0)
    ring(ax, *upper_guide[2], color=GOLD, radius=0.019, lw=3.0)
    ring(ax, *lower_guide[0], color=GOLD, radius=0.019, lw=3.0)
    if ((approach[0] - target[0]) ** 2 +
            ((approach[1] - target[1]) / ASPECT) ** 2) ** 0.5 <= 0.024:
        raise RuntimeError("route arrow overlaps the target cup")

    note(ax, 0.539, 0.793, "frontier selects\nstair flight",
         color=GREEN, fontsize=12.0)
    leader = FancyArrowPatch((0.493, 0.762), upper_guide[2],
                             arrowstyle="-|>", mutation_scale=16,
                             color=GOLD, lw=2.0, zorder=16)
    leader.set_path_effects([pe.Stroke(linewidth=4.5,
                                       foreground="white"), pe.Normal()])
    ax.add_patch(leader)

    graph_panel(ax, 0.541, stale=True)
    graph_panel(ax, 0.080, stale=False)
    arrow = FancyArrowPatch((0.8515, 0.529), (0.8515, 0.482),
                            arrowstyle="-|>", mutation_scale=19,
                            color="#7b8b91", lw=2.5, zorder=7)
    ax.add_patch(arrow)
    check_text_layout(fig, ax)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf", ".svg"):
        target_path = args.out.with_suffix(ext)
        fig.savefig(target_path, dpi=args.dpi, facecolor="white")
        print(f"wrote {target_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
