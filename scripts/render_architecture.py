#!/usr/bin/env python3
"""The system figure: scene-graph builder on top, surface search underneath.

Every box is a module that exists in `src/osg`, named with the same word the
code and `docs/` use, so the figure and the prose cannot drift apart.  The
three insets are this project's own data -- track crops from a saved map, the
orthographic floor render, and the rebuilt 3-D scene graph.

    python scripts/render_architecture.py --scene 00848-ziup5kvtCCR
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

from osg.graph.map_store import load_map, _track_from_record

INK = "#16202a"
MUTED = "#5d6b76"
BAND_A = "#eef4f7"
BAND_B = "#fdf4ec"
C_OBJ = "#c02a22"
C_CONT = "#b8471f"
C_ROOM = "#1f8a2f"
C_FLOOR = "#0d7c8c"
C_FRONT = "#15637f"


def box(ax, x, y, w, h, text, *, fc="white", ec=INK, ls="--", lw=1.1,
        fs=9.0, weight="normal", tc=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc or INK, zorder=4, linespacing=1.35,
            fontweight=weight)
    return (x, y, w, h)


def arrow(ax, a, b, *, color=MUTED, lw=1.2, rad=0.0, side="e2w"):
    """Connect two boxes edge to edge; `side` picks which edges."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    pts = {
        "e2w": ((ax1 + aw, ay1 + ah / 2), (bx1, by1 + bh / 2)),
        "s2n": ((ax1 + aw / 2, ay1), (bx1 + bw / 2, by1 + bh)),
        "n2s": ((ax1 + aw / 2, ay1 + ah), (bx1 + bw / 2, by1)),
        "e2n": ((ax1 + aw, ay1 + ah / 2), (bx1 + bw / 2, by1 + bh)),
        "s2w": ((ax1 + aw / 2, ay1), (bx1, by1 + bh / 2)),
    }
    p, q = pts[side]
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=11, color=color, lw=lw,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1.5, shrinkB=2.5, zorder=2))


def caption(ax, x, y, text, color=MUTED, fs=7.4, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color,
            style="italic", zorder=4)


def inset(fig, rect, img, *, border=INK):
    a = fig.add_axes(rect)
    a.imshow(img)
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_edgecolor(border); s.set_linewidth(0.9)
    return a


def load_assets(scene: str):
    """Track crops and the cached orthographic floor render, if present."""
    blob = load_map(WORKSPACE / f"outputs/maps_v5/{scene}/{scene}.json")
    crops = []
    for rec in sorted(blob["tracks"], key=lambda r: -float(r["best_score"])):
        track = _track_from_record(rec)
        if track.best_crop is None:
            continue
        crops.append((track.label, track.best_crop))
        if len(crops) >= 3:
            break
    top = None
    for cand in sorted((WORKSPACE / "outputs/figures").glob(f"topdown_{scene}_*.npz")):
        top = np.load(cand)["img"]
        break
    return crops, top


def load_section(scene: str):
    """The two-storey cut-away, with the floor heights the map estimated.

    Reuses the cache `render_scene_graph_multifloor.py` writes, so the inset is
    the same render as the standalone figure rather than a second one that
    could drift from it.
    """
    hits = sorted((WORKSPACE / "outputs/figures").glob(f"side_{scene}_*.npz"))
    if not hits:
        return None, None, []
    blob = np.load(hits[0])
    img = blob["img"]
    extent = tuple(float(v) for v in blob["extent"])
    heights = []
    mp = WORKSPACE / f"outputs/maps_15/{scene}/{scene}.json"
    if mp.exists():
        for spec in load_map(mp).get("floors", []):
            heights.append((int(spec["key"]), float(spec["height_y"])))
    # HM3D renders the void black; drop it so the section floats on the band.
    alpha = np.where(img.sum(axis=2) > 16, 255, 0).astype(np.uint8)
    return np.dstack([img, alpha]), extent, sorted(heights)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="00848-ziup5kvtCCR")
    ap.add_argument("--inset-scene", default="00808-y9hTuugGdiq",
                    help="scene for the cut-away inset; a multi-floor one")
    ap.add_argument("--out", default=str(WORKSPACE / "outputs/figures/architecture.pdf"))
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    crops, top = load_assets(args.scene)
    section, sec_extent, floors = load_section(args.inset_scene)

    fig = plt.figure(figsize=(14.0, 7.0))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_axis_off()

    # ---------------------------------------------------------------- band A
    ax.add_patch(FancyBboxPatch((0.012, 0.500), 0.976, 0.482,
                                boxstyle="round,pad=0.004,rounding_size=0.010",
                                facecolor=BAND_A, edgecolor="none", zorder=0))
    ax.text(0.5, 0.958, "Hierarchical Scene Graph Builder", ha="center",
            fontsize=12.5, fontweight="bold", style="italic", color=INK, zorder=4)

    # Row 1 is the object channel; row 2 is the geometry channel.  Presence is
    # a per-track belief, so it belongs on the object row, not between the room
    # segmenter and the container layer as an earlier draft had it.
    rgbd = box(ax, 0.022, 0.700, 0.062, 0.105, "RGB-D\n+ pose", fs=9.0)
    kf = box(ax, 0.100, 0.700, 0.078, 0.105, "Keyframe\nSelection")
    # Two perception channels, not one.  The named channel is calibrated and
    # gates on its own score; the class-agnostic channel has no score to
    # calibrate and is admitted on a CLIP cosine, so it stays a separate tier
    # all the way through to candidate ranking.  Drawing them as one box was
    # the figure's only real disagreement with the method.
    yolo = box(ax, 0.184, 0.762, 0.114, 0.043, "YOLOE-seg  (named)", fs=7.8)
    fsam = box(ax, 0.184, 0.700, 0.114, 0.043,
               "FastSAM + CLIP  (open-set)", fs=7.8)
    fuse = box(ax, 0.304, 0.700, 0.046, 0.105, "Fused\nDetections", fs=8.2)
    ell = box(ax, 0.414, 0.700, 0.074, 0.105, "Association\n+ Ellipsoid Fit", fs=8.2)
    ref = box(ax, 0.492, 0.700, 0.066, 0.105, "Multi-view\nRefinement", fs=8.2)
    obj = box(ax, 0.562, 0.700, 0.056, 0.105, "Object\nNodes", ec=C_OBJ, tc=C_OBJ,
              fs=8.4)
    pres = box(ax, 0.622, 0.700, 0.070, 0.105, "Presence\nFilter", fs=8.4)

    cost = box(ax, 0.100, 0.545, 0.078, 0.100, "Costmap\n(per floor)")
    flr = box(ax, 0.196, 0.545, 0.092, 0.100, "Floor Estimation\n+ Portals",
              ec=C_FLOOR, tc=C_FLOOR)
    room = box(ax, 0.300, 0.545, 0.092, 0.100, "Room\nSegmentation",
               ec=C_ROOM, tc=C_ROOM)
    cont = box(ax, 0.550, 0.545, 0.080, 0.100, "Container\nNodes",
               ec=C_CONT, tc=C_CONT, fs=8.6)

    arrow(ax, rgbd, kf)
    arrow(ax, kf, yolo); arrow(ax, kf, fsam)
    arrow(ax, yolo, fuse); arrow(ax, fsam, fuse)
    arrow(ax, ell, ref); arrow(ax, ref, obj); arrow(ax, obj, pres)
    arrow(ax, kf, cost, side="s2n")
    arrow(ax, cost, flr); arrow(ax, flr, room)
    tile = (0.414, 0.548, 0.070, 0.094)  # matches the inset rect below
    arrow(ax, room, tile, color=C_ROOM)
    arrow(ax, tile, cont, color=C_ROOM)
    arrow(ax, obj, cont, side="s2n", color=C_OBJ)
    caption(ax, 0.379, 0.822, "instances", fs=7.4)
    caption(ax, 0.066, 0.662, "depth", fs=7.2)
    caption(ax, 0.449, 0.660, "room-labelled costmap", fs=7.0, color=C_ROOM)
    caption(ax, 0.525, 0.822, r"$\geq 3$ views", fs=7.2)

    if crops:
        for i, (_, crop) in enumerate(crops[:2]):
            inset(fig, [0.356 + i * 0.029, 0.706, 0.026, 0.092], crop, border=C_OBJ)
    if top is not None:
        inset(fig, [0.418, 0.548, 0.070, 0.094], top, border=MUTED)

    if section is not None:
        a = fig.add_axes([0.706, 0.560, 0.282, 0.360])
        a.imshow(section, extent=[sec_extent[0], sec_extent[1],
                                  sec_extent[2], sec_extent[3]], origin="upper")
        for key, height in floors:
            a.axhline(height, color=C_FLOOR, lw=1.0, ls="--", alpha=0.9)
            a.annotate(f"floor {key}", (sec_extent[0], height), xytext=(3, 3),
                       textcoords="offset points", fontsize=6.8, color=C_FLOOR,
                       fontweight="bold",
                       path_effects=[matplotlib.patheffects.withStroke(
                           linewidth=2.2, foreground="white")])
        a.set_xlim(sec_extent[0], sec_extent[1])
        a.set_ylim(sec_extent[2], sec_extent[3])
        a.set_aspect("equal"); a.set_axis_off()
        ax.text(0.847, 0.938, f"{args.inset_scene}  ·  orthographic section",
                ha="center", fontsize=7.6, color=MUTED, style="italic", zorder=4)
        ax.text(0.847, 0.528, "Floor -> Room -> Container -> Object",
                ha="center", fontsize=8.6, fontweight="bold", color=INK, zorder=4)

    # ---------------------------------------------------------------- band B
    ax.add_patch(FancyBboxPatch((0.012, 0.030), 0.976, 0.330,
                                boxstyle="round,pad=0.004,rounding_size=0.010",
                                facecolor=BAND_B, edgecolor="none", zorder=0))
    ax.text(0.5, 0.336, "Posterior Search over Support Surfaces", ha="center",
            fontsize=12.5, fontweight="bold", style="italic", color=INK, zorder=4)

    tgt = box(ax, 0.026, 0.176, 0.070, 0.092, "Target\nquery", fs=9.0)
    prior = box(ax, 0.124, 0.176, 0.116, 0.092,
                "Surface Prior\naffinity x proximity", fs=8.6)
    post = box(ax, 0.270, 0.176, 0.104, 0.092, "Surface\nPosterior", ec=C_CONT, tc=C_CONT)
    fro = box(ax, 0.270, 0.056, 0.104, 0.092, "Frontier\nExtraction", ec=C_FRONT, tc=C_FRONT)
    arb = box(ax, 0.412, 0.116, 0.116, 0.092, "Arbitration", lw=1.5, ls="-", fs=9.6,
              weight="bold")
    fp = box(ax, 0.564, 0.226, 0.100, 0.080, "Floor Policy\n(portal)", ec=C_FLOOR,
             tc=C_FLOOR, fs=8.6)
    vp = box(ax, 0.564, 0.116, 0.100, 0.092, "Viewpoint\nPlanner", fs=8.8)
    mv = box(ax, 0.700, 0.116, 0.100, 0.092, "PointNav Mover\n+ Closing Walk", fs=8.4)
    vlm = box(ax, 0.700, 0.006, 0.100, 0.078, "VLM\nAbsence Sensor", fs=8.2)
    act = box(ax, 0.836, 0.116, 0.084, 0.092, "Action", fs=9.6, weight="bold")

    arrow(ax, tgt, prior); arrow(ax, prior, post)
    arrow(ax, post, arb, side="e2w"); arrow(ax, fro, arb, side="e2w")
    arrow(ax, arb, vp); arrow(ax, arb, fp, side="e2w")
    arrow(ax, vp, mv); arrow(ax, mv, act)
    arrow(ax, vlm, mv, side="n2s")

    caption(ax, 0.470, 0.222, "b*d / c   vs   score / cost", fs=7.6, color=INK)
    caption(ax, 0.207, 0.152, "presence belief", fs=7.0)
    caption(ax, 0.812, 0.070, "absence", fs=7.0, ha="left")

    # the two feedback edges that make it a loop, not a pipeline
    ax.add_patch(FancyArrowPatch((0.178, 0.578), (0.312, 0.148), arrowstyle="-|>",
                                 mutation_scale=11, color=C_FRONT, lw=1.1,
                                 linestyle=(0, (4, 3)),
                                 connectionstyle="arc3,rad=-0.28", zorder=2))
    caption(ax, 0.212, 0.400, "costmap", fs=7.2, color=C_FRONT)
    ax.add_patch(FancyArrowPatch((0.592, 0.560), (0.380, 0.232), arrowstyle="-|>",
                                 mutation_scale=11, color=C_CONT, lw=1.1,
                                 linestyle=(0, (4, 3)),
                                 connectionstyle="arc3,rad=0.22", zorder=2))
    caption(ax, 0.520, 0.418, "containers + presence belief", fs=7.2, color=C_CONT)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out}\nwrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
