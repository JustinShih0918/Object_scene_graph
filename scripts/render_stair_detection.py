#!/usr/bin/env python3
"""How a staircase is found, drawn from a real saved map.

Four panels, left to right, in the order the method describes: the height layer
the agent built from depth; the neighbour height-difference that separates floor
from stairs from wall; the stair mask that test produces; and the flight, which
answers the question the mask cannot -- where the staircase runs and which end
is the bottom.

Every panel is computed with the same code the agent runs (`Costmap2D
.height_gradient`, `mapping.stairs.detect_stairs` / `find_flights`), so the
figure cannot drift from the method.

    python scripts/render_stair_detection.py --scene 00800-TEEsavR23oF
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

WORKSPACE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from osg.mapping.costmap import Costmap2D, UNKNOWN
from osg.mapping.stairs import find_flights

INK = "#16202a"
MUTED = "#5d6b76"
C_STEP = "#e8952f"
C_FLIGHT = "#15637f"
C_FOOT = "#c02a22"


def load_storey(maps_root: pathlib.Path, scene: str, index: int):
    """Rebuild the agent's own costmap for one storey, height layer included."""
    for path in (maps_root / scene / f"{scene}.json", maps_root / f"{scene}.json"):
        if path.exists():
            break
    else:
        raise SystemExit(f"no saved map for {scene} under {maps_root}")
    blob = json.loads(path.read_text())
    npz = np.load(path.with_suffix(".npz"))
    floors = sorted(blob["floors"], key=lambda f: float(f["height_y"]))
    if index is None:
        # Default to the storey that actually recorded a staircase, so the
        # figure does not have to be pointed at one by hand.
        index = max(range(len(floors)),
                    key=lambda i: int(npz[floors[i]["prefix"] + "stair_mask"].sum())
                    if floors[i]["prefix"] + "stair_mask" in npz.files else -1)
    spec = floors[index]
    prefix = spec["prefix"]
    if prefix + "height" not in npz.files:
        raise SystemExit(f"{path.name} storey {index} has no height layer")

    res = float(spec["resolution"])
    grid = npz[prefix + "grid"]
    cm = Costmap2D(resolution=res, size_m=grid.shape[0] * res, track_height=True)
    cm.grid = grid
    cm.origin = np.asarray(npz[prefix + "origin"], dtype=float)
    cm.height = npz[prefix + "height"]
    if prefix + "stair_mask" in npz.files:
        cm.stair_mask = npz[prefix + "stair_mask"]
    heights = [float(f["height_y"]) for f in floors]
    return cm, float(spec["height_y"]), heights


def crop_around(mask: np.ndarray, pad: int, shape) -> tuple:
    """Bounding box of `mask`, padded, clipped to the grid."""
    rc = np.argwhere(mask)
    if rc.size == 0:
        raise SystemExit("nothing to crop around")
    r0, c0 = rc.min(axis=0) - pad
    r1, c1 = rc.max(axis=0) + pad
    return (max(0, r0), min(shape[0], r1), max(0, c0), min(shape[1], c1))


def panel(ax, title, subtitle=None):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#c8cdd2")
    ax.set_title(title, fontsize=10.5, color=INK, pad=6, fontweight="bold")
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=8.2, color=MUTED, labelpad=6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="00800-TEEsavR23oF")
    ap.add_argument("--maps", default=str(WORKSPACE / "outputs/maps_p1500_osg"))
    ap.add_argument("--storey", type=int, default=None,
                    help="index, bottom-up; default: the storey with a stair mask")
    ap.add_argument("--cell-m", type=float, default=0.1)
    ap.add_argument("--climb-limit-m", type=float, default=0.2)
    ap.add_argument("--min-dh-m", type=float, default=0.03)
    ap.add_argument("--pad", type=int, default=25)
    ap.add_argument("--out", default=str(WORKSPACE / "outputs/figures/stair_detection.png"))
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    cm, floor_y, storeys = load_storey(pathlib.Path(args.maps), args.scene, args.storey)
    next_y = min((h for h in storeys if h > floor_y + 1.0), default=floor_y + 3.0)

    stair_mask = getattr(cm, "stair_mask", None)
    if stair_mask is None or not stair_mask.any():
        raise SystemExit("this storey stored no stair mask")
    r0, r1, c0, c1 = crop_around(stair_mask, args.pad, cm.grid.shape)
    sl = (slice(r0, r1), slice(c0, c1))

    dh, valid = cm.height_gradient(cell_m=args.cell_m)
    block = max(1, int(round(args.cell_m / cm.resolution)))
    dh_fine = np.kron(dh, np.ones((block, block)))[: cm.grid.shape[0], : cm.grid.shape[1]]
    valid_fine = np.kron(valid, np.ones((block, block))).astype(bool)[
        : cm.grid.shape[0], : cm.grid.shape[1]]

    flights = find_flights(cm, floor_y)
    flight_mask = np.zeros(cm.grid.shape, dtype=bool)
    foot = top = None
    if flights:
        f = max(flights, key=lambda x: x.n_cells)
        flight_mask[f.cells_rc[:, 0], f.cells_rc[:, 1]] = True
        foot, top = f.foot_xy, f.top_xy

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.3))

    # (a) the height layer -------------------------------------------------
    h = cm.height[sl]
    im = axes[0].imshow(np.transpose(h), origin="lower", cmap="viridis",
                        vmin=floor_y - 0.2, vmax=next_y + 0.2, interpolation="nearest")
    panel(axes[0], "(a)  height layer",
          "lowest surface seen in each cell,\nbuilt from depth")
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    for y, lab in ((floor_y, "this storey"), (next_y, "next storey")):
        cb.ax.axhline(y, color="white", lw=1.4)
        cb.ax.annotate(lab, (0.5, y), xycoords=("axes fraction", "data"),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", fontsize=6.6, color="white", fontweight="bold")

    # (b) the three regimes ------------------------------------------------
    d = dh_fine[valid_fine & np.isfinite(dh_fine)]
    d = d[d < 2.0]
    axes[1].hist(d, bins=140, color="#b9c2c9", log=True)
    axes[1].axvspan(args.min_dh_m, args.climb_limit_m, color=C_STEP, alpha=0.30, lw=0)
    axes[1].axvline(args.min_dh_m, color=C_STEP, lw=1.3)
    axes[1].axvline(args.climb_limit_m, color=C_STEP, lw=1.3)
    for x, txt, ha in ((0.005, "flat floor\n$\\approx 0$", "left"),
                       ((args.min_dh_m + args.climb_limit_m) / 2, "stairs\n(riser height)", "center"),
                       (0.9, "wall\n(metres)", "center")):
        axes[1].annotate(txt, (x, 0.80), xycoords=("data", "axes fraction"),
                         ha=ha, fontsize=8.0, color=INK)
    axes[1].set_xlim(0, 1.2)
    axes[1].set_yticks([])
    axes[1].set_title("(b)  height step to neighbours",
                      fontsize=10.5, color=INK, pad=6, fontweight="bold")
    axes[1].set_xlabel("$\\Delta h$  (m)     — shaded band = steppable",
                       fontsize=8.2, color=MUTED, labelpad=6)
    for s in ("top", "right", "left"):
        axes[1].spines[s].set_visible(False)

    # (c) the mask ---------------------------------------------------------
    base = np.full(cm.grid[sl].shape + (3,), 0.80)
    base[cm.grid[sl] != UNKNOWN] = (0.96, 0.96, 0.95)
    base[cm.grid[sl] == 100] = (0.16, 0.19, 0.23)
    shown = base.copy()
    shown[stair_mask[sl]] = matplotlib.colors.to_rgb(C_STEP)
    axes[2].imshow(np.transpose(shown, (1, 0, 2)), origin="lower", interpolation="nearest")
    panel(axes[2], "(c)  stair mask",
          "cells that pass (b): where the agent\nmay step — broken into riser edges")

    # (d) the flight -------------------------------------------------------
    shown = base.copy()
    shown[stair_mask[sl]] = (0.93, 0.87, 0.78)
    shown[flight_mask[sl]] = matplotlib.colors.to_rgb(C_FLIGHT)
    axes[3].imshow(np.transpose(shown, (1, 0, 2)), origin="lower", interpolation="nearest")
    if foot is not None:
        for xy, lab, col in ((foot, "foot", C_FOOT), (top, "top", C_FLIGHT)):
            rc = cm.world_to_grid(np.asarray(xy, dtype=float))
            axes[3].plot(rc[0] - r0, rc[1] - c0, "o", ms=8, color=col,
                         markeredgecolor="white", markeredgewidth=1.4, zorder=5)
            axes[3].annotate(lab, (rc[0] - r0, rc[1] - c0), xytext=(7, 6),
                             textcoords="offset points", fontsize=8.4,
                             color=col, fontweight="bold", zorder=6)
    panel(axes[3], "(d)  flight",
          "cells whose height lies between the two\nstoreys: contiguous, with a foot and a top")

    fig.tight_layout()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  storey y={floor_y:.2f} -> {next_y:.2f}   mask cells={int(stair_mask.sum())}"
          f"   flights={len(flights)}   flight cells={int(flight_mask.sum())}")


if __name__ == "__main__":
    main()
