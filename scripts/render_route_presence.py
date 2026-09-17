#!/usr/bin/env python3
"""The route figure: where the agent went, and what it believed while going.

Paper Fig. 5 analogue, but carrying the thing a Voronoi-and-path picture cannot
show -- the presence belief. A static route picture says the agent walked from
A to B. This one says *why*: which mapped tracks still held belief mass, which
had been disproved, which storey the mass was on, and what it finally committed
to. All of it is read back from a finished run's `episodes.jsonl`; nothing here
re-simulates anything.

The route is drawn at whatever resolution the run recorded. With
`eval.behaviour_log=true` that is the dense per-step pose in `step_trace`; a run
without it still has the decision points -- `frontier_select_log` (where the
agent stood, which frontier it picked), `goal_commit_log`, `approach_diag` and
`final_xy` -- and those are drawn as the decision polyline instead. The legend
says which one you are looking at, because a dashed decision polyline is not a
path and must never be read as one.

    python scripts/render_route_presence.py \
        --run outputs/authored_val/00821-eF36g7L6Z9M/dynamic \
        --episode-id 00821-eF36g7L6Z9M__in_anchor_01__50002__s0 \
        --maps outputs/authored_val/maps \
        --out outputs/figures/route_presence.png
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
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch

from osg.mapping.costmap import FREE, OCCUPIED, UNKNOWN

INK = "#16202a"
MUTED = "#5d6b76"
FREE_RGB = (0.975, 0.975, 0.968)
OCC_RGB = (0.16, 0.19, 0.23)
UNK_RGB = (0.80, 0.81, 0.82)
C_PATH = "#e8952f"          # the route, ASCENT/Fig.5 orange
C_ROBOT = "#1668b0"
C_TARGET = "#c02a22"
C_COMMIT = "#0d7c8c"
BELIEVED = "#2e7d32"        # a track that still holds mass
DISPROVED = "#b0bec5"       # a track the filter has written off


# ---------------------------------------------------------------------------
# reading the run
# ---------------------------------------------------------------------------
def load_episode(run: pathlib.Path, episode_id: str | None) -> dict:
    rows = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines()
            if l.strip()]
    if not rows:
        raise SystemExit(f"{run}: no episodes")
    if episode_id:
        for r in rows:
            if r.get("episode_id") == episode_id:
                return r
        raise SystemExit(f"{run}: no episode {episode_id}\n  have: "
                         + "\n        ".join(r.get("episode_id", "?") for r in rows))
    # Default to the most legible episode: a success with the most decisions on it.
    rows.sort(key=lambda r: (bool(r.get("success")),
                             len(r.get("frontier_select_log") or []),
                             len(r.get("search_log_events") or [])), reverse=True)
    return rows[0]


def load_storey(maps_root: pathlib.Path, scene: str, height_y: float):
    """The stored costmap for the storey nearest `height_y`, flat or nested layout."""
    for path in (maps_root / scene / f"{scene}.json", maps_root / f"{scene}.json"):
        if path.exists():
            break
    else:
        raise SystemExit(f"no saved map for {scene} under {maps_root}")
    blob = json.loads(path.read_text())
    npz = np.load(path.with_suffix(".npz"))
    floors = blob.get("floors") or []
    if not floors:                                    # single-storey snapshot
        return npz["grid"], np.asarray(npz["origin"], float), float(blob["resolution"]), None
    best = min(floors, key=lambda f: abs(float(f["height_y"]) - height_y))
    prefix = best["prefix"]
    rooms = npz[prefix + "room_labels"] if prefix + "room_labels" in npz.files else None
    return (npz[prefix + "grid"], np.asarray(npz[prefix + "origin"], float),
            float(best["resolution"]), rooms)


def storey_height(episode: dict, floor_key: int) -> float:
    """Height of a floor key, from the run's own floor_log."""
    for step, key, y in (episode.get("floor_log") or []):
        if int(key) == int(floor_key):
            return float(y)
    return float(episode.get("start_y") or 0.0)


# ---------------------------------------------------------------------------
# the route
# ---------------------------------------------------------------------------
def route_from(episode: dict):
    """(xy array, kind). `kind` is 'dense' or 'decisions' and must reach the legend."""
    trace = episode.get("step_trace")
    if trace:
        goal = int(episode.get("goal_floor") or 0)
        pts = [t["xy"] for t in trace if t.get("xy") and int(t.get("floor", goal)) == goal]
        if len(pts) > 2:
            return np.asarray(pts, float), "dense"
    pts = []
    for row in (episode.get("frontier_select_log") or []):
        # [step, agent_xy, frontier_xy, cost, n_frontiers]
        if len(row) >= 3 and row[1]:
            pts.append(row[1])
            pts.append(row[2])
    for row in (episode.get("goal_commit_log") or []):
        c = row.get("center")
        if c:
            pts.append([c[0], c[2]])
    diag = episode.get("approach_diag") or {}
    if diag.get("goal_xy"):
        pts.append(diag["goal_xy"])
    if episode.get("final_xy"):
        pts.append(episode["final_xy"])
    return (np.asarray(pts, float) if len(pts) >= 2 else np.zeros((0, 2))), "decisions"


def known_extent(grid, origin, res, pad_m=1.2):
    rows, cols = np.nonzero(grid != UNKNOWN)
    if rows.size == 0:
        raise SystemExit("storey has no observed cells")
    x0 = origin[0] + rows.min() * res - pad_m
    x1 = origin[0] + rows.max() * res + pad_m
    y0 = origin[1] + cols.min() * res - pad_m
    y1 = origin[1] + cols.max() * res + pad_m
    return x0, x1, y0, y1


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------
def draw_map(ax, grid, origin, res):
    img = np.empty(grid.shape + (3,), float)
    img[:] = UNK_RGB
    img[grid == FREE] = FREE_RGB
    img[grid == OCCUPIED] = OCC_RGB
    h, w = grid.shape
    ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", interpolation="nearest",
              extent=[origin[0], origin[0] + h * res, origin[1], origin[1] + w * res],
              zorder=0)


def draw_beliefs(ax, episode, floor_key, *, max_other=40):
    """Every track the filter still has an opinion about, sized by that opinion.

    Target-class tracks are named and ringed; the rest are context. Radius is
    the belief, so a disproved track is visibly a dot and a believed one a disc
    -- which is the whole claim of the presence layer in one glance.
    """
    drawn = []
    targets = [t for t in (episode.get("target_tracks") or [])
               if int(t.get("floor_key", floor_key)) == int(floor_key)]
    # `presence_events` are the tracks whose belief CHANGED; keep the last state.
    latest: dict[int, dict] = {}
    for ev in (episode.get("presence_events") or []):
        latest[int(ev["track_id"])] = ev
    others = sorted(latest.values(), key=lambda e: -float(e.get("p", 0.0)))[:max_other]

    for ev in others:
        c = ev.get("center")
        if not c:
            continue
        p = float(ev.get("p", 0.0))
        ax.add_patch(Circle((c[0], c[2]), 0.10 + 0.55 * p, facecolor=DISPROVED,
                            edgecolor="none", alpha=0.45, zorder=3))
    for t in targets:
        c = t.get("center")
        if not c:
            continue
        p = float(t.get("p", 0.0))
        believed = p >= 0.45
        ax.add_patch(Circle((c[0], c[2]), 0.18 + 0.75 * p,
                            facecolor=BELIEVED if believed else "#ffffff",
                            edgecolor=BELIEVED, lw=1.6,
                            alpha=0.80 if believed else 0.95, zorder=5))
        ax.annotate(f"{t.get('label','?')}  p={p:.2f}", (c[0], c[2]),
                    xytext=(0, 12), textcoords="offset points", ha="center",
                    fontsize=7.5, color=INK, zorder=7,
                    path_effects=[matplotlib.patheffects.withStroke(linewidth=2.5,
                                                                    foreground="white")])
        drawn.append(t)
    return drawn


def draw_mass_panel(ax, episode):
    """Belief mass per storey over the episode -- the floor-level argmax, visible.

    `search_log_events` records `floor_mass` at every floor decision, so the
    panel is the decision as the agent saw it, not a reconstruction of it.
    """
    events = episode.get("search_log_events") or []
    if not events:
        ax.axis("off")
        ax.text(0.5, 0.5, "no floor decisions recorded", ha="center", va="center",
                fontsize=8, color=MUTED, transform=ax.transAxes)
        return
    keys = sorted({k for e in events for k in (e.get("floor_mass") or {})},
                  key=lambda k: int(k))
    steps = [int(e["step"]) for e in events]
    for i, k in enumerate(keys):
        series = [float((e.get("floor_mass") or {}).get(k, 0.0)) for e in events]
        ax.plot(steps, series, lw=1.8, marker="o", ms=3,
                color=plt.cm.viridis(0.15 + 0.7 * i / max(1, len(keys) - 1)),
                label=f"storey {k}")
    chosen = {int(e["step"]): int(e["selected_floor"]) for e in events
              if e.get("selected_floor") is not None}
    for step, sel in chosen.items():
        mass = float((dict((int(e["step"]), e) for e in events)[step]
                      .get("floor_mass") or {}).get(str(sel), 0.0))
        ax.plot([step], [mass], marker="*", ms=11, color=C_COMMIT, zorder=5,
                linestyle="none")
    ax.set_xlabel("step", fontsize=8)
    ax.set_ylabel(r"belief mass  $\sum_x b(x)\lambda(x)$", fontsize=8)
    ax.tick_params(labelsize=7)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="*", ms=9, color=C_COMMIT, linestyle="none"))
    labels.append("storey chosen")
    ax.legend(handles, labels, fontsize=7, frameon=False, loc="best")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a run directory with episodes.jsonl")
    ap.add_argument("--episode-id", default=None)
    ap.add_argument("--maps", required=True, help="directory of saved OSG maps")
    ap.add_argument("--out", default=str(WORKSPACE / "outputs/figures/route_presence.png"))
    ap.add_argument("--floor", type=int, default=None,
                    help="storey to draw; default the episode's goal storey")
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    run = pathlib.Path(args.run)
    ep = load_episode(run, args.episode_id)
    scene = ep["scene"]
    floor_key = args.floor if args.floor is not None else int(ep.get("goal_floor") or 0)
    grid, origin, res, _rooms = load_storey(pathlib.Path(args.maps), scene,
                                            storey_height(ep, floor_key))

    fig = plt.figure(figsize=(11.6, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.15, 1.0], wspace=0.22)
    ax = fig.add_subplot(gs[0, 0])
    axm = fig.add_subplot(gs[0, 1])

    draw_map(ax, grid, origin, res)
    drawn = draw_beliefs(ax, ep, floor_key)

    route, kind = route_from(ep)
    if route.size:
        style = dict(color=C_PATH, lw=3.0, solid_capstyle="round", zorder=4)
        if kind == "decisions":
            style.update(lw=2.2, linestyle=(0, (5, 3)))
        ax.plot(route[:, 0], route[:, 1], **style)
        # The robot: a heading arrow at the start, so the picture has a subject.
        ax.plot(route[0, 0], route[0, 1], "o", ms=13, color=C_ROBOT,
                markeredgecolor="white", markeredgewidth=1.6, zorder=8)
        if len(route) > 1:
            d = route[1] - route[0]
            n = float(np.hypot(*d)) or 1.0
            ax.add_patch(FancyArrowPatch(route[0], route[0] + 0.9 * d / n,
                                         arrowstyle="-|>", mutation_scale=14,
                                         color=C_ROBOT, lw=2.0, zorder=9))

    diag = ep.get("approach_diag") or {}
    if diag.get("goal_xy"):
        ax.plot(*diag["goal_xy"], marker="P", ms=12, color=C_COMMIT,
                markeredgecolor="white", markeredgewidth=1.2, zorder=9)
    if ep.get("target_obj_xy"):
        ax.plot(*ep["target_obj_xy"], marker="*", ms=19, color=C_TARGET,
                markeredgecolor="white", markeredgewidth=1.2, zorder=10)

    ax.set_xlim(*known_extent(grid, origin, res)[:2])
    ax.set_ylim(*known_extent(grid, origin, res)[2:])
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color("#c8cdd2")

    outcome = "success" if ep.get("success") else "failure"
    ax.set_title(f"{scene} · target \"{ep.get('target')}\" · {ep.get('steps')} steps · "
                 f"{outcome} (SPL {float(ep.get('spl') or 0.0):.2f})",
                 fontsize=9.5, color=INK, pad=8)

    legend = [
        Line2D([], [], color=C_PATH, lw=3,
               linestyle="-" if kind == "dense" else (0, (5, 3)),
               label="path" if kind == "dense" else "decision polyline"),
        Line2D([], [], marker="o", ms=9, color=C_ROBOT, linestyle="none", label="start"),
        Line2D([], [], marker="o", ms=10, markerfacecolor=BELIEVED, color=BELIEVED,
               linestyle="none", label=r"track believed ($p\geq0.45$)"),
        Line2D([], [], marker="o", ms=10, markerfacecolor="white", color=BELIEVED,
               linestyle="none", label=r"track disproved ($p<0.45$)"),
        Line2D([], [], marker="o", ms=7, markerfacecolor=DISPROVED, color="none",
               linestyle="none", label="other tracks (radius $\\propto p$)"),
        Line2D([], [], marker="P", ms=10, color=C_COMMIT, linestyle="none",
               label="committed goal"),
        Line2D([], [], marker="*", ms=14, color=C_TARGET, linestyle="none",
               label="true target"),
    ]
    ax.legend(handles=legend, fontsize=7.2, frameon=True, framealpha=0.94,
              loc="lower left", borderpad=0.6, labelspacing=0.5)

    draw_mass_panel(axm, ep)
    axm.set_title("presence mass per storey", fontsize=9.5, color=INK, pad=8)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  episode  {ep['episode_id']}  storey {floor_key}  route={kind} "
          f"({len(route)} pts)  target tracks drawn={len(drawn)}")


if __name__ == "__main__":
    import matplotlib.patheffects  # noqa: F401  (used by draw_beliefs)
    main()
