#!/usr/bin/env python3
"""The layered scene graph over the real scene: building -> room -> object.

Bottom layer is an orthographic top-down render of the HM3D mesh, so the floor
under the graph is the actual scene, not a schematic.  Above it sit the object
ellipsoids exactly where the map puts them, each on a stem up to its object
node; object nodes group into room nodes, and the rooms into one building node.

The hierarchy is rebuilt by the agent's own `SceneGraph.rebuild`, and the
ellipsoids are the tracks' own fitted ellipsoids -- nothing is re-derived for
the picture.

    python scripts/render_scene_graph_3d.py --scene 00848-ziup5kvtCCR
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np

WORKSPACE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from osg.mapping.costmap import PLANE, HEIGHT_AXIS, FREE, UNKNOWN
from osg.core.paths import hm3d_scene_root

_spec = importlib.util.spec_from_file_location(
    "_rsg", WORKSPACE / "scripts/render_scene_graph.py")
_rsg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rsg)
load_graph = _rsg.load_graph

# Layer heights, in world metres above the floor.  Chosen so the stems are long
# enough to read as a hierarchy without dwarfing the rooms.
H_OBJECT = 3.6
H_ROOM = 5.2
H_BUILDING = 6.6

ROOM_EDGE_COLORS = ["#f0a500", "#29b6d6", "#e05fa8", "#c5a3e0", "#8bc34a",
                    "#ff7043", "#5c6bc0", "#26a69a"]
OBJ_NODE = "#e02b20"
ROOM_NODE = "#1f8a2f"
BLD_NODE = "#3b3f8f"
PATH_COLOR = "#12d612"


def render_topdown(scene: str, res: int, pad_m: float, cache: pathlib.Path):
    """Orthographic bird's-eye RGB of the scene mesh, plus its world extent.

    habitat's ORTHOGRAPHIC sensor covers ``1 / ortho_scale`` metres across the
    frame, independent of resolution (measured: 0.05 -> 20 m, 0.1 -> 10 m,
    0.2 -> 5 m).  World x grows with the column index and world z with the row
    index, so the image drops straight into an origin="lower" extent.
    """
    if cache.exists():
        blob = np.load(cache)
        return blob["img"], tuple(float(v) for v in blob["extent"]), float(blob["floor_y"])

    import habitat_sim

    root = pathlib.Path(hm3d_scene_root())
    stem = scene.split("-", 1)[1]
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(root / f"val/{scene}/{stem}.basis.glb")
    cfg.scene_dataset_config_file = str(root / "hm3d_annotated_basis.scene_dataset_config.json")
    cfg.enable_physics = False

    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "rgb"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
    sensor.resolution = [res, res]
    sensor.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    sensor.orientation = np.array([-np.pi / 2, 0.0, 0.0], dtype=np.float32)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]

    # One throwaway sim to read the bounds, because ortho_scale has to be known
    # before the sensor is built.
    sensor.ortho_scale = 0.05
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
    lo, hi = sim.pathfinder.get_bounds()
    verts = np.asarray(sim.pathfinder.build_navmesh_vertices())
    hist = np.histogram(verts[:, 1], bins=60)
    floor_y = float(hist[1][int(np.argmax(hist[0]))])
    span = max(float(hi[0] - lo[0]), float(hi[2] - lo[2])) + 2 * pad_m
    sim.close()

    sensor.ortho_scale = 1.0 / span
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
    cx, cz = float((lo[0] + hi[0]) / 2), float((lo[2] + hi[2]) / 2)
    state = habitat_sim.AgentState()
    state.position = np.array([cx, float(hi[1]) + 6.0, cz], dtype=np.float32)
    sim.get_agent(0).set_state(state)
    img = np.asarray(sim.get_sensor_observations()["rgb"][..., :3], dtype=np.uint8)
    sim.close()

    extent = (cx - span / 2, cx + span / 2, cz - span / 2, cz + span / 2)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, img=img, extent=np.array(extent), floor_y=floor_y)
    return img, extent, floor_y


def draw_floor(ax, img, extent, z, *, n=220):
    """The render as a textured quad at the floor height."""
    x0, x1, z0, z1 = extent
    step = max(1, img.shape[0] // n)
    tile = img[::step, ::step] / 255.0
    h, w = tile.shape[:2]
    xs = np.linspace(x0, x1, w + 1)
    ys = np.linspace(z0, z1, h + 1)
    X, Y = np.meshgrid(xs, ys)
    Z = np.full_like(X, z)
    # HM3D renders the void black; drop it so the floor plan floats.
    alpha = (tile.sum(axis=2) > 0.06).astype(float)
    facecolors = np.concatenate([tile, alpha[..., None]], axis=2)
    ax.plot_surface(X, Y, Z, facecolors=facecolors, shade=False,
                    rstride=1, cstride=1, linewidth=0, antialiased=False,
                    zorder=0)


def draw_path(ax, grids, costmap, z):
    """The free-space skeleton: where the agent could actually walk."""
    from skimage.morphology import skeletonize, remove_small_objects

    free = grids["grid"] == FREE
    free = remove_small_objects(free, 64)
    skel = skeletonize(free)
    rows, cols = np.nonzero(skel)
    if rows.size == 0:
        return
    pts = np.stack([rows, cols], axis=1).astype(float)
    world = np.stack([costmap.grid_to_world(p) for p in pts])
    # 8-connectivity between skeleton cells becomes the drawn graph.
    index = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
    segs = []
    for (r, c), i in index.items():
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            j = index.get((r + dr, c + dc))
            if j is not None:
                segs.append((world[i], world[j]))
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    lines = [[(a[0], a[1], z), (b[0], b[1], z)] for a, b in segs]
    ax.add_collection3d(Line3DCollection(lines, colors=PATH_COLOR, linewidths=1.0,
                                         alpha=0.9, zorder=1))


def ellipsoid_surface(center, axes, R, n=14):
    u = np.linspace(0, 2 * np.pi, n * 2)
    v = np.linspace(0, np.pi, n)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    pts = np.stack([x, y, z], axis=-1) * np.asarray(axes, dtype=float)
    pts = pts @ np.asarray(R, dtype=float).T + np.asarray(center, dtype=float)
    return pts[..., 0], pts[..., 1], pts[..., 2]


def pick_objects(graph, layer, limit):
    """Show the objects the map is most sure of, plus every support surface."""
    surface_ids = {tid for node in graph.containers.values() for tid in node.track_ids}
    scored = []
    for obj in graph.objects:
        track = layer._tracks.get(obj.track_id)
        if track is None:
            continue
        scored.append((obj.track_id in surface_ids, float(track.evidence), obj, track))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [(o, t) for _, _, o, t in scored[:limit]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--maps-root", default=str(WORKSPACE / "outputs/maps_v5"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--top-res", type=int, default=1400)
    ap.add_argument("--pad-m", type=float, default=1.0)
    ap.add_argument("--max-objects", type=int, default=45)
    ap.add_argument("--lift", type=float, default=1.5,
                    help="raise the ellipsoid layer by this many metres")
    ap.add_argument("--elev", type=float, default=26.0)
    ap.add_argument("--azim", type=float, default=-72.0)
    ap.add_argument("--no-path", action="store_true")
    ap.add_argument("--dpi", type=int, default=350)
    args = ap.parse_args()

    blob, grids, costmap, layer, graph = load_graph(
        args.scene, pathlib.Path(args.maps_root))
    cache = WORKSPACE / f"outputs/figures/topdown_{args.scene}_{args.top_res}.npz"
    img, extent, floor_y = render_topdown(args.scene, args.top_res, args.pad_m, cache)

    fig = plt.figure(figsize=(14.0, 7.4))
    ax = fig.add_axes([0.11, -0.06, 0.92, 1.12], projection="3d",
                      computed_zorder=False)
    ax.set_proj_type("ortho")

    draw_floor(ax, img, extent, floor_y)
    if not args.no_path:
        draw_path(ax, grids, costmap, floor_y + 0.12)

    picked = pick_objects(graph, layer, args.max_objects)
    room_of = {o.track_id: o.room_id for o, _ in picked}

    cmap = plt.get_cmap("tab20")
    for i, (obj, track) in enumerate(picked):
        c = np.asarray(layer.center_of(track), dtype=float)
        c[HEIGHT_AXIS] += args.lift
        X, Y, Z = ellipsoid_surface(c, track.ellipsoid.axes, track.ellipsoid.R)
        colour = to_rgba(cmap(i % 20), 0.42)
        ax.plot_surface(X, Z, Y, color=colour, shade=True, linewidth=0,
                        rstride=1, cstride=1, antialiased=True, zorder=3)
        # world (x, y_up, z) -> plot (x, z, y_up)
        px, py, pz = c[PLANE[0]], c[PLANE[1]], c[HEIGHT_AXIS]
        ax.plot([px, px], [py, py], [pz, floor_y + H_OBJECT],
                color="#6b6b6b", lw=0.8, alpha=0.9, zorder=4)
        ax.scatter([px], [py], [floor_y + H_OBJECT], s=26, marker="s",
                   color=OBJ_NODE, depthshade=False, zorder=6)

    rooms = {}
    for obj, _ in picked:
        rooms.setdefault(obj.room_id, []).append(obj)
    rooms.pop(None, None)

    centroids = {}
    for k, (rid, objs) in enumerate(sorted(rooms.items())):
        node = graph.rooms.get(rid)
        if node is None:
            continue
        rx, ry = np.asarray(node.centroid_xy, dtype=float)
        centroids[rid] = (rx, ry)
        colour = ROOM_EDGE_COLORS[k % len(ROOM_EDGE_COLORS)]
        for obj in objs:
            c = np.asarray(obj.center, dtype=float)
            ax.plot([rx, c[PLANE[0]]], [ry, c[PLANE[1]]],
                    [floor_y + H_ROOM, floor_y + H_OBJECT],
                    color=colour, lw=0.7, alpha=0.85, zorder=5)
        ax.scatter([rx], [ry], [floor_y + H_ROOM], s=110, marker="s",
                   color=ROOM_NODE, edgecolors="#0d4d18", linewidths=0.6,
                   depthshade=False, zorder=7)

    if centroids:
        bx = float(np.mean([v[0] for v in centroids.values()]))
        by = float(np.mean([v[1] for v in centroids.values()]))
        for rx, ry in centroids.values():
            ax.plot([bx, rx], [by, ry], [floor_y + H_BUILDING, floor_y + H_ROOM],
                    color="#4a4a4a", lw=1.0, alpha=0.9, zorder=6)
        ax.scatter([bx], [by], [floor_y + H_BUILDING], s=190, marker="s",
                   color=BLD_NODE, edgecolors="#20224d", linewidths=0.8,
                   depthshade=False, zorder=8)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_zlim(floor_y - 0.2, floor_y + H_BUILDING + 0.25)
    ax.set_box_aspect((1, 1, 0.62))
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_axis_off()

    for y, text in ((0.87, "Building Node"), (0.71, "Room Node"),
                    (0.48, "Object Node\nand\nEllipsoid"), (0.18, "Path")):
        fig.text(0.012, y, text, fontsize=17, fontweight="bold", va="center",
                 family="serif")
        fig.text(0.012, y, text, fontsize=17, fontweight="bold", va="center",
                 family="serif", alpha=0)

    out = pathlib.Path(args.out) if args.out else (
        WORKSPACE / f"outputs/figures/scene_graph_3d_{args.scene}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Axes3D reserves a wide margin around its cube, so "tight" still
    # leaves most of the page blank.  Measure where the ink actually is
    # and re-save to that box.
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    ink = np.nonzero((buf < 250).any(axis=2))
    h, w = buf.shape[:2]
    pad = 12
    r0, r1 = max(ink[0].min() - pad, 0), min(ink[0].max() + pad, h - 1)
    c0, c1 = max(ink[1].min() - pad, 0), min(ink[1].max() + pad, w - 1)
    dpi = fig.dpi
    bbox = matplotlib.transforms.Bbox(
        [[c0 / dpi, (h - r1) / dpi], [c1 / dpi, (h - r0) / dpi]])
    fig.savefig(out, dpi=args.dpi, bbox_inches=bbox)
    fig.savefig(out.with_suffix(".png"), dpi=args.dpi, bbox_inches=bbox)
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.png')}")
    print(f"ellipsoids drawn={len(picked)} rooms={len(centroids)} "
          f"(map has {len(graph.objects)} objects, {len(graph.rooms)} rooms)")


if __name__ == "__main__":
    main()
