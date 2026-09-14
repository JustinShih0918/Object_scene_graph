"""Persist the ASCENT obstacle-map stack so a later pass can start from it.

`osg/graph/map_store.py` stores the OSG side of a pass -- object tracks, their
presence beliefs, the room segmentation, the per-storey occupancy the OSG
planner drives. This stores the other half: the maps `Map_Controller` builds
(`navigation/mapping/obstacle_map.py`), one `ObstacleMap` per storey, exactly
as the transcription leaves them at the end of the episode.

The two are written side by side, keyed by scene, because the dynamic-scene
protocol needs both and they are built in different frames. What makes them
reusable together is the **anchor**: ASCENT's maps live in an episodic frame
whose origin is the pose the episode started at and whose axes are rotated to
its facing (`geometry.EpisodeAnchor`, `ascent_policy.py:233-237`), while the
OSG snapshot is in world coordinates. So the anchor is stored with the grids,
and `apply_to_costmap` maps every cell back through it. The rotation is not a
detail to be approximated away: see that function.

What is stored is EVIDENCE, not conclusions. The obstacle, explored and stair
masks are what the depth camera actually witnessed; the navigable maps are a
dilation of the obstacle mask by the agent radius and are recomputed on load
rather than trusted, so a change to `agent_radius` cannot be silently frozen
into an old snapshot.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from osg.mapping.costmap import FREE, OCCUPIED, UNKNOWN

SCHEMA_VERSION = 1


class ObstacleStoreError(RuntimeError):
    """A snapshot cannot be written or read."""


# The boolean evidence layers, saved verbatim. `_navigable_map` and
# `_strict_navigable_map` are deliberately NOT here: they are a pure function of
# `_map` and the agent radius, and recomputing them on load keeps a snapshot
# honest when the radius changes.
_MASKS = (
    "_map",
    "explored_area",
    "_up_stair_map",
    "_down_stair_map",
    "_disabled_stair_map",
    "stair_boundary",
    "stair_boundary_goal",
)

# Point/segment geometry the stair controller and the planner read back.
_POINTS = (
    "_up_stair_start",
    "_up_stair_end",
    "_down_stair_start",
    "_down_stair_end",
    "_up_stair_frontiers",
    "_down_stair_frontiers",
    "_up_stair_frontiers_px",
    "_down_stair_frontiers_px",
    "frontiers",
    "_frontiers_px",
    "_disabled_frontiers_px",
)

# Flags that say what the floor has been shown to contain, and how long the
# agent spent on it. `_floor_num_steps` gates the reference's first-step
# detector skip (A11), so a reloaded floor must not claim to be on step 0.
_FLAGS = (
    "_has_up_stair",
    "_has_down_stair",
    "_explored_up_stair",
    "_explored_down_stair",
    "_this_floor_explored",
    "_done_initializing",
    "_look_for_downstair_flag",
    "_reinitialize_flag",
    "_tight_search_thresh",
    "_disable_end",
)


def _self_consistency(om) -> Dict[str, Any]:
    """How much of this floor's own trajectory lies on its own EXPLORED area.

    A DIAGNOSTIC, NOT A VALIDITY TEST, and the distinction cost a long
    investigation. It reads as though a low score meant a broken map, and it
    does not: `explored_area` is what the agent SAW, not where it walked.
    `obstacle_map.py:374` erases every cell within an agent-radius dilation of
    an obstacle on every step, and `reveal_fog_of_war` propagates only through
    navigable cells, so a stairwell interior is never marked at all. An agent
    in a corridor or on stairs is therefore standing on cells its own map
    deliberately excludes.

    Measured on 00800, one episode each: bowl 0.54/0.77, banana 0.00/0.18,
    pitcher 0.11 -- and pitcher never climbed a single step, so this does not
    even track stair time. Every one of those maps is correctly framed. Gating
    reuse on this number would reject all of them.

    What it is good for is comparing coverage between passes of the same scene,
    and for noticing that a pass spent most of its budget somewhere its map
    cannot represent.
    """
    poses = list(getattr(om, "_camera_positions", []) or [])
    if not poses:
        return {"traj_poses": 0, "traj_on_map": None}
    px = om._xy_to_px(np.asarray(poses, dtype=float).reshape(-1, 2))
    size = int(om.size)
    cover = np.asarray(om.explored_area, dtype=bool)
    for name in ("_up_stair_map", "_down_stair_map"):
        mask = getattr(om, name, None)
        if mask is not None:
            cover = cover | np.asarray(mask, dtype=bool)
    inside = ((px[:, 0] >= 0) & (px[:, 0] < size)
              & (px[:, 1] >= 0) & (px[:, 1] < size))
    hits = int(cover[px[inside, 0], px[inside, 1]].sum())
    return {"traj_poses": int(len(poses)),
            "traj_on_map": round(hits / float(len(poses)), 4)}


def _occupancy(om) -> np.ndarray:
    """The floor's occupancy in OSG's Costmap2D coding, still in ASCENT pixels.

    -1 unknown, 0 free, 100 occupied. The coding is OSG's because that is what
    consumes it; the FRAME is still ASCENT's episodic one, and turning those
    pixels into world cells is `apply_to_costmap`'s job -- it needs the anchor,
    and the anchor is a rotation, so there is no correct way to do it with an
    origin alone.
    """
    return np.where(om._map.astype(bool), OCCUPIED,
                    np.where(om.explored_area, FREE, UNKNOWN)).astype(np.int8)


def save_obstacle_maps(
    path: Path, agent, *, scene: str = "", layout_id: str = "",
    episode_id: str = "",
) -> Path:
    """Write every floor's `ObstacleMap`. `path` is the JSON; grids go beside it.

    Called with the ascentnav agent itself: it owns `_floors`, the list
    `Map_Controller` grows as stairs are discovered (`agent.py:_new_floor`),
    and `anchor`, the episode-start pose every pixel is measured from.
    """
    floors = list(getattr(agent, "_floors", []) or [])
    if not floors:
        raise ObstacleStoreError("agent has no floors to save")
    anchor = getattr(agent, "anchor", None)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    records: List[Dict[str, Any]] = []
    for index, floor in enumerate(floors):
        om = floor.get("obstacle")
        if om is None:
            raise ObstacleStoreError(f"floor {index} has no obstacle map")
        prefix = f"floor_{index}_"
        for name in _MASKS:
            value = getattr(om, name, None)
            if value is not None:
                # Packed: a 1600x1600 bool mask is 2.56 MB raw and seven of
                # them per floor, which is most of a snapshot's size.
                arrays[prefix + name] = np.packbits(np.asarray(value, dtype=bool))
        for name in _POINTS:
            value = getattr(om, name, None)
            if value is not None and np.size(value):
                arrays[prefix + name] = np.asarray(value, dtype=float)
        arrays[prefix + "occupancy"] = _occupancy(om)
        poses = list(getattr(om, "_camera_positions", []) or [])
        if poses:
            arrays[prefix + "traj_ep"] = np.asarray(poses, dtype=float).reshape(-1, 2)

        value_map = floor.get("value")
        if value_map is not None:
            # `ValueMap._map` is its CONFIDENCE channel (BaseMap's array), not
            # an occupancy mask -- namespaced so it cannot collide with the
            # obstacle map's `_map` under the same floor prefix.
            for name in ("_value_map", "_map"):
                value = getattr(value_map, name, None)
                if value is not None:
                    arrays[prefix + "value" + name] = np.asarray(value, dtype=np.float32)

        records.append({
            "index": index,
            "prefix": prefix,
            "size": int(om.size),
            "pixels_per_meter": float(om.pixels_per_meter),
            "resolution": 1.0 / float(om.pixels_per_meter),
            "episode_pixel_origin": [int(v) for v in om._episode_pixel_origin],
            "agent_radius": float(om.agent_radius),
            "min_height": float(om._min_height),
            "max_height": float(om._max_height),
            "floor_num_steps": int(getattr(om, "_floor_num_steps", 0)),
            "explored_cells": int(np.count_nonzero(om.explored_area)),
            "obstacle_cells": int(np.count_nonzero(om._map)),
            **_self_consistency(om),
            "flags": {name: bool(getattr(om, name, False)) for name in _FLAGS},
            "mask_shape": [int(om.size), int(om.size)],
            "value_map": value_map is not None,
        })

    npz_path = path.with_suffix(".npz")
    np.savez_compressed(npz_path, **arrays)
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "scene": scene,
                "layout_id": layout_id,
                # WHICH episode's map this is. The protocol runs several per
                # scene and keeps the one that explored most, so without this
                # a checker cannot tell a wrong frame from a pose another
                # episode walked and this one never mapped.
                "episode_id": episode_id,
                "grids": npz_path.name,
                "floor_index": int(getattr(agent, "_floor_idx", 0)),
                "anchor": None if anchor is None else {
                    "xy": [float(v) for v in anchor.xy],
                    "heading": float(anchor.heading),
                },
                "floors": records,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return path


def load_obstacle_maps(path: Path) -> Dict[str, Any]:
    """Read a snapshot back, masks unpacked to their stored shape."""
    path = Path(path)
    if not path.exists():
        raise ObstacleStoreError(f"no obstacle snapshot at {path}")
    blob = json.loads(path.read_text(encoding="utf-8"))
    npz_path = path.with_name(str(blob.get("grids") or path.with_suffix(".npz").name))
    if not npz_path.exists():
        raise ObstacleStoreError(f"snapshot {path} names missing grids {npz_path}")
    with np.load(npz_path) as data:
        arrays = {key: data[key] for key in data.files}
    for record in blob.get("floors", []):
        shape = tuple(int(v) for v in record["mask_shape"])
        count = int(np.prod(shape))
        for name in _MASKS:
            key = record["prefix"] + name
            if key in arrays:
                arrays[key] = np.unpackbits(
                    arrays[key], count=count
                ).astype(bool).reshape(shape)
    blob["_arrays"] = arrays
    return blob


def floor_summaries(blob: Dict[str, Any]) -> List[Dict[str, Any]]:
    """What each stored storey contains, bottom storey first.

    That IS the order `_floors` keeps: `agent.py` appends on a discovered
    up-stair and inserts at 0 on a down-stair, so index 0 is always the lowest
    storey the pass ever stood on.
    """
    out = []
    for record in blob.get("floors", []):
        flags = record.get("flags") or {}
        out.append({
            "index": int(record["index"]),
            "resolution": float(record["resolution"]),
            "traj_poses": int(record.get("traj_poses", 0)),
            "traj_on_map": record.get("traj_on_map"),
            "explored_cells": int(record.get("explored_cells", 0)),
            "obstacle_cells": int(record.get("obstacle_cells", 0)),
            "has_up_stair": bool(flags.get("_has_up_stair", False)),
            "has_down_stair": bool(flags.get("_has_down_stair", False)),
            "explored": bool(flags.get("_this_floor_explored", False)),
            "steps": int(record.get("floor_num_steps", 0)),
        })
    return out


def _anchor_from(blob: Dict[str, Any]):
    meta = blob.get("anchor")
    if not meta:
        return None
    from ..geometry import EpisodeAnchor

    return EpisodeAnchor(np.asarray(meta["xy"], dtype=float),
                         float(meta["heading"]))


def apply_to_costmap(
    blob: Dict[str, Any],
    index: int,
    costmap,
    *,
    overwrite: bool = False,
    margin_m: float = 2.0,
    floor_y: Optional[float] = None,
    next_floor_y: Optional[float] = None,
) -> int:
    """Paste a stored ASCENT floor into an OSG `Costmap2D`. Returns cells written.

    This is the whole point of the snapshot: the occupancy ASCENT's navigation
    built becomes the occupancy OSG's planner, frontier extractor and room
    segmenter read, so pass 2 searches a map it did not build.

    IT IS A RESAMPLE, NOT A PASTE. Three things separate the two lattices:
    `BaseMap._xy_to_px` swaps the axes and flips the row, OSG's PLANE is
    habitat's (x, z) where ASCENT's frame is (x, -z), and ASCENT's map is
    rotated to the episode's start facing (`geometry.EpisodeAnchor`,
    `ascent_policy.py:233-237`). Copying with a shifted origin -- which is all
    `_CostmapView` does, and it is only ever drawn, never navigated -- puts the
    walls 10 m out at heading 0 and 22 m out at heading pi/2 (measured). So
    every destination cell is mapped back through all three (nearest
    neighbour, exact at equal resolution).

    `overwrite=False` writes only where the destination is still UNKNOWN, so a
    live map keeps what it has seen for itself and gains what it has not.

    STAIRS COME ACROSS TOO, into `costmap.stair_mask`. ASCENT keeps staircases
    in `_up_stair_map` / `_down_stair_map` and out of `explored_area`, so the
    occupancy alone has a hole exactly where a cross-floor run needs it -- the
    00800 map holds 933 up-stair and 933 down-stair cells, and the poses that
    fell outside it were overwhelmingly climb states (278 of 296 on one
    episode). `Costmap2D` means the same thing by the field: cells confirmed
    traversable, written FREE and exempted from the OCCUPIED stamp
    (`mapping/costmap.py:36-40, 186-191`), which is how `mapping/stairs.py`
    writes it.
    """
    arrays = blob.get("_arrays")
    if arrays is None:
        raise ObstacleStoreError("blob was not produced by load_obstacle_maps")
    record = next((f for f in blob.get("floors", []) if int(f["index"]) == int(index)), None)
    if record is None:
        raise ObstacleStoreError(f"snapshot has no floor {index}")
    source = arrays.get(record["prefix"] + "occupancy")
    if source is None:
        raise ObstacleStoreError(f"floor {index} has no stored occupancy")

    resolution = float(record["resolution"])
    if abs(resolution - float(costmap.resolution)) > 1e-9:
        raise ObstacleStoreError(
            f"obstacle snapshot is {resolution} m/cell and this costmap is "
            f"{costmap.resolution}; resample before loading"
        )
    anchor = _anchor_from(blob)
    size = int(record["size"])
    epo = np.asarray(record["episode_pixel_origin"], dtype=float)
    ppm = float(record["pixels_per_meter"])

    # Everything this floor has evidence for. The stair masks are part of it
    # and lie OUTSIDE `explored_area` by construction, so a bounding box taken
    # from the occupancy alone clips the staircases off -- which is the one
    # region a cross-floor pass cannot do without.
    stair_masks = [
        arrays[record["prefix"] + name]
        for name in ("_up_stair_map", "_down_stair_map")
        if record["prefix"] + name in arrays
    ]
    covered = source != UNKNOWN
    for mask in stair_masks:
        covered = covered | mask

    # Grow the destination to cover what was mapped, before writing into it.
    known = np.argwhere(covered)
    if known.size == 0:
        return 0
    corners_px = np.array([
        [known[:, 0].min(), known[:, 1].min()], [known[:, 0].min(), known[:, 1].max()],
        [known[:, 0].max(), known[:, 1].min()], [known[:, 0].max(), known[:, 1].max()],
    ], dtype=float)
    for row, col in corners_px:
        costmap.ensure_contains(
            _px_to_world(row, col, size, epo, ppm, anchor), margin_m=margin_m
        )

    # Only the window the source can actually reach. The destination grows by
    # doubling and a late-episode costmap is mostly empty, so meshing all of it
    # would allocate tens of megabytes to look up cells no storey covers.
    dest_corners = np.array([
        costmap.world_to_grid(_px_to_world(row, col, size, epo, ppm, anchor))
        for row, col in corners_px
    ])
    r0 = max(0, int(dest_corners[:, 0].min()) - 1)
    r1 = min(costmap.grid.shape[0], int(dest_corners[:, 0].max()) + 2)
    c0 = max(0, int(dest_corners[:, 1].min()) - 1)
    c1 = min(costmap.grid.shape[1], int(dest_corners[:, 1].max()) + 2)
    if r0 >= r1 or c0 >= c1:
        return 0

    # Inverse map: every destination cell centre -> the source pixel it came
    # from. Forward-mapping the source would leave holes wherever the rotation
    # spreads neighbouring pixels apart.
    rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
    world_x = costmap.origin[0] + (rr + 0.5) * resolution
    world_z = costmap.origin[1] + (cc + 0.5) * resolution
    # OSG's PLANE is (x, z) of habitat's y-up world; ASCENT's CCW frame is
    # (x, -z) -- the same flip `geometry.robot_xy_heading` applies.
    ascent_xy = np.stack([world_x, -world_z], axis=-1)
    if anchor is not None:
        # `EpisodeAnchor.to_episodic`, one matrix multiply for the whole
        # window instead of a Python call per cell.
        flat = ascent_xy.reshape(-1, 2) - anchor.xy
        ep = flat @ anchor._r_world_to_ep.T
        ascent_xy = ep.reshape(rr.shape[0], rr.shape[1], 2)
    src_col = np.rint(ascent_xy[..., 0] * ppm + epo[1]).astype(int)
    src_row = np.rint(size - (ascent_xy[..., 1] * ppm + epo[0])).astype(int)
    inside = (
        (src_row >= 0) & (src_row < size) & (src_col >= 0) & (src_col < size)
    )
    values = np.full(rr.shape, UNKNOWN, dtype=np.int8)
    values[inside] = source[src_row[inside], src_col[inside]]

    window = costmap.grid[r0:r1, c0:c1]
    writable = values != UNKNOWN
    if not overwrite:
        writable &= window == UNKNOWN
    window[writable] = values[writable]

    # The stair evidence, sampled through the same inverse map. Written FREE as
    # well as into the mask: `Costmap2D` treats a stair cell as traversable,
    # and a staircase left UNKNOWN is a hole the planner will not cross.
    stairs = np.zeros(rr.shape, dtype=bool)
    for mask in stair_masks:
        stairs |= mask[src_row.clip(0, size - 1), src_col.clip(0, size - 1)] & inside
    if stairs.any():
        if costmap.stair_mask is None:
            costmap.stair_mask = np.zeros(costmap.grid.shape, dtype=bool)
        costmap.stair_mask[r0:r1, c0:c1] |= stairs
        window[stairs] = FREE
        _ramp_stair_heights(
            costmap, stairs, r0, c0, floor_y, next_floor_y,
            near_rc=_stair_base_rc(
                arrays, record, costmap, size, epo, ppm, anchor, r0, c0,
                floor_y, next_floor_y,
            ),
        )

    return int(np.count_nonzero(writable))


def _stair_base_rc(arrays, record, costmap, size, epo, ppm, anchor,
                   r0, c0, floor_y, next_floor_y):
    """The flight's two ends, this-storey first, in window coordinates.

    ASCENT records the two ends of each staircase, and they are the one piece
    of direction the tread cloud itself does not carry. `_up_stair_start` is
    where an ascent begins (this storey) and `_up_stair_end` where it arrives;
    a descent names them the other way round, and the two storeys' records
    agree -- measured on 00800, floor 1's `_up_stair_start` and floor 2's
    `_down_stair_end` are the SAME pixel, as are the other two.

    THE CONVENTION IS (x, y) = (col, row), not (row, col), which is the
    opposite of the masks beside it. Read the wrong way round on 00800 these
    points land 10 m from the mask they belong to; read this way they sit
    within 0.4 m of its ends.
    """
    prefix = record["prefix"]
    ascending = (
        floor_y is not None and next_floor_y is not None
        and float(next_floor_y) > float(floor_y)
    )
    kind = "_up_stair" if ascending else "_down_stair"
    out = []
    for suffix in ("_start", "_end"):
        point = arrays.get(prefix + kind + suffix)
        if point is None or np.size(point) < 2:
            return None
        px = np.asarray(point, dtype=float).ravel()
        world = _px_to_world(px[1], px[0], size, epo, ppm, anchor)  # (col, row)
        rc = costmap.world_to_grid(world)
        out.append(np.array([float(rc[0]) - r0, float(rc[1]) - c0], dtype=float))
    return out[0], out[1]


def _ramp_stair_heights(costmap, stairs, r0, c0, floor_y, next_floor_y,
                        *, near_rc=None) -> None:
    """Give the pasted treads a HEIGHT, so OSG can see a staircase at all.

    This is the seam that carries ASCENT's multi-floor navigation into OSG's.
    `mapping/stairs.find_flights` -- which feeds `FloorPolicy.pursuit_flight`
    and therefore `NavAgent._flight_carrot`, the waypoint climber -- reads
    `costmap.height` and returns [] the moment it is None. It looks for cells
    whose height sits strictly BETWEEN two storeys. ASCENT's `ObstacleMap` is
    2D and stores no heights, so pasting occupancy and a stair mask alone
    leaves those cells NaN, no flight is ever found, `pursuit_flight` stays
    None and the carrot silently never fires -- measured: zero
    `climb_flight_carrot` counters across a whole cross-anchor run.

    So the treads are given a linear ramp from this storey to the next, along
    each cell's projection onto the flight's own principal axis. The heights
    are SYNTHETIC and say so: ASCENT knows where the staircase is, not how
    tall each tread is. What they have to be is monotonic and spanning, which
    is all `find_flights` asks of them (`min_span_m`, and strictly between the
    two storeys).
    """
    if floor_y is None or next_floor_y is None or costmap.height is None:
        return
    lo, hi = float(floor_y), float(next_floor_y)
    if abs(hi - lo) < 1e-6:
        return
    rc = np.argwhere(stairs)
    if rc.shape[0] < 2:
        return
    pts = rc.astype(float)
    centred = pts - pts.mean(axis=0)
    axis = None
    if near_rc is not None:
        # ASCENT's own two stair endpoints, low end first. This is the whole
        # direction signal and there is no other: measured on 00800 the
        # `_up_stair_map` is a 3.2 x 2.0 m BLOB over the stairwell, not a long
        # thin flight, so its first singular vector is not the climb direction
        # at all -- the recorded start projects to t=-4.5 in a range of
        # [-30.5, 36.2], the middle of the run rather than an end.
        base, top = np.asarray(near_rc[0], float), np.asarray(near_rc[1], float)
        delta = top - base
        norm = float(np.linalg.norm(delta))
        if norm > 1.0:            # ends a cell apart say nothing about direction
            axis = delta / norm
    if axis is None:
        # No endpoints stored: fall back to the tread cloud's principal axis,
        # which gets the SPAN right and the sign only by luck. `find_flights`
        # asks for monotonic and spanning, so a flight is still found; whether
        # its foot is the real foot is then unknown.
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        axis = vt[0]
    t = centred @ axis
    span = float(t.max() - t.min())
    if span <= 0:
        return
    frac = (t - t.min()) / span
    # Keep the ends strictly inside the two storeys: `find_flights` excludes
    # cells within `margin_m` of either, and a tread exactly AT floor level is
    # indistinguishable from the floor.
    heights = lo + (hi - lo) * (0.12 + 0.76 * frac)
    costmap.height[rc[:, 0] + r0, rc[:, 1] + c0] = heights.astype(np.float32)


def _px_to_world(row, col, size, epo, ppm, anchor) -> np.ndarray:
    """One ASCENT pixel -> the OSG world (x, z) of its centre."""
    x_ep = (float(col) - epo[1]) / ppm
    y_ep = (size - float(row) - epo[0]) / ppm
    xy = np.array([x_ep, y_ep], dtype=float)
    if anchor is not None:
        xy = anchor.to_world(xy)
    return np.array([xy[0], -xy[1]], dtype=float)


def apply_obstacle_maps(agent, blob: Dict[str, Any]) -> int:
    """Restore a saved stack onto a fresh ascentnav agent. Returns floors loaded.

    The navigable maps are recomputed from the restored obstacle mask by the
    agent's own kernels rather than read from the file, so the reloaded map
    obeys the radius THIS run is configured with.
    """
    arrays = blob.get("_arrays")
    if arrays is None:
        raise ObstacleStoreError("blob was not produced by load_obstacle_maps")
    records = list(blob.get("floors", []))
    if not records:
        return 0

    # Grow the stack to the stored height before filling it: `_new_floor` is
    # the only thing that knows how to build a floor for THIS config.
    while len(agent._floors) < len(records):
        agent._floors.append(agent._new_floor())

    for record in records:
        om = agent._floors[int(record["index"])]["obstacle"]
        prefix = record["prefix"]
        for name in _MASKS:
            key = prefix + name
            current = getattr(om, name, None)
            if key in arrays and current is not None:
                stored = arrays[key]
                if stored.shape == np.asarray(current).shape:
                    setattr(om, name, stored.copy())
        for name in _POINTS:
            key = prefix + name
            if key in arrays:
                setattr(om, name, arrays[key].copy())
        for name, value in (record.get("flags") or {}).items():
            if hasattr(om, name):
                setattr(om, name, bool(value))
        om._floor_num_steps = int(record.get("floor_num_steps", 0))
        _recompute_navigable(om)

    agent._floor_idx = min(int(blob.get("floor_index", 0)), len(agent._floors) - 1)
    planner = getattr(agent, "planner", None)
    if planner is not None and hasattr(planner, "floor_num"):
        planner.floor_num = len(agent._floors)
    return len(records)


def _recompute_navigable(om) -> None:
    """`ObstacleMap.update_map`'s dilation step, on a restored obstacle mask."""
    import cv2

    obstacles = om._map.astype(np.uint8)
    om._navigable_map = (
        1 - cv2.dilate(obstacles, om._navigable_kernel, iterations=1)
    ).astype(bool)
    om._strict_navigable_map = (
        1 - cv2.dilate(obstacles, om._strict_navigable_kernel, iterations=1)
    ).astype(bool)
