"""The two-pass protocol: navigate from a map the world has since invalidated.

This is what makes the benchmark a DYNAMIC-scene benchmark rather than an
incomplete-map one. Pass 1 explores the static layout and writes one snapshot
per scene; pass 2 runs the moved layout starting from that snapshot, so the map
the agent navigates with is genuinely stale -- it confidently says the object is
at A, and the object is at B.

The staleness IS the experiment. An agent that rebuilds from scratch every
episode is never wrong about anything, and measures nothing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .record import authored_episode_metadata, episode_tag, safe_tag


def authored_scene(episode) -> str:
    return str(authored_episode_metadata(episode).get("scene", "scene"))


def _map_path(root: str, scene: str) -> Path:
    return Path(str(root)) / f"{safe_tag(scene)}.json"


def save_obstacle_map_for_scene(cfg, agent, episode) -> Optional[str]:
    """Pass 1: keep the ASCENT obstacle-map stack this episode built.

    Only an agent that HAS one -- `navigation.AscentNavAgent` owns `_floors`
    and the episode anchor its pixels are measured from. Any other policy
    leaves this alone rather than writing an empty snapshot, because an empty
    snapshot loads silently and a missing one does not.
    """
    root = str(getattr(cfg.ycb, "obstacle_map_out", "") or "")
    if not root or not getattr(agent, "_floors", None):
        return None
    from navigation.mapping.map_store import save_obstacle_maps

    authored = authored_episode_metadata(episode)
    scene = authored_scene(episode)
    path = _map_path(root, scene)
    # Keep the pass that mapped the most: the protocol runs several episodes
    # per scene and the map worth keeping is the one that saw the most of it.
    #
    # Ranked on CELLS, with storeys only as a tie-break, and both counted over
    # storeys the agent actually WALKED -- `_new_floor` allocates one the
    # moment a staircase is merely detected, so empty storeys are evidence of
    # nothing. Ranking on storeys FIRST repeats that same mistake one layer up:
    # measured on 00821, an episode that ran 215 steps over a 3.4 x 1.2 m box
    # while touching four storeys beat episodes that covered 10 x 28 m, and the
    # scene's stored map came out at 2975 cells against 30-36k for its
    # neighbours. A storey stepped onto is not a storey mapped.
    if path.exists():
        try:
            from navigation.mapping.map_store import floor_summaries, load_obstacle_maps

            previous = [f for f in floor_summaries(load_obstacle_maps(path))
                        if f["explored_cells"] > 0]
            mine = [f["obstacle"] for f in agent._floors
                    if int(f["obstacle"].explored_area.sum()) > 0]
            old = (sum(f["explored_cells"] for f in previous), len(previous))
            new = (sum(int(om.explored_area.sum()) for om in mine), len(mine))
            if old >= new:
                return str(path)
        except Exception:
            # A corrupt prior must not stop a fresh valid snapshot replacing it.
            pass
    save_obstacle_maps(path, agent, scene=scene,
                       layout_id=str(authored.get("layout_id", "")),
                       episode_id=str(episode_tag(episode)))
    return str(path)


def load_obstacle_map(cfg, agent, scene: str) -> Optional[dict]:
    """Pass 2: plan over the occupancy ASCENT's navigation built.

    Floors are matched BY ORDER -- the snapshot's list runs bottom storey
    first (`agent.py` inserts at 0 on a down-stair), and the OSG stack is
    sorted by height -- because an `ObstacleMap` records no world height to
    match on. With a multi-storey `map_in` already loaded the two orders agree;
    on a fresh single-floor agent only the lowest MAPPED storey is restored,
    which is reported here rather than guessed at.

    EMPTY STOREYS ARE SKIPPED, and that is not a tidy-up. `_new_floor`
    allocates a storey the moment a staircase is DETECTED, so a real pass
    ends with placeholders above and below the storeys it actually walked --
    the first 00800 snapshot had four floors of which two held nothing, and
    floor 0 was one of them. Zipping those against the OSG stack hands a fresh
    agent an empty map and reports success, which is the one failure this
    whole path must not have.
    """
    root = str(getattr(cfg.ycb, "obstacle_map_in", "") or "")
    if not root:
        return None
    from navigation.mapping.map_store import (
        apply_to_costmap,
        floor_summaries,
        load_obstacle_maps,
    )

    path = _map_path(root, scene)
    if not path.exists():
        if str(getattr(cfg.ycb, "obstacle_map_out", "") or "") == root:
            return None  # pass 1 accumulating into its own output
        raise FileNotFoundError(f"no obstacle snapshot for {scene} at {path}")
    blob = load_obstacle_maps(path)
    all_floors = floor_summaries(blob)
    stored = [f for f in all_floors if f["explored_cells"] > 0]

    stack = agent._floor_stack
    keys = sorted(stack._layers, key=lambda k: float(
        getattr(stack._layers[k], "floor_y", 0.0)))
    overwrite = bool(getattr(cfg.ycb, "obstacle_map_overwrite", False))
    # The storey heights, so the pasted treads can be given a height ramp --
    # without it `find_flights` sees no staircase and the waypoint climber
    # never fires (navigation/mapping/map_store.py:_ramp_stair_heights).
    heights = [float(getattr(stack._layers[k], "floor_y", 0.0)) for k in keys]
    written, pairs = 0, []
    for i, (floor, key) in enumerate(zip(stored, keys)):
        # The storey a flight from here would arrive at: the one above, or for
        # the topmost the one below, since its staircase descends.
        nxt = heights[i + 1] if i + 1 < len(heights) else (
            heights[i - 1] if i > 0 else None)
        n = apply_to_costmap(blob, floor["index"], stack._layers[key].costmap,
                             overwrite=overwrite,
                             floor_y=heights[i], next_floor_y=nxt)
        written += n
        pairs.append({"ascent_floor": floor["index"], "osg_floor": int(key),
                      "cells": int(n)})
    return {
        "path": str(path),
        "from_layout": str(blob.get("layout_id", "")),
        "stored_floors": len(all_floors),
        "mapped_floors": len(stored),
        "traj_on_map": [f.get("traj_on_map") for f in stored],
        "matched_floors": pairs,
        "cells_written": int(written),
        "unmatched_floors": max(0, len(stored) - len(keys)),
    }


def save_map_for_scene(cfg, agent, episode) -> None:
    """Pass 1: keep the map this episode built, keyed by scene.

    Both artifacts come from one pass when the driving policy carries an OSG
    world model: `ascentnav` owns the `ObstacleMap` stack and its `world` owns
    the scene graph (`agent/world_model.py`), so the scene is walked once.
    """
    save_obstacle_map_for_scene(cfg, agent, episode)
    root = str(cfg.ycb.map_out or "")
    if not root:
        return
    # The snapshot is written from whatever holds the object layer: NavAgent
    # itself, or the world model a policy without one was given.
    agent = getattr(agent, "world", None) or agent
    if not hasattr(agent, "_floor_stack"):
        return
    from ..graph.map_store import load_map, save_map

    authored = authored_episode_metadata(episode)
    path = _map_path(root, authored_scene(episode))
    if path.exists():
        try:
            previous = load_map(path)
            old_grids = previous.get("_grids", {})
            old_known = sum(
                int((grid != -1).sum()) for name, grid in old_grids.items()
                if name.endswith("grid")
            )
            old_score = (
                len(previous.get("floors") or [0]), old_known,
                len(previous.get("tracks") or []),
            )
            new_score = (
                len(agent._floor_stack._layers),
                sum(int((floor.costmap.grid != -1).sum())
                    for floor in agent._floor_stack._layers.values()),
                len(list(agent.object_layer.tracks(include_blacklisted=True))),
            )
            if old_score > new_score:
                return
        except Exception:
            # A corrupt prior must not prevent a fresh valid snapshot replacing it.
            pass
    save_map(
        path,
        agent,
        scene=authored_scene(episode),
        layout_id=str(authored.get("layout_id", "")),
    )


def load_prior_map(
    cfg, agent, scene: str, *, initial_floor_y: Optional[float] = None
) -> Optional[dict]:
    """Pass 2: start from the map pass 1 built, not from an empty one."""
    root = str(cfg.ycb.map_in or "")
    if not root:
        return None
    from ..graph.map_store import apply_map, load_map

    path = _map_path(root, scene)
    if not path.exists() and str(cfg.ycb.map_out or "") == root:
        # Pass 1 accumulating into its own output: the first episode of a
        # scene has nothing to start from yet, and that is not an error.
        return None
    blob = load_map(path)
    restore_occupancy = bool(getattr(cfg.ycb, "map_in_occupancy", True))
    n = apply_map(
        agent, blob,
        max_log_odds=float(cfg.scene_graph.presence.reload_max_log_odds),
        initial_floor_y=initial_floor_y,
        restore_occupancy=restore_occupancy,
    )
    return {
        "path": str(path),
        "from_layout": str(blob.get("layout_id", "")),
        "tracks": int(n),
        "occupancy_restored": restore_occupancy,
        "schema_version": int(blob.get("schema_version", 1)),
        "floors": len(blob.get("floors") or [0]),
    }
