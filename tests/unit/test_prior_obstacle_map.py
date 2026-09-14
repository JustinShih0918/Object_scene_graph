"""Handing a stored ASCENT obstacle map to the OSG pipeline.

The shapes here are taken from a real pass, not invented: the first 00800
snapshot had FOUR storeys of which two were empty, because `_new_floor`
allocates a storey the moment a staircase is detected and the agent never
walked two of them. Floor 0 was one of the empty ones, so naive order-matching
handed a fresh agent a map with nothing in it and called that a success.
"""
from __future__ import annotations

import numpy as np
import pytest

from navigation.geometry import EpisodeAnchor
from navigation.mapping.map_store import save_obstacle_maps
from navigation.mapping.obstacle_map import ObstacleMap
from osg.eval.prior_map import load_obstacle_map
from osg.mapping.costmap import Costmap2D

SIZE, PPM = 400, 20


def _om(explored=None, occupied=None) -> ObstacleMap:
    om = ObstacleMap(min_height=0.61, max_height=0.88, agent_radius=0.18,
                     area_thresh=1.5, hole_area_thresh=100000, size=SIZE)
    if explored:
        r0, r1, c0, c1 = explored
        om.explored_area[r0:r1, c0:c1] = True
    if occupied:
        r0, r1, c0, c1 = occupied
        om._map[r0:r1, c0:c1] = True
    return om


class _SaveAgent:
    def __init__(self, floors, anchor, floor_idx=0):
        self._floors = [{"obstacle": om} for om in floors]
        self.anchor = anchor
        self._floor_idx = floor_idx


class _Layer:
    def __init__(self, floor_y, costmap):
        self.floor_y = floor_y
        self.costmap = costmap


class _Stack:
    def __init__(self, layers):
        self._layers = layers


class _LoadAgent:
    """Only what `load_obstacle_map` touches on a NavAgent."""

    def __init__(self, layers):
        self._floor_stack = _Stack(layers)


class _Cfg:
    class ycb:
        obstacle_map_in = ""
        obstacle_map_out = ""
        obstacle_map_overwrite = False


def _cfg(map_in: str, overwrite: bool = False):
    cfg = _Cfg()
    cfg.ycb.obstacle_map_in = str(map_in)
    cfg.ycb.obstacle_map_out = ""
    cfg.ycb.obstacle_map_overwrite = overwrite
    return cfg


def _four_floors(tmp_path):
    """Two empty placeholders around two real storeys -- the 00800 shape."""
    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)
    floors = [
        _om(),                                                  # 0: placeholder
        _om(explored=(150, 250, 150, 250), occupied=(160, 170, 160, 170)),
        _om(explored=(180, 230, 180, 230), occupied=(190, 195, 190, 195)),
        _om(),                                                  # 3: placeholder
    ]
    save_obstacle_maps(tmp_path / "00800-TEEsavR23oF.json",
                       _SaveAgent(floors, anchor, floor_idx=2),
                       scene="00800-TEEsavR23oF", layout_id="static")
    return tmp_path


def test_a_fresh_single_floor_agent_gets_a_mapped_storey(tmp_path):
    """The regression: floor 0 of that snapshot is EMPTY, and a fresh agent has
    exactly one layer. Matching by raw order loads nothing and says so."""
    maps = _four_floors(tmp_path)
    costmap = Costmap2D(resolution=1.0 / PPM, size_m=20.0)
    agent = _LoadAgent({0: _Layer(0.0, costmap)})

    note = load_obstacle_map(_cfg(maps), agent, "00800-TEEsavR23oF")

    assert note is not None
    assert note["stored_floors"] == 4
    assert note["mapped_floors"] == 2
    assert note["cells_written"] > 0, "a fresh agent was handed an empty map"
    assert note["matched_floors"][0]["ascent_floor"] == 1, "skipped the placeholder"
    assert (costmap.grid != -1).any()


def test_two_storeys_match_bottom_up(tmp_path):
    maps = _four_floors(tmp_path)
    lower = Costmap2D(resolution=1.0 / PPM, size_m=20.0)
    upper = Costmap2D(resolution=1.0 / PPM, size_m=20.0)
    # Deliberately out of key order: the match is by HEIGHT, not by key.
    agent = _LoadAgent({7: _Layer(3.1, upper), 2: _Layer(0.0, lower)})

    note = load_obstacle_map(_cfg(maps), agent, "00800-TEEsavR23oF")

    assert [p["ascent_floor"] for p in note["matched_floors"]] == [1, 2]
    assert [p["osg_floor"] for p in note["matched_floors"]] == [2, 7]
    assert all(p["cells"] > 0 for p in note["matched_floors"])


def test_no_configured_input_is_not_an_error(tmp_path):
    agent = _LoadAgent({0: _Layer(0.0, Costmap2D(resolution=0.05, size_m=20.0))})
    assert load_obstacle_map(_cfg(""), agent, "00800-TEEsavR23oF") is None


def test_a_missing_snapshot_is_loud(tmp_path):
    """Silently navigating without the map under test is the failure mode that
    produces a number nobody can interpret."""
    agent = _LoadAgent({0: _Layer(0.0, Costmap2D(resolution=0.05, size_m=20.0))})
    with pytest.raises(FileNotFoundError):
        load_obstacle_map(_cfg(tmp_path), agent, "00800-TEEsavR23oF")


def test_more_stored_storeys_than_the_agent_has_is_reported(tmp_path):
    maps = _four_floors(tmp_path)
    agent = _LoadAgent({0: _Layer(0.0, Costmap2D(resolution=1.0 / PPM, size_m=20.0))})

    note = load_obstacle_map(_cfg(maps), agent, "00800-TEEsavR23oF")

    assert note["unmatched_floors"] == 1
    assert len(note["matched_floors"]) == 1


# --------------------------------------------------------- which map is kept

def test_the_kept_map_is_the_one_that_walked_more_storeys(tmp_path):
    """Ranking on `len(_floors)` would keep the wrong episode.

    `_new_floor` allocates a storey the moment a staircase is DETECTED, so an
    episode that glimpsed three staircases and explored one room carries four
    entries while one that mapped the whole building carries two.
    """
    from osg.eval.prior_map import save_obstacle_map_for_scene

    class _Episode:
        episode_id = "0"
        scene_id = "data/scene_datasets/hm3d/val/00800-x/x.basis.glb"

    cfg = _cfg("")
    cfg.ycb.obstacle_map_out = str(tmp_path)
    episode = _Episode()

    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)
    # First: four storeys, three of them empty placeholders, one small room.
    glimpser = _SaveAgent([_om(explored=(100, 110, 100, 110)), _om(), _om(), _om()],
                          anchor)
    # Second: two storeys, both genuinely walked, far more of the building.
    mapper = _SaveAgent([_om(explored=(100, 300, 100, 300)),
                         _om(explored=(120, 260, 120, 260))], anchor)

    save_obstacle_map_for_scene(cfg, glimpser, episode)
    save_obstacle_map_for_scene(cfg, mapper, episode)

    from navigation.mapping.map_store import floor_summaries, load_obstacle_maps

    written = list(tmp_path.glob("*.json"))
    assert len(written) == 1, f"expected one snapshot, got {written}"
    kept = [f for f in floor_summaries(
        load_obstacle_maps(written[0])) if f["explored_cells"] > 0]
    assert len(kept) == 2, "kept the episode that only saw staircases"
    assert sum(f["explored_cells"] for f in kept) > 40000


def test_a_policy_without_an_obstacle_map_writes_nothing(tmp_path):
    """`nav_agent` has no `_floors`; an empty snapshot loads silently."""
    from osg.eval.prior_map import save_obstacle_map_for_scene

    cfg = _cfg("")
    cfg.ycb.obstacle_map_out = str(tmp_path)

    class _NoFloors:
        pass

    class _Episode:
        episode_id = "0"
        scene_id = "data/scene_datasets/hm3d/val/00800-x/x.basis.glb"

    assert save_obstacle_map_for_scene(cfg, _NoFloors(), _Episode()) is None
    assert not list(tmp_path.glob("*.json"))


# ------------------------------------------------- the map's own diagnostic

def test_traj_on_map_is_recorded_but_does_not_gate_reuse(tmp_path):
    """`traj_on_map` is a coverage diagnostic, not a validity test.

    It reads like one, which is why this test exists. `explored_area` is what
    the agent SAW: obstacle_map.py:374 erases an agent-radius band around every
    obstacle each step, so a real pass scores far below 1.0 with a perfectly
    good map (measured on 00800: bowl 0.54, pitcher 0.11, and pitcher never
    climbed). A map must still load.
    """
    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)
    om = _om(explored=(150, 250, 150, 250))
    om._camera_positions = [np.array([12.0, 12.0]) for _ in range(50)]  # all off it

    save_obstacle_maps(tmp_path / "00800-TEEsavR23oF.json",
                       _SaveAgent([om], anchor), scene="00800-TEEsavR23oF")
    agent = _LoadAgent({0: _Layer(0.0, Costmap2D(resolution=1.0 / PPM, size_m=20.0))})

    note = load_obstacle_map(_cfg(tmp_path), agent, "00800-TEEsavR23oF")

    assert note is not None, "a low traj_on_map must not block reuse"
    assert note["traj_on_map"] == [0.0], "but it must still be reported"
    assert note["cells_written"] > 0


def test_traj_on_map_is_one_when_the_path_is_on_the_explored_area(tmp_path):
    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)
    om = _om(explored=(150, 250, 150, 250))
    om._camera_positions = [np.array([-1.0, 1.0]) for _ in range(50)]

    save_obstacle_maps(tmp_path / "00800-TEEsavR23oF.json",
                       _SaveAgent([om], anchor), scene="00800-TEEsavR23oF")
    from navigation.mapping.map_store import floor_summaries, load_obstacle_maps

    summary = floor_summaries(load_obstacle_maps(tmp_path / "00800-TEEsavR23oF.json"))
    assert summary[0]["traj_poses"] == 50
    assert summary[0]["traj_on_map"] == 1.0


def test_an_old_snapshot_without_the_diagnostic_still_loads(tmp_path):
    import json

    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)
    om = _om(explored=(150, 250, 150, 250))
    p = save_obstacle_maps(tmp_path / "00800-TEEsavR23oF.json",
                           _SaveAgent([om], anchor), scene="00800-TEEsavR23oF")
    blob = json.loads(p.read_text())
    for f in blob["floors"]:
        f.pop("traj_on_map", None); f.pop("traj_poses", None)
    p.write_text(json.dumps(blob))

    agent = _LoadAgent({0: _Layer(0.0, Costmap2D(resolution=1.0 / PPM, size_m=20.0))})
    note = load_obstacle_map(_cfg(tmp_path), agent, "00800-TEEsavR23oF")
    assert note["traj_on_map"] == [None]


def test_the_bigger_map_is_kept(tmp_path):
    """Retention ranks on CELLS, storeys only as a tie-break."""
    from osg.eval.prior_map import save_obstacle_map_for_scene
    from navigation.mapping.map_store import floor_summaries, load_obstacle_maps

    class _Episode:
        episode_id = "0"
        scene_id = "data/scene_datasets/hm3d/val/00800-x/x.basis.glb"

    cfg = _cfg("")
    cfg.ycb.obstacle_map_out = str(tmp_path)
    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)

    small = _om(explored=(150, 250, 150, 250))
    big = _om(explored=(100, 400, 100, 400))
    save_obstacle_map_for_scene(cfg, _SaveAgent([small], anchor), _Episode())
    save_obstacle_map_for_scene(cfg, _SaveAgent([big], anchor), _Episode())

    kept = floor_summaries(load_obstacle_maps(next(tmp_path.glob("*.json"))))
    assert kept[0]["explored_cells"] == 300 * 300


def test_a_policy_without_an_obstacle_map_writes_nothing(tmp_path):
    """`nav_agent` has no `_floors`; an empty snapshot loads silently."""
    from osg.eval.prior_map import save_obstacle_map_for_scene

    cfg = _cfg("")
    cfg.ycb.obstacle_map_out = str(tmp_path)

    class _NoFloors:
        pass

    class _Episode:
        episode_id = "0"
        scene_id = "data/scene_datasets/hm3d/val/00800-x/x.basis.glb"

    assert save_obstacle_map_for_scene(cfg, _NoFloors(), _Episode()) is None
    assert not list(tmp_path.glob("*.json"))


def test_a_wide_one_storey_map_beats_a_tiny_multi_storey_one(tmp_path):
    """The 00821 regression.

    An episode that ran 215 steps over a 3.4 x 1.2 m box while touching four
    storeys beat episodes covering 10 x 28 m, because storeys were ranked
    before cells -- the scene's stored map came out at 2975 cells against
    30-36k for its neighbours. A storey stepped onto is not a storey mapped.
    """
    from osg.eval.prior_map import save_obstacle_map_for_scene
    from navigation.mapping.map_store import floor_summaries, load_obstacle_maps

    class _Episode:
        episode_id = "0"
        scene_id = "data/scene_datasets/hm3d/val/00821-x/x.basis.glb"

    cfg = _cfg("")
    cfg.ycb.obstacle_map_out = str(tmp_path)
    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)

    # Four storeys, a few cells each: the episode that used to win.
    tiny = [_om(explored=(150 + 10 * i, 155 + 10 * i, 150, 158)) for i in range(4)]
    # One storey, the whole building.
    wide = [_om(explored=(100, 400, 100, 400))]

    save_obstacle_map_for_scene(cfg, _SaveAgent(tiny, anchor), _Episode())
    save_obstacle_map_for_scene(cfg, _SaveAgent(wide, anchor), _Episode())

    kept = [f for f in floor_summaries(load_obstacle_maps(next(tmp_path.glob("*.json"))))
            if f["explored_cells"] > 0]
    assert sum(f["explored_cells"] for f in kept) == 300 * 300, (
        "the tiny four-storey map displaced the one that mapped the building")
