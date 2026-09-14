"""The ASCENT obstacle map, stored and reused by the OSG pipeline.

The transform is the whole risk here. ASCENT's maps are anchored at the pose
the episode started from and rotated to its facing; OSG's are world-axis
aligned. Every test below checks the loader's inverse against the FORWARD
projection the obstacle map itself uses (`BaseMap._xy_to_px`), because the two
being each other's inverse is the only property that makes a reloaded map
mean the same thing as the map that was saved.
"""
from __future__ import annotations

import numpy as np
import pytest

from navigation.geometry import EpisodeAnchor
from navigation.mapping.map_store import (
    apply_to_costmap,
    floor_summaries,
    load_obstacle_maps,
    save_obstacle_maps,
)
from navigation.mapping.obstacle_map import ObstacleMap
from osg.mapping.costmap import Costmap2D

SIZE = 400
PPM = 20


def _obstacle_map() -> ObstacleMap:
    return ObstacleMap(min_height=0.61, max_height=0.88, agent_radius=0.18,
                       area_thresh=1.5, hole_area_thresh=100000, size=SIZE)


class _Agent:
    """The two attributes `save_obstacle_maps` reads off the ascentnav agent."""

    def __init__(self, floors, anchor, floor_idx=0):
        self._floors = floors
        self.anchor = anchor
        self._floor_idx = floor_idx

    def _new_floor(self):
        return {"obstacle": _obstacle_map(), "value": None, "object": None}


def _world_to_px(om, anchor, world_xz):
    """The agent's own forward path: world -> ASCENT episodic -> pixel."""
    ascent_xy = np.array([world_xz[0], -world_xz[1]], dtype=float)
    ep = anchor.to_episodic(ascent_xy)
    return om._xy_to_px(np.atleast_2d(ep))[0]


@pytest.mark.parametrize("heading", [0.0, 0.7, -2.3, np.pi / 2])
def test_marked_cell_lands_at_its_world_coordinate(tmp_path, heading):
    """A pixel set occupied at world P reads back occupied at world P.

    Run at four start headings because a straight copy-with-offset -- which is
    what the runner's `_CostmapView` does -- is only correct at multiples of
    90 degrees. At 0.7 rad it puts the wall metres from where it was seen.
    """
    anchor = EpisodeAnchor(np.array([3.5, -2.0]), heading)
    om = _obstacle_map()
    targets = [np.array([4.0, -1.0]), np.array([1.25, 0.5]), np.array([6.0, -4.5])]
    for world_xz in targets:
        row, col = _world_to_px(om, anchor, world_xz)
        # A 3x3 block, not a single pixel: at equal resolution a rotation
        # resamples one lattice onto another, and a one-pixel feature can be
        # sampled by no destination cell at all. Real obstacles are walls.
        om._map[row - 1:row + 2, col - 1:col + 2] = True
        om.explored_area[row - 1:row + 2, col - 1:col + 2] = True
    # A witnessed free patch, so the floor has some known area around them.
    om.explored_area[SIZE // 2 - 40:SIZE // 2 + 40,
                     SIZE // 2 - 40:SIZE // 2 + 40] = True

    path = save_obstacle_maps(tmp_path / "scene.json", _Agent([{"obstacle": om}], anchor),
                              scene="scene", layout_id="static")
    blob = load_obstacle_maps(path)
    costmap = Costmap2D(resolution=1.0 / PPM, size_m=20.0)
    written = apply_to_costmap(blob, 0, costmap)

    assert written > 0
    occupied = np.argwhere(costmap.grid == 100)
    assert len(occupied) > 0, "nothing came back occupied"
    centres = np.array([costmap.grid_to_world(rc) for rc in occupied])
    for world_xz in targets:
        # The real quantity: how far the obstacle moved between the map that
        # was saved and the map that was loaded. Copy-with-offset misses by
        # 10 m at heading 0 and 22 m at pi/2; the resample must be sub-cell.
        displacement = float(np.min(np.linalg.norm(centres - world_xz, axis=1)))
        assert displacement < 1.5 / PPM, (
            f"{world_xz} moved {displacement:.3f} m at heading {heading}")


def test_free_and_unknown_survive_the_round_trip(tmp_path):
    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)
    om = _obstacle_map()
    om.explored_area[100:150, 100:150] = True   # seen, and empty
    om._map[120:130, 120:130] = True            # seen, and occupied

    path = save_obstacle_maps(tmp_path / "s.json", _Agent([{"obstacle": om}], anchor))
    blob = load_obstacle_maps(path)
    costmap = Costmap2D(resolution=1.0 / PPM, size_m=20.0)
    apply_to_costmap(blob, 0, costmap)

    assert (costmap.grid == 100).any(), "the occupied block did not survive"
    assert (costmap.grid == 0).any(), "the free block did not survive"
    assert (costmap.grid == -1).any(), "everything became known"
    # Areas the pass never witnessed must stay unknown, not become free.
    assert int((costmap.grid == 0).sum()) < int(costmap.grid.size)


def test_masks_and_flags_round_trip(tmp_path):
    anchor = EpisodeAnchor(np.array([1.0, 2.0]), 0.3)
    om = _obstacle_map()
    om.explored_area[10:20, 10:20] = True
    om._map[12:14, 12:14] = True
    om._up_stair_map[30:35, 30:35] = True
    om._has_up_stair = True
    om._this_floor_explored = True
    om._floor_num_steps = 137

    path = save_obstacle_maps(tmp_path / "s.json", _Agent([{"obstacle": om}], anchor))
    blob = load_obstacle_maps(path)

    fresh = _Agent([], anchor)
    fresh._floors = []
    from navigation.mapping.map_store import apply_obstacle_maps

    assert apply_obstacle_maps(fresh, blob) == 1
    restored = fresh._floors[0]["obstacle"]
    assert np.array_equal(restored._map, om._map)
    assert np.array_equal(restored.explored_area, om.explored_area)
    assert np.array_equal(restored._up_stair_map, om._up_stair_map)
    assert restored._has_up_stair is True
    assert restored._this_floor_explored is True
    assert restored._floor_num_steps == 137
    # Navigable maps are RECOMPUTED, never restored, so a changed agent radius
    # cannot be frozen into an old snapshot.
    assert restored._navigable_map.shape == om._map.shape
    assert not restored._navigable_map[12:14, 12:14].any()


def test_multiple_floors_keep_their_order(tmp_path):
    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)
    lower, upper = _obstacle_map(), _obstacle_map()
    lower.explored_area[50:60, 50:60] = True
    lower._has_up_stair = True
    upper.explored_area[70:90, 70:90] = True
    upper._has_down_stair = True

    path = save_obstacle_maps(
        tmp_path / "s.json",
        _Agent([{"obstacle": lower}, {"obstacle": upper}], anchor, floor_idx=1),
    )
    blob = load_obstacle_maps(path)
    floors = floor_summaries(blob)

    assert [f["index"] for f in floors] == [0, 1]
    assert floors[0]["has_up_stair"] and not floors[0]["has_down_stair"]
    assert floors[1]["has_down_stair"] and not floors[1]["has_up_stair"]
    assert floors[1]["explored_cells"] > floors[0]["explored_cells"]
    assert blob["floor_index"] == 1


def test_overwrite_respects_what_the_live_map_already_knows(tmp_path):
    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)
    om = _obstacle_map()
    om.explored_area[100:150, 100:150] = True

    path = save_obstacle_maps(tmp_path / "s.json", _Agent([{"obstacle": om}], anchor))
    blob = load_obstacle_maps(path)

    costmap = Costmap2D(resolution=1.0 / PPM, size_m=20.0)
    costmap.grid[:] = 100          # a live map that believes everything is wall
    assert apply_to_costmap(blob, 0, costmap, overwrite=False) == 0
    assert (costmap.grid == 100).all()

    costmap2 = Costmap2D(resolution=1.0 / PPM, size_m=20.0)
    costmap2.grid[:] = 100
    assert apply_to_costmap(blob, 0, costmap2, overwrite=True) > 0
    assert (costmap2.grid == 0).any()


def test_resolution_mismatch_is_refused(tmp_path):
    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)
    om = _obstacle_map()
    om.explored_area[100:110, 100:110] = True
    path = save_obstacle_maps(tmp_path / "s.json", _Agent([{"obstacle": om}], anchor))
    blob = load_obstacle_maps(path)

    with pytest.raises(Exception, match="resample"):
        apply_to_costmap(blob, 0, Costmap2D(resolution=0.10, size_m=20.0))


def test_the_vectorised_anchor_matches_to_episodic(tmp_path):
    """`apply_to_costmap` rotates a whole window with one matrix multiply.

    It must agree with `EpisodeAnchor.to_episodic`, which is what the agent
    itself uses per point -- if these ever drift, the stored map and the map
    the agent built stop being the same map.
    """
    anchor = EpisodeAnchor(np.array([1.5, -2.5]), 0.83)
    pts = np.array([[0.0, 0.0], [3.0, 1.0], [-2.0, 4.5], [1.5, -2.5]])

    one_by_one = np.array([anchor.to_episodic(p) for p in pts])
    vectorised = (pts - anchor.xy) @ anchor._r_world_to_ep.T

    assert np.allclose(one_by_one, vectorised, atol=1e-12)


def test_stairs_come_across_as_traversable(tmp_path):
    """ASCENT keeps staircases OUT of `explored_area`, so the occupancy alone
    leaves a hole exactly where a cross-floor run has to pass. They must land
    in `Costmap2D.stair_mask`, which means the same thing: traversable."""
    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.41)
    om = _obstacle_map()
    om.explored_area[150:250, 150:250] = True
    om._up_stair_map[200:210, 260:275] = True     # beyond the explored block
    om._down_stair_map[120:130, 140:150] = True

    path = save_obstacle_maps(tmp_path / "s.json", _Agent([{"obstacle": om}], anchor))
    blob = load_obstacle_maps(path)
    costmap = Costmap2D(resolution=1.0 / PPM, size_m=20.0)
    apply_to_costmap(blob, 0, costmap)

    assert costmap.stair_mask is not None, "stair evidence was dropped"
    assert costmap.stair_mask.shape == costmap.grid.shape
    n = int(costmap.stair_mask.sum())
    # 10x15 + 10x10 = 250 source cells; a rotation resamples, so allow slack.
    assert 150 <= n <= 400, f"expected ~250 stair cells, got {n}"
    # And they must be traversable, not unknown holes.
    assert (costmap.grid[costmap.stair_mask] == 0).all()


def test_a_floor_without_stairs_leaves_the_mask_alone(tmp_path):
    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)
    om = _obstacle_map()
    om.explored_area[150:250, 150:250] = True

    path = save_obstacle_maps(tmp_path / "s.json", _Agent([{"obstacle": om}], anchor))
    blob = load_obstacle_maps(path)
    costmap = Costmap2D(resolution=1.0 / PPM, size_m=20.0)
    apply_to_costmap(blob, 0, costmap)

    assert costmap.stair_mask is None


def test_a_pasted_staircase_becomes_a_flight_osg_can_climb(tmp_path):
    """The seam that carries ASCENT's multi-floor navigation into OSG's.

    `mapping/stairs.find_flights` feeds `FloorPolicy.pursuit_flight`, which is
    the only input to `NavAgent._flight_carrot` -- the climber that walks
    pointnav up the treads one waypoint at a time. It reads `costmap.height`
    and returns [] the moment it is None, so pasting occupancy and a stair mask
    alone leaves the treads NaN and the carrot silently never fires (measured:
    zero `climb_flight_carrot` counters across a whole cross-anchor run).
    """
    from osg.mapping.stairs import find_flights

    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)
    om = _obstacle_map()
    om.explored_area[150:260, 150:260] = True
    # A long, thin run of treads, as a staircase is.
    om._up_stair_map[190:230, 198:206] = True

    path = save_obstacle_maps(tmp_path / "s.json", _Agent([{"obstacle": om}], anchor))
    blob = load_obstacle_maps(path)

    costmap = Costmap2D(resolution=1.0 / PPM, size_m=20.0, track_height=True)
    apply_to_costmap(blob, 0, costmap, floor_y=0.0, next_floor_y=3.0)

    assert costmap.stair_mask is not None and costmap.stair_mask.any()
    ramped = np.isfinite(costmap.height)
    assert ramped.any(), "treads left at NaN: find_flights would see nothing"
    h = costmap.height[ramped]
    # Strictly between the two storeys, and spanning enough to be a flight.
    assert h.min() > 0.0 and h.max() < 3.0
    assert (h.max() - h.min()) > 1.0

    flights = find_flights(costmap, floor_y=0.0, new_level_m=3.4,
                           min_span_m=1.0, min_cells=50)
    assert flights, "no flight found: pursuit_flight stays None and the carrot cannot fire"
    flight = max(flights, key=lambda f: f.n_cells)
    assert flight.kind == "up"
    assert flight.span_m > 1.0
    # foot below top, which is what the carrot bands against.
    assert flight.heights.min() < flight.heights.max()


def test_no_storey_heights_means_no_ramp_and_no_crash(tmp_path):
    """Callers that do not know the storey heights still get occupancy."""
    anchor = EpisodeAnchor(np.array([0.0, 0.0]), 0.0)
    om = _obstacle_map()
    om.explored_area[150:260, 150:260] = True
    om._up_stair_map[190:230, 198:206] = True

    path = save_obstacle_maps(tmp_path / "s.json", _Agent([{"obstacle": om}], anchor))
    blob = load_obstacle_maps(path)
    costmap = Costmap2D(resolution=1.0 / PPM, size_m=20.0, track_height=True)

    assert apply_to_costmap(blob, 0, costmap) > 0
    assert not np.isfinite(costmap.height).any(), "heights invented without storeys"
