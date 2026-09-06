"""The observation-novelty weight, ported from WACV'27 #1308 (TextNav) Eq. 3.

w_r = exp(-(n_r - free)/sigma) over rooms, with n_r the mean count of
observations THIS EPISODE added to the object nodes in room r. It exists for one
measured pathology: 00848's kitchen holds 68 containers, the surface queue never
empties, and `search_room_saturation` cannot break it because that counter
increments only on ARRIVALS -- about a dozen an episode -- while observation
counts run away every keyframe.

The delta, not the absolute count, is the whole difficulty of the port: the
paper builds its graph from scratch each episode and we load a prior map whose
per-region mean n_obs is already 5.0 in 00848 and 13.3 in 00880.

Habitat-free: the weight is arithmetic over the scene graph.
"""
import numpy as np

from osg.core.config import OSGConfig
from osg.exploration.strategy import ExplorationStrategy, WorldView


class _Obj:
    def __init__(self, track_id, room_id, n_obs):
        self.track_id, self.room_id, self.n_obs = track_id, room_id, n_obs


class _Graph:
    def __init__(self, objects):
        self.objects = objects


def _strategy(**overrides):
    cfg = OSGConfig()
    for k, v in overrides.items():
        setattr(cfg.exploration, k, v)
    return ExplorationStrategy(cfg, None, None, None, None, {}, None)


def _world(objects):
    return WorldView(
        frame=None, step=0, agent_xy=np.zeros(2), costmap=None,
        scene_graph=_Graph(objects), object_layer=None, keyframes=None,
        target="bowl", goal_xy=None,
    )


def test_off_by_default_so_every_earlier_condition_reproduces():
    s = _strategy()
    assert s.cfg.search_obs_novelty_sigma == 0.0
    assert s._room_novelty(_world([_Obj(1, 1, 50)])) == {}


def test_a_loaded_prior_map_costs_a_room_nothing():
    """The failure this port would otherwise walk into. A track that arrives
    with 40 observations from the map-building pass -- which gets ~6x a scored
    episode's budget -- has told this episode's search nothing."""
    s = _strategy(search_obs_novelty_sigma=8.0, search_obs_novelty_floor=0.0)
    w = s._room_novelty(_world([_Obj(1, 1, 40), _Obj(2, 2, 3)]))
    assert w == {1: 1.0, 2: 1.0}


def test_a_room_the_episode_has_chewed_on_falls_behind_a_fresh_one():
    s = _strategy(search_obs_novelty_sigma=8.0, search_obs_novelty_floor=0.0)
    objs = [_Obj(1, 1, 40), _Obj(2, 2, 3)]
    s._room_novelty(_world(objs))          # baseline
    objs[0].n_obs += 16                    # sixteen more looks at room 1
    w = s._room_novelty(_world(objs))
    assert w[2] == 1.0
    assert w[1] == np.exp(-16.0 / 8.0)
    assert w[1] < w[2]


def test_a_track_discovered_mid_episode_enters_at_zero():
    """Otherwise the first sighting of a new object would instantly stale the
    room it is in, which is backwards: a new object is news."""
    s = _strategy(search_obs_novelty_sigma=8.0, search_obs_novelty_floor=0.0)
    objs = [_Obj(1, 1, 5)]
    s._room_novelty(_world(objs))
    objs.append(_Obj(2, 1, 1))             # newly detected this episode
    assert s._room_novelty(_world(objs))[1] == 1.0


def test_the_weight_reads_the_mean_not_the_total():
    """A room is not stale because it is BIG. Summing would conflate room size
    with room staleness and empty every large room first."""
    s = _strategy(search_obs_novelty_sigma=8.0, search_obs_novelty_floor=0.0)
    objs = [_Obj(i, 1, 0) for i in range(20)] + [_Obj(99, 2, 0)]
    s._room_novelty(_world(objs))
    for o in objs:
        o.n_obs += 4
    w = s._room_novelty(_world(objs))
    assert w[1] == w[2]


def test_the_floor_keeps_a_disappointing_room_reachable():
    """If the target IS in the room the agent has been staring at, the search
    must still be able to come back to it."""
    s = _strategy(search_obs_novelty_sigma=1.0, search_obs_novelty_floor=0.25)
    objs = [_Obj(1, 1, 0)]
    s._room_novelty(_world(objs))
    objs[0].n_obs = 500
    assert s._room_novelty(_world(objs))[1] == 0.25


def test_free_observations_protect_the_window_where_successes_happen():
    """The lesson `search_room_saturation_free` records: decaying from the first
    look weakens the prior inside the window where successes actually occur."""
    s = _strategy(search_obs_novelty_sigma=8.0, search_obs_novelty_free=10.0,
                  search_obs_novelty_floor=0.0)
    objs = [_Obj(1, 1, 0), _Obj(2, 2, 0)]
    s._room_novelty(_world(objs))
    objs[0].n_obs, objs[1].n_obs = 6, 18
    w = s._room_novelty(_world(objs))
    assert w[1] == 1.0
    assert w[2] == np.exp(-8.0 / 8.0)


def test_unplaced_nodes_belong_to_no_room():
    """room_id 0 means unassigned. Pooling those into a room-0 bucket would
    invent a room and hand its weight to nothing."""
    s = _strategy(search_obs_novelty_sigma=8.0)
    assert 0 not in s._room_novelty(_world([_Obj(1, 0, 99), _Obj(2, 1, 1)]))


# ------------------------------------------- measuring the scale, and the null


def test_the_scale_is_recorded_even_when_the_weight_is_off():
    """A control run has to be what measures sigma's scale, or sigma is a
    guess -- and this codebase has already shipped two arms that tested nothing
    because their knob never moved."""
    s = _strategy()                        # sigma = 0, weight disabled
    objs = [_Obj(1, 1, 4)]
    s._room_novelty(_world(objs))
    objs[0].n_obs = 15
    s._room_novelty(_world(objs))
    assert s.survival_report()["novelty_n_max"] == 11.0
    assert s.survival_report()["novelty_applied"] == 0


def test_a_disabled_arm_reports_a_void_not_a_null():
    s = _strategy()
    r = s.survival_report()
    assert r["novelty_applied"] == 0 and r["novelty_reordered"] == 0
    assert r["novelty_min"] == 1.0 and r["novelty_n_max"] == 0.0
