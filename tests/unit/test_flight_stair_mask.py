"""Which flight the climber drives to, when several look like staircases.

`find_flights` asks the height layer for cells sitting between two storeys.
Furniture answers: a sofa back, a counter, a mezzanine lip. Selection was
`min(usable, key=distance to foot)`, so whatever is nearest wins -- and the
nearest intermediate-height blob is almost never the staircase.

Measured on 00800 cross_anchor_01 (outputs/mf5_pass2_v3), real staircase at
world (-8.1, -13.0):

    ep1  step 172  (-0.67, -4.73)   210 cells   ~10 m away, and pursued from
                                                the TOP storey as a flight_up
    ep2  step  68  (-3.32, -0.97)  1166 cells
    ep2  step 192  (-9.42, -3.32)   652 cells
    ep2  step 245  (-11.02, -3.82)  778 cells
    ep2  step 296  (-8.32, -12.67)  596 cells   <- the staircase, at last

Both episodes spent their climb on furniture and rose 0.00 m. The stair mask
carried over from the ASCENT pass is exactly the evidence that separates the
two, and nothing was reading it.
"""
from types import SimpleNamespace

import numpy as np

from osg.agent.floor_policy import FloorPolicy

from .test_climb import GROUND, UPPER, make_cfg


def _flight(kind, cells_rc, foot_xy):
    return SimpleNamespace(
        kind=kind, cells_rc=np.asarray(cells_rc), foot_xy=np.asarray(foot_xy, float),
        heights=np.linspace(0.3, 1.5, len(cells_rc)), n_cells=len(cells_rc),
    )


def _policy(*, levels=(0.0, 2.9), current=GROUND):
    cfg = make_cfg()
    cfg.floor.enabled = True
    cfg.floor.cross_floor = True
    cfg.floor.flights_prefer_stair_mask = True
    policy = FloorPolicy(cfg, stats={})
    keys = [GROUND, UPPER][: len(levels)]
    policy.estimator._levels = dict(zip(keys, levels))
    policy.estimator.current = current
    for key, height in zip(keys, levels):
        policy.stack.layer(key, step=0).floor_y = height
    policy.stack.current_id = current
    return policy


def test_the_stair_map_beats_the_nearer_sofa():
    policy = _policy()
    policy.costmap.stair_mask = np.zeros(policy.costmap.grid.shape, dtype=bool)
    policy.costmap.stair_mask[40:50, 40:50] = True
    sofa = _flight("up", [[5, 5], [5, 6], [6, 5]], foot_xy=[0.2, 0.2])
    stairs = _flight("up", [[42, 42], [43, 43], [44, 44]], foot_xy=[9.0, 9.0])
    picked = policy._best_flight([sofa, stairs], np.zeros(2))
    assert picked is stairs, "the near blob won again; the stair map was not read"
    assert policy.stats["flight_pick_corroborated"] == 1


def test_nearest_still_wins_among_corroborated_flights():
    policy = _policy()
    policy.costmap.stair_mask = np.zeros(policy.costmap.grid.shape, dtype=bool)
    policy.costmap.stair_mask[40:60, 40:60] = True
    near = _flight("up", [[42, 42]], foot_xy=[1.0, 0.0])
    far = _flight("up", [[55, 55]], foot_xy=[9.0, 0.0])
    assert policy._best_flight([near, far], np.zeros(2)) is near


def test_without_a_stair_map_nothing_is_corroborated_and_nearest_wins():
    """The mask is absent on a run with no prior map and before the detector
    has marked anything. That must degrade to the old behaviour, not to none."""
    policy = _policy()
    policy.costmap.stair_mask = None
    near = _flight("up", [[5, 5]], foot_xy=[1.0, 0.0])
    far = _flight("up", [[9, 9]], foot_xy=[9.0, 0.0])
    assert policy._best_flight([near, far], np.zeros(2)) is near
    assert policy.stats["flight_pick_uncorroborated"] == 1


def test_no_up_flight_from_the_topmost_storey():
    """ep1's error: on the top storey every flight_up is furniture, because
    there is no storey above for it to reach."""
    policy = _policy(current=UPPER)
    up = _flight("up", [[5, 5]], foot_xy=[1.0, 0.0])
    down = _flight("down", [[7, 7]], foot_xy=[8.0, 0.0])
    kept = policy._rank_flights([up, down], floor_y=2.9)
    assert kept == [down]
    assert policy.stats["flights_dropped_no_storey"] == 1


def test_one_known_storey_drops_nothing():
    """At episode start the estimator knows a single level, and every flight
    would look impossible -- which would remove the only way up."""
    policy = _policy(levels=(0.0,))
    up = _flight("up", [[5, 5]], foot_xy=[1.0, 0.0])
    assert policy._rank_flights([up], floor_y=0.0) == [up]
    assert "flights_dropped_no_storey" not in policy.stats


def test_the_filter_never_empties_the_pool():
    """If the direction test would drop everything, keep the flights: a wrong
    guess about storeys must not be able to strand the agent."""
    policy = _policy(current=UPPER)
    up = _flight("up", [[5, 5]], foot_xy=[1.0, 0.0])
    assert policy._rank_flights([up], floor_y=2.9) == [up]


# ------------------------------------------------------- climb direction

def test_an_up_flight_starts_an_up_climb_even_when_the_storeys_tie():
    """The measured bug (outputs/mf5_pass2_v5): flight kind `up`, `here_y` and
    `target_y` both 0.16, and `target_y > here_y` is False at equality -- so
    the tie fell through to -1 and the carrot hunted for treads BELOW an agent
    on the ground floor for 200 steps."""
    from osg.agent.nav_agent import State

    from .test_nav_agent import make_agent, make_cfg

    agent = make_agent(make_cfg(climb_enabled=True, climb_direction_from_flight=True))
    agent.floors.pursuit_flight = SimpleNamespace(kind="up")
    agent._goal_floor_y_cache = 0.163          # identical to the storey below
    agent.floors.estimator._levels = {0: 0.163}
    agent.floors.stack.current_id = 0
    agent._goal_xy = np.array([1.0, 1.0])
    frame = SimpleNamespace(camera_position=np.array([0.0, 1.5, 0.0]))
    agent._start_climb(frame)
    assert agent._climb_direction == 1, "an up flight must climb up"
    assert agent.stats["climb_start_up"] == 1
    assert agent.state is State.CLIMB


def test_a_down_flight_starts_a_down_climb():
    from .test_nav_agent import make_agent, make_cfg

    agent = make_agent(make_cfg(climb_enabled=True, climb_direction_from_flight=True))
    agent.floors.pursuit_flight = SimpleNamespace(kind="down")
    agent._goal_floor_y_cache = 3.163          # would say UP on heights alone
    agent.floors.estimator._levels = {0: 0.163}
    agent.floors.stack.current_id = 0
    agent._goal_xy = np.array([1.0, 1.0])
    agent._start_climb(SimpleNamespace(camera_position=np.array([0.0, 1.5, 0.0])))
    assert agent._climb_direction == -1
    assert agent.stats["climb_start_down"] == 1


def test_the_height_rule_is_untouched_when_the_flag_is_off():
    """Every arm measured before this existed must be unaffected."""
    from .test_nav_agent import make_agent, make_cfg

    agent = make_agent(make_cfg(climb_enabled=True))
    assert agent.cfg.agent.climb_direction_from_flight is False
    agent.floors.pursuit_flight = SimpleNamespace(kind="up")
    agent._goal_floor_y_cache = 0.163
    agent.floors.estimator._levels = {0: 0.163}
    agent.floors.stack.current_id = 0
    agent._goal_xy = np.array([1.0, 1.0])
    agent._start_climb(SimpleNamespace(camera_position=np.array([0.0, 1.5, 0.0])))
    assert agent._climb_direction == -1, "the shipped tie-goes-down behaviour"
