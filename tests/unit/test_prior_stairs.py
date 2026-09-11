"""The staircase the prior map already walked, read back.

`save_map` writes every committed floor transition into the prior map's
`connectivity`, and `apply_map` restores them into `FloorStack.stair_edges`.
Until `floor.use_prior_stairs`, nothing consumed them: an agent that knew
exactly where the stairs were still had to rediscover them by chance.

Measured on outputs/osg_authored_15, that chance is poor. Of 11 cross-floor
episodes, 4 never saw a portal and never attempted a switch, and across the run
405 storey requests produced 12 attempts.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from osg.agent.floor_policy import FloorPolicy
from osg.mapping.floor_stack import StairEdge

from .test_nav_agent import make_cfg

GROUND, UPPER = 0, 1


def _policy(use_prior_stairs: bool, edges):
    cfg = make_cfg()
    cfg.floor.enabled = True
    cfg.floor.cross_floor = True
    cfg.floor.stairs = True
    cfg.floor.use_prior_stairs = use_prior_stairs
    policy = FloorPolicy(cfg, stats={})
    policy.stack.current_id = GROUND
    policy.stack.stair_edges = list(edges)
    policy.estimator._levels = {GROUND: 0.0, UPPER: 2.9}
    return policy


def _remembered(policy, target_floor=None):
    return policy._remembered_stair(target_floor)


def test_the_mouth_on_this_floor_is_the_edge_we_walked_up_from():
    """`entry_xy` is where the agent stood when it committed to the new storey:
    the mouth of the staircase on the floor it LEFT."""
    policy = _policy(True, [StairEdge(GROUND, UPPER, entry_xy=np.array([1.9, 0.5]),
                                      exit_xy=np.array([2.1, 0.9]), step=132)])
    goal, other = _remembered(policy)
    assert other == UPPER
    assert goal == pytest.approx([1.9, 0.5])


def test_an_edge_walked_the_other_way_still_names_a_point_here():
    """Coming DOWN from the upper floor lands at `exit_xy` on this one, which is
    the same staircase seen from the other end."""
    policy = _policy(True, [StairEdge(UPPER, GROUND, entry_xy=np.array([2.1, 0.9]),
                                      exit_xy=np.array([1.9, 0.5]), step=200)])
    goal, other = _remembered(policy)
    assert other == UPPER
    assert goal == pytest.approx([1.9, 0.5])


def test_a_directed_request_ignores_a_staircase_to_the_wrong_storey():
    policy = _policy(True, [StairEdge(GROUND, 2, entry_xy=np.array([5.0, 5.0]), step=10)])
    policy.estimator._levels[2] = 5.8
    assert _remembered(policy, target_floor=UPPER) is None
    assert _remembered(policy, target_floor=2) is not None


def test_the_most_walked_staircase_wins():
    """A staircase pass 1 used repeatedly is the one that works."""
    policy = _policy(True, [
        StairEdge(GROUND, UPPER, entry_xy=np.array([9.0, 9.0]), step=400, n_traversals=1),
        StairEdge(GROUND, UPPER, entry_xy=np.array([1.9, 0.5]), step=132, n_traversals=3),
    ])
    goal, _ = _remembered(policy)
    assert goal == pytest.approx([1.9, 0.5])


def test_edges_that_do_not_touch_this_floor_are_ignored():
    policy = _policy(True, [StairEdge(1, 2, entry_xy=np.array([4.0, 4.0]), step=10)])
    assert _remembered(policy) is None


def test_an_edge_with_no_recorded_point_is_not_a_goal():
    policy = _policy(True, [StairEdge(GROUND, UPPER, entry_xy=None, exit_xy=None)])
    assert _remembered(policy) is None


def test_a_prior_map_with_no_transitions_offers_nothing():
    """Pass 1 never changed floor, so there is no staircase to remember and the
    agent is back to looking for one."""
    assert _remembered(_policy(True, [])) is None
