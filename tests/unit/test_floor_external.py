"""A storey declared from outside the agent, instead of inferred from height.

On the robot the pipeline's primary floor signal does not exist: Nav2's `map`
is 2D, so `camera_position[1]` is the same on both storeys. The operator says
which floor it is on instead (`/osg/floor`), and that assertion has to move
everything the estimator would have moved -- the active grid, the scene graph's
storey, the object layer's filing -- or half the pipeline keeps mapping the
floor the robot left.

`docs/MULTI_FLOOR.md` makes the single-floor equivalence gate a hard one, so
the first test here is that the default source is untouched by all of this.
"""
from __future__ import annotations

import numpy as np
import pytest

from osg.core.config import OSGConfig


def _policy(**floor_overrides):
    from osg.agent.floor_policy import FloorPolicy

    cfg = OSGConfig()
    for key, value in floor_overrides.items():
        setattr(cfg.floor, key, value)
    return FloorPolicy(cfg, {})


def _frame_at(y: float, xy=(0.0, 0.0)):
    from osg.core.types import CameraIntrinsics

    from tests.unit.conftest import make_frame

    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = float(xy[0]), float(y), float(xy[1])
    k = CameraIntrinsics(fx=320.0, fy=320.0, cx=320.0, cy=240.0, width=640, height=480)
    return make_frame(k, T)


def _live(**over):
    """The robot's floor configuration: declared storey, per-storey grids."""
    return _policy(source="external", enabled=True, estimate_only=False,
                   per_floor_costmap=True, **over)


# ------------------------------------------------------- the schedule parser


def test_the_schedule_is_step_to_key():
    from osg.agent.floor_policy import _parse_schedule

    assert _parse_schedule(["120:1", "300:0"]) == {120: 1, 300: 0}
    assert _parse_schedule(None) == {}
    with pytest.raises(ValueError, match="not 'step:key'"):
        _parse_schedule(["120"])


# --------------------------------------------------------- the default is safe


def test_the_estimator_remains_the_default_source():
    cfg = OSGConfig()
    assert cfg.floor.source == "estimator"
    assert cfg.floor.external_schedule == []
    assert _policy().external is False


def test_an_external_config_changes_nothing_until_a_switch_arrives():
    """Declaring the source does not by itself move the agent off floor 0:
    every pipeline that never calls `request_floor` behaves as before."""
    policy = _live()
    first = policy.observe(_frame_at(1.5), step=1)
    for step in range(2, 40):
        # A height trace that WOULD commit a second level under the estimator.
        policy.observe(_frame_at(4.6), step=step)
    assert policy.current_id == 0
    assert list(policy.levels) == [0]
    assert policy.observe(_frame_at(4.6), step=40) == first


def test_the_estimator_path_is_byte_identical_with_the_switch_compiled_in():
    """The refactor that added the external branch must not perturb the source
    the benchmark runs on."""
    estimator = _policy(enabled=True, estimate_only=False, per_floor_costmap=True)
    heights = [1.5] * 3 + [2.6, 3.4] + [4.6] * 30
    out = [estimator.observe(_frame_at(y, (0.3 * i, 0.0)), step=i)
           for i, y in enumerate(heights, start=1)]
    assert len(estimator.levels) == 2, "the estimator still commits a storey"
    assert out[0] == pytest.approx(1.5 - 0.88)
    assert estimator.floor_log[0][0] == 1 and len(estimator.floor_log) >= 2


# --------------------------------------------------------------- the switch


def test_a_requested_floor_is_applied_on_the_next_step_not_the_current_one():
    """The switch arrives mid-step from the env. Applying it where it lands
    would change `self.costmap` underneath a control loop that already read
    it."""
    policy = _live()
    policy.observe(_frame_at(1.5), step=1)
    grid_before = policy.costmap
    policy.request_floor(1)
    assert policy.current_id == 0 and policy.costmap is grid_before
    policy.observe(_frame_at(4.5), step=2)
    assert policy.current_id == 1


def test_a_switch_writes_the_whole_floor_state_not_just_the_stack():
    """`apply_map` had to solve this exact problem and the answer is that the
    estimator and the stack are one state; writing the stack alone leaves the
    estimator on floor 0 and detaches the key from the storey it names."""
    policy = _live()
    policy.observe(_frame_at(1.5), step=1)
    policy.request_floor(1)
    floor_y = policy.observe(_frame_at(4.5), step=2)

    assert policy.stack.current_id == 1
    assert policy.estimator.current == 1
    assert 1 in policy.estimator.levels
    assert policy.estimator.levels[1] == pytest.approx(4.5 - 0.88)
    assert policy._floor_y == pytest.approx(4.5 - 0.88)
    assert floor_y == pytest.approx(4.5 - 0.88), "the costmap bands at the new storey"
    assert policy.height_of(1) == pytest.approx(4.5 - 0.88)
    assert policy.stack.by_key(1).floor_y == pytest.approx(4.5 - 0.88)


def test_a_switch_swaps_the_grid_and_keeps_the_one_it_left():
    """The whole point: floor 1 gets its own map, and floor 0's survives to be
    returned to."""
    policy = _live()
    policy.observe(_frame_at(1.5), step=1)
    ground = policy.costmap
    ground.grid[10, 10] = 100

    policy.request_floor(1)
    policy.observe(_frame_at(4.5), step=2)
    assert policy.costmap is not ground
    assert policy.costmap.grid[10, 10] == -1, "a fresh storey starts unknown"

    policy.request_floor(0)
    policy.observe(_frame_at(1.5), step=3)
    assert policy.costmap is ground
    assert ground.grid[10, 10] == 100, "the floor below was kept, not rebuilt"


def test_switching_back_reuses_the_key_rather_than_inventing_one():
    policy = _live()
    policy.observe(_frame_at(1.5), step=1)
    for step, key in ((2, 1), (3, 0), (4, 1)):
        policy.request_floor(key)
        policy.observe(_frame_at(1.5 + 3.0 * key), step=step)
    assert sorted(policy.levels) == [0, 1]
    assert policy.stack.n_floors() == 2
    assert policy.stats["floor_switch_external"] == 3


def test_a_switch_to_the_current_floor_is_a_no_op():
    policy = _live()
    policy.observe(_frame_at(1.5), step=1)
    policy.request_floor(0)
    policy.observe(_frame_at(1.5), step=2)
    assert policy.stats.get("floor_switch_external", 0) == 0
    assert len(policy.floor_log) == 1


def test_a_switch_is_recorded_where_the_episode_record_reads_it():
    """`eval/record.py` writes `floor_log` and the switch counter; an external
    switch must be visible there exactly like a committed one."""
    policy = _live()
    policy.observe(_frame_at(1.5), step=1)
    policy.request_floor(1)
    policy.observe(_frame_at(4.5), step=7)
    assert policy.floor_log[-1][0] == 7
    assert policy.floor_log[-1][1] == 1
    assert policy.stats["floor_switch_external"] == 1
    assert policy.stack.stats()["floor_switches"] == 1


def test_a_switch_ends_a_portal_pursuit_with_its_own_reason():
    policy = _live(cross_floor=True)
    policy.observe(_frame_at(1.5), step=1)
    policy.pursuing = True
    policy.request_floor(1)
    policy.observe(_frame_at(4.5), step=2)
    assert policy.pursuing is False
    assert policy.stats.get("portal_end_external_switch") == 1
    assert "portal_end_arrived" not in policy.stats


def test_the_schedule_drives_the_switch_with_no_caller_at_all():
    """This is what makes the external path testable in habitat: the same code
    the robot's topic reaches, fired by step number."""
    policy = _live(external_schedule=["3:1", "5:0"])
    for step in range(1, 7):
        policy.observe(_frame_at(1.5), step=step)
        expect = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0, 6: 0}[step]
        assert policy.current_id == expect, f"step {step}"


# ---------------------------------------------- what the rest of the pipeline sees


def test_the_agent_follows_the_switch_into_its_map_and_its_scene_graph():
    """End to end through NavAgent: after a switch the costmap the FSM plans
    on, the storey objects are filed under, and the storey the scene graph
    rebuilds are all the new one."""
    from tests.unit.test_nav_agent import _frame, make_agent, make_cfg

    cfg = make_cfg()
    cfg.floor.source = "external"
    cfg.floor.enabled = True
    cfg.floor.estimate_only = False
    cfg.floor.per_floor_costmap = True
    agent = make_agent(cfg)
    agent.act(_frame([0.0, 0.0]))
    ground = agent.costmap

    rebuilt = []
    agent.scene_graph.rebuild_floor = lambda *a, **kw: rebuilt.append(kw.get("floor_key"))
    filed = []
    real_update = agent.object_layer.update
    agent.object_layer.update = lambda *a, **kw: (
        filed.append(kw.get("floor_key")), real_update(*a, **kw))[1]

    agent.floors.request_floor(1)
    agent.act(_frame([0.2, 0.0], frame_id=1))

    assert agent.floors.current_id == 1
    assert agent.costmap is not ground
    assert agent.costmap is agent._floor_stack.layer(1).costmap
    assert agent._floor_stack.by_key(0).costmap is ground
    # Room segmentation is rate-limited (scene_graph.room_seg_every_kf), so the
    # graph follows within a cycle rather than on the same step.
    for i in range(12):
        agent._on_keyframe(_frame([0.2, 0.0], frame_id=2 + i))
    assert filed and filed[-1] == 1, "new observations are filed on the new storey"
    assert rebuilt and rebuilt[-1] == 1, "the scene graph rebuilds the new storey"


def test_two_storeys_stay_separable_by_height_in_the_scene_graph():
    """`graph/scene_graph.py` files an object by `floor_of_height`, so two
    storeys at the SAME height would collapse into one -- which is exactly what
    a 2D `map` frame reports. The virtual storey offset is what keeps them
    apart, and this is the property it has to deliver."""
    policy = _live()
    policy.observe(_frame_at(1.5), step=1)
    policy.request_floor(1)
    policy.observe(_frame_at(1.5 + 3.0), step=2)  # lifted by virtual_storey_m
    assert policy.estimator.floor_of_height(1.5 - 0.88) == 0
    assert policy.estimator.floor_of_height(4.5 - 0.88) == 1


def test_a_restored_snapshot_starts_on_the_declared_storey():
    """Run 2 on the robot: the operator says which floor the robot is standing
    on, and that must beat a height comparison against a synthetic vertical."""
    from osg.graph.map_store import apply_map

    from tests.unit.test_nav_agent import _frame, make_agent, make_cfg

    def _agent():
        cfg = make_cfg()
        cfg.floor.source = "external"
        cfg.floor.enabled = True
        cfg.floor.estimate_only = False
        cfg.floor.per_floor_costmap = True
        return make_agent(cfg)

    mapper = _agent()
    mapper.act(_frame([0.0, 0.0]))
    mapper.floors.request_floor(1)
    mapper.act(_frame([0.2, 0.0], frame_id=1))
    blob = _snapshot(mapper)

    searcher = _agent()
    apply_map(searcher, blob, initial_floor_y=0.0, initial_floor_key=1)
    assert searcher.floors.current_id == 1
    assert searcher.floors.estimator.current == 1

    # Without the override the nearest height wins, which on a robot pose is
    # a comparison between two synthetic numbers.
    other = _agent()
    apply_map(other, blob, initial_floor_y=0.0)
    assert other.floors.current_id == 0


def test_an_unknown_declared_key_falls_back_instead_of_raising():
    """Starting run 2 on a storey run 1 never mapped is legitimate: it should
    begin with an empty map, not an exception."""
    from osg.graph.map_store import apply_map

    from tests.unit.test_nav_agent import _frame, make_agent, make_cfg

    mapper = make_agent(make_cfg())
    mapper.act(_frame([0.0, 0.0]))
    blob = _snapshot(mapper)
    agent = make_agent(make_cfg())
    apply_map(agent, blob, initial_floor_y=0.0, initial_floor_key=7)
    assert agent.floors.current_id == 0


def _snapshot(agent) -> dict:
    """Write and read back a map, the way the two robot runs do."""
    import tempfile
    from pathlib import Path

    from osg.graph.map_store import load_map, save_map

    with tempfile.TemporaryDirectory() as tmp:
        path = save_map(Path(tmp) / "lab.json", agent, scene="lab")
        return load_map(path)
