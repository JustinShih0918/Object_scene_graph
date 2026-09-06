import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from osg.eval.comparison import (
    HABITAT_TO_DUALMAP, paired_outcome, select_diagnostic_episodes,
    validate_replay_identity,
)


def test_selection_balances_layout_outcome_and_scene_without_competitor_results():
    records = [{"episode_id": f"{layout}_{success}_{target}", "target": target,
                "success": success, "authored_layout": {"scene": scene, "layout_type": layout}}
               for scene in ("a", "b", "c") for layout in ("in_anchor", "cross_anchor")
               for success in (0, 1) for target in ("bowl", "can", "plate")]
    result = select_diagnostic_episodes(records)
    assert len(result) == 12
    assert result == select_diagnostic_episodes(reversed(records))
    for offset in range(0, 12, 3):
        assert len({r["target"] for r in result[offset:offset + 3]}) == 3
        assert {r["authored_layout"]["scene"] for r in result[offset:offset + 3]} == {"a", "b", "c"}
    with pytest.raises(ValueError, match="Duplicate"):
        select_diagnostic_episodes(records + records[:1])


def test_identity_rejects_changed_start_and_target():
    expected = dict(scene="a", layout_id="in_anchor_01", sha256="abc", target_handle="bowl",
                    target_semantic_id=12, start=dict(position=[0, 0, 0], rotation=[0, 0, 0, 1]),
                    viewpoints=[{"position": [1, 0, 0]}])
    validate_replay_identity(expected, copy.deepcopy(expected))
    actual = copy.deepcopy(expected)
    actual["start"]["position"][0] = 0.01
    with pytest.raises(ValueError, match="Start mismatch"):
        validate_replay_identity(expected, actual)
    actual = copy.deepcopy(expected)
    actual["target_semantic_id"] = 13
    with pytest.raises(ValueError, match="target_semantic_id"):
        validate_replay_identity(expected, actual)


def test_world_transform_preserves_distance_and_maps_up():
    matrix = HABITAT_TO_DUALMAP
    np.testing.assert_allclose(matrix @ [0, 1, 0, 0], [0, 0, 1, 0])
    np.testing.assert_allclose(matrix @ [0, 0, -1, 0], [0, 1, 0, 0])
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(4))
    assert paired_outcome(False, True) == "dualmap_only"


def test_native_polyline_steering_uses_optical_forward_and_stops_at_endpoint():
    spec = importlib.util.spec_from_file_location("native_driver", Path(__file__).parents[2] / "scripts/dualmap_native.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # OpenGL forward is world -Z; optical pose flips Y and Z camera axes.
    pose = np.diag([1., -1., -1., 1.])
    frame = SimpleNamespace(T_wc=pose)
    assert module.PolylineFollower().action([[0, 0, 0], [0, 1, 0]], frame) == "move_forward"
    assert module.PolylineFollower().action([[0, 0, 0], [-1, 0, 0]], frame) == "turn_left"
    assert module.PolylineFollower().action([[0, 0, 0], [1, 0, 0]], frame) == "turn_right"
    assert module.PolylineFollower().action([[0, 0, 0]], frame) == "stop"
