"""Integration coverage for the mounted authored-YCB staging dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.sim

pytest.importorskip("habitat")

from osg.core.paths import collector_data_root, ycb_authoring_root

DATA_ROOT = collector_data_root()
LAYOUT_ROOT = ycb_authoring_root()


def _mounted() -> bool:
    return DATA_ROOT.is_dir() and LAYOUT_ROOT.is_dir()


def _config():
    from hydra import compose, initialize_config_dir

    from osg.core.config import register_configs

    register_configs()
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
        return compose(
            config_name="config",
            overrides=[
                "+experiment=ycb_authored_nav",
                # The mounted authoring set now contains 15 complete scenes;
                # keep this six-episode smoke test focused on its original
                # deterministic scene.
                "ycb.scenes=[00829-QaLdnwvtxbs]",
                "eval.save_viz=false",
                "eval.debug_frames=false",
            ],
        )


def test_current_wildcard_discovery_selects_complete_scene():
    if not _mounted():
        pytest.skip("collector data is not mounted")
    from osg.core.config import YCB_TARGET_LABELS
    from osg.sim.ycb_layouts import discover_authored_layouts

    found = discover_authored_layouts(
        data_root=DATA_ROOT,
        layout_root=LAYOUT_ROOT,
        scenes=["*"],
        layout_types=["static"],
        layout_indices=[1, 2, 3],
        target_labels=YCB_TARGET_LABELS,
        hm3d_root=DATA_ROOT / "versioned_data/hm3d-0.2/hm3d",
    )
    assert len(found.layouts) == 15
    assert {layout.scene_name for layout in found.layouts} == {
        "00808-y9hTuugGdiq",
        "00810-CrMo8WxCyVb",
        "00813-svBbv1Pavdk",
        "00820-mL8ThkuaVTM",
        "00821-eF36g7L6Z9M",
        "00823-7MXmsvcQjpJ",
        "00824-Dd4bFSTQ8gi",
        "00839-zt1RVoi7PcG",
        "00844-q5QZSEeHe5g",
        "00848-ziup5kvtCCR",
        "00853-5cdEh9F2hJL",
        "00871-VBzV5z6i1WS",
        "00876-mv2HUxq3B53",
        "00880-Nfvxx8J5NCo",
        "00891-cvZr5TUy5C5",
    }


@pytest.mark.timeout(600)
def test_six_deterministic_episodes_and_reset_injection():
    if not _mounted():
        pytest.skip("collector data is not mounted")
    from osg.sim.ycb_env import YCBAuthoredNavEnv, prepare_ycb_benchmark

    cfg = _config()
    prepared = prepare_ycb_benchmark(cfg, force=True)
    regenerated = prepare_ycb_benchmark(cfg, force=True)
    assert regenerated.manifests == prepared.manifests
    episodes = [item for manifest in prepared.manifests for item in manifest["episodes"]]
    assert len(episodes) == 6
    assert all(item["start"]["initial_geodesic_distance"] >= 3.0 for item in episodes)
    assert all(item["viewpoints"] for item in episodes)
    assert all(
        viewpoint["visible_pixels"] >= cfg.ycb.viewpoint_min_visible_pixels
        for item in episodes
        for viewpoint in item["viewpoints"]
    )

    env = YCBAuthoredNavEnv(cfg)
    env.reset()
    authored = env.episode_metadata()
    layout = env._layout_by_key[(authored["scene"], authored["layout_id"])]
    expected = {obj.semantic_id: np.asarray(obj.translation) for obj in layout.objects}
    object_manager = env.env.sim.get_rigid_object_manager()
    assert object_manager.get_num_objects() == 6
    objects = env._active_objects
    actual = {
        int(obj.semantic_id): np.asarray(obj.translation, dtype=float)
        for obj in objects
    }
    assert actual.keys() == expected.keys()
    for semantic_id, translation in expected.items():
        np.testing.assert_allclose(actual[semantic_id], translation, atol=1e-5)

    env.step("stop")
    metrics = env.metrics()
    assert {"success", "spl", "distance_to_goal"} <= metrics.keys()
    assert all(
        np.isfinite(float(metrics[name]))
        for name in ("success", "spl", "distance_to_goal")
    )
    env.close()
