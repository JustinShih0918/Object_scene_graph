"""CPU-only identity and selection rules for matched diagnostic comparisons."""
from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Iterable

import numpy as np


def episode_key(record: dict) -> tuple[str, str]:
    meta = record["authored_layout"]
    return str(meta["scene"]), str(record["episode_id"])


def select_diagnostic_episodes(records: Iterable[dict]) -> list[dict]:
    """One episode per scene in each layout/outcome stratum, diverse targets.

    Diversity is maximized within each stratum before consulting episode IDs.
    Selection uses historical OSG outcomes only, never competitor results.
    """
    records = list(records)
    keys = [episode_key(e) for e in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate scene/episode identity")
    scenes = sorted({k[0] for k in keys})
    if len(scenes) != 3:
        raise ValueError("The diagnostic design requires exactly three scenes")
    selected = []
    used_targets: Counter = Counter()
    for layout in ("in_anchor", "cross_anchor"):
        for success in (False, True):
            groups = [[e for e in records if episode_key(e)[0] == scene
                       and e["authored_layout"]["layout_type"] == layout
                       and bool(e["success"]) == success] for scene in scenes]
            if any(not group for group in groups):
                raise ValueError(f"Missing scene in stratum {layout}/{success}")
            def rank(combo):
                targets = [e["target"] for e in combo]
                return (-len(set(targets)), sum(used_targets[t] for t in targets),
                        tuple(episode_key(e) for e in combo))
            chosen = min(product(*groups), key=rank)
            selected.extend(chosen)
            used_targets.update(e["target"] for e in chosen)
    return selected


def validate_replay_identity(expected: dict, actual: dict) -> None:
    """Fail before motion if an episode was regenerated with different inputs."""
    for key in ("scene", "layout_id", "sha256", "target_handle", "target_semantic_id"):
        if expected[key] != actual[key]:
            raise ValueError(f"Episode mismatch: {key}")
    for key in ("position", "rotation"):
        if not np.allclose(expected["start"][key], actual["start"][key], atol=1e-6, rtol=0):
            raise ValueError(f"Start mismatch: {key}")
    if expected["viewpoints"] != actual["viewpoints"]:
        raise ValueError("Goal viewpoint mismatch")


# Habitat uses Y-up; DualMap navigation uses Z-up. Both camera poses here are
# optical (+Z forward, +Y down); the world rotation is separate from camera axes.
HABITAT_TO_DUALMAP = np.array([[1., 0., 0., 0.], [0., 0., -1., 0.],
                              [0., 1., 0., 0.], [0., 0., 0., 1.]])


def paired_outcome(osg_success: bool, dualmap_success: bool) -> str:
    return {(True, True): "both_succeed", (True, False): "osg_only",
            (False, True): "dualmap_only", (False, False): "both_fail"}[
                bool(osg_success), bool(dualmap_success)]
