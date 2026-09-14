"""`ycb` group: the authored dynamic-scene benchmark.

Runtime discovery of the collector's layouts, deterministic episode generation,
and the two knobs the dynamic protocol turns on: `relocate_at_step` for a move
the agent can witness, and `map_out`/`map_in` for the two-pass protocol where
pass 2 navigates from a map that pass 1 built and the world has since
invalidated. The staleness IS the experiment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ..paths import collector_data_root, hm3d_scene_root, ycb_authoring_root


# Handle -> the class name the detector is asked for. Not always the object's
# common name: YOLOE's text head scores "pitcher" at 0.00 on this asset at every
# resolution and "blue plastic pitcher" at 0.71, so the descriptive phrase is the
# label (scripts/probe_ycb_detection.py, mode=labels).
YCB_TARGET_LABELS: Dict[str, str] = {
    "002_master_chef_can": "coffee can",
    "003_cracker_box": "cracker box",
    "005_tomato_soup_can": "tin can",
    # "mustard bottle" scored 0.31-0.39 on the asset at 0.6-2 m and 0/5 in situ
    # on the 15-scene runs; "yellow bottle" scores 0.50-0.93 on the same frames
    # (scripts/render_asset_views.py, render.names probe, 2026-09-10).
    "006_mustard_bottle": "yellow bottle",
    "011_banana": "banana",
    "019_pitcher_base": "blue plastic pitcher",
    "021_bleach_cleanser": "bleach bottle",
    "024_bowl": "bowl",
    "025_mug": "mug",
    "029_plate": "red plate",
    "037_scissors": "scissors",
    "053_mini_soccer_ball": "soccer ball",
    "077_rubiks_cube": "rubiks cube",
    # The 15-scene authoring root places this in 14 of its 15 scenes; the name
    # is the detector query, and this is the plain one.
    "072-a_toy_airplane": "toy airplane",
}



@dataclass
class YCBAuthoredConfig:
    """Runtime discovery and deterministic episode generation for authored YCB layouts."""

    data_root: str = field(default_factory=lambda: str(collector_data_root()))
    layout_root: str = field(default_factory=lambda: str(ycb_authoring_root()))
    # Full canonical HM3D v0.2 scene tree.  Authored JSON may still contain
    # the collector's historical ``scene_datasets/hm3d`` path; the loader
    # rebases that spelling here rather than following the mutable symlink.
    hm3d_root: str = field(default_factory=lambda: str(hm3d_scene_root()))
    scenes: List[str] = field(default_factory=lambda: ["*"])
    layout_types: List[str] = field(default_factory=lambda: ["static"])
    layout_indices: List[int] = field(default_factory=lambda: [1, 2, 3])
    # Restrict episodes to these targets (YCB handle or label; empty = all).
    # Several assets in the collector's dataset render in a way the open-vocab
    # detector cannot recognise at any authored viewpoint, so their episodes
    # measure asset coverage rather than dynamic-scene handling. The layouts
    # themselves are DualMap's original data and are never edited.
    targets: List[str] = field(default_factory=list)
    # Combined dynamic/multi-floor benchmark selectors.  Disabled by default so
    # every existing authored manifest and experiment keeps its source behavior.
    cross_floor_relocations_only: bool = False
    relocation_directions: List[str] = field(
        default_factory=lambda: ["upward", "downward"]
    )
    start_on_prior_floor: bool = False
    relocation_floor_tolerance_m: float = 0.5
    # Explicit scene lists normally fail on any absent requested slot. Combined
    # campaigns span heterogeneous authoring coverage and opt into recording
    # those gaps while keeping every available layout.
    skip_incomplete_layouts: bool = False
    starts_per_target: int = 1
    seed: int = 42
    manifest_cache_dir: str = "outputs/ycb_manifests"
    # Mid-episode relocation (docs/DYNAMIC_SCENES.md, Phase 2). -1 disables it
    # and every episode behaves exactly as before. When enabled, an episode
    # whose layout is a dynamic one starts the world in the paired STATIC
    # layout and moves the objects to the episode's own poses at this step --
    # so the goals are where the object ends up, and the change is something
    # the agent can witness rather than wake up to.
    relocate_at_step: int = -1
    # `in_view` waits until the target is actually visible from the current
    # pose, `out_of_view` waits until it is not, `any` fires immediately. The
    # two conditions measure different things: in_view is the clean test of
    # negative evidence, out_of_view tests whether the search recovers.
    relocate_when: str = "any"
    # If the visibility condition never comes true, relocate anyway this many
    # steps later, rather than silently turning the episode into a static one.
    relocate_deadline_steps: int = 120
    # Two-pass benchmark (docs/DYNAMIC_SCENES.md, Phase 2). Pass 1 explores the
    # STATIC layout and writes one snapshot per scene to `map_out`; pass 2 runs
    # the moved layout and starts from `map_in`, so the map the agent navigates
    # with is genuinely stale. The staleness IS the experiment -- an agent that
    # rebuilds from scratch is never wrong about anything and measures nothing.
    map_out: str = ""
    map_in: str = ""
    # The same two passes, for the OTHER map. `navigation/` builds ASCENT's
    # per-storey `ObstacleMap` while it explores; these store that stack and
    # give it back, so pass 2's presence-filter pipeline plans over occupancy
    # ASCENT's navigation built rather than rebuilding its own from scratch.
    # Kept separate from `map_out`/`map_in` because the two artifacts are
    # written in different frames and either can be used without the other
    # (src/navigation/mapping/map_store.py).
    obstacle_map_out: str = ""
    obstacle_map_in: str = ""
    # Whether `map_in` also restores the OCCUPANCY it stored, or only the
    # scene graph (tracks, presence beliefs, storey heights and connectivity).
    # False is the "ASCENT's map is the sole occupancy" arm: the snapshot keeps
    # what the objects hang off, and every cell the planner reads comes from
    # `obstacle_map_in` instead of from the pass that built the scene graph.
    map_in_occupancy: bool = True
    # Whether a loaded obstacle map may overwrite cells the live map has
    # already witnessed for itself. False keeps this session's own evidence
    # and fills in only what it has not seen.
    obstacle_map_overwrite: bool = False
    # Score a STOP by horizontal distance to the OBJECT, the released DualMap
    # benchmark's rule, instead of habitat's geodesic distance to an authored
    # viewpoint (whose rings at 0.8-2.0 m make the effective tolerance about
    # 2 m; DUALMAP_OFFICIAL_RERUN.md). Attempts are spent by the same rule.
    # Habitat's own numbers are kept under habitat_success / habitat_spl.
    score_by_object_distance: bool = False
    object_success_distance_m: float = 1.0
    target_labels: Dict[str, str] = field(
        default_factory=lambda: dict(YCB_TARGET_LABELS)
    )
    viewpoint_radii_m: List[float] = field(
        default_factory=lambda: [0.8, 1.2, 1.5, 2.0]
    )
    viewpoint_angular_samples: int = 24
    viewpoint_max_snap_m: float = 0.5
    viewpoint_dedup_m: float = 0.2
    viewpoint_min_visible_pixels: int = 20
    start_min_geodesic_m: float = 3.0
    start_sample_attempts: int = 2000
