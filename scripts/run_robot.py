"""Run the pipeline on a real robot, over ROS 2 / Nav2.

Two runs make an experiment, differing in one field:

    python scripts/run_robot.py +experiment=stretch3_map      # explore, keep the map
    python scripts/run_robot.py +experiment=stretch3_search   # load it, find the thing

That is the protocol the whole dynamic-scene line is built around
(docs/DYNAMIC_SCENES.md): run 1 leaves a scene graph and per-storey occupancy;
run 2 starts believing it, and the object has moved. The difference on a robot
is only that the world changes because somebody moved a chair rather than
because a script re-authored a layout.

Separate from `run_eval.py` because a run here is not an evaluation: there is
no episode dataset, no ground truth, nothing to score. What it shares is
everything that matters -- `build_env`, `build_run_components`, `build_agent`
and `run_episode` are the same functions the benchmark uses, so the robot runs
the measured pipeline and not a copy of it.

Needs the bridge up first:

    bash scripts/ros2/bridge.sh                   # a robot, or
    bash scripts/ros2/check_pipeline.sh           # the fake one
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osg.core.config import register_configs  # noqa: E402

register_configs()


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Imported late, as run_eval.py does, so `--help` does not load habitat.
    from osg.eval.episode import run_episode
    from osg.core.profiler import Profiler
    from osg.eval.record import (build_episode_record, detector_identity,
                                 episode_tag)
    from osg.eval.visualize import save_topdown
    from osg.graph.map_store import apply_map, load_map, save_map
    from osg.pipeline.components import (build_agent, build_env,
                                         build_run_components,
                                         probe_served_models)

    if str(cfg.eval.mode) != "ros2":
        raise ValueError(
            f"run_robot.py drives a robot, so eval.mode must be 'ros2' "
            f"(got {cfg.eval.mode!r}). Use +experiment=stretch3_map.")
    map_mode = str(cfg.ros2.map_mode)
    if map_mode not in ("map", "search"):
        raise ValueError(f"ros2.map_mode must be map|search, got {map_mode!r}")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    # Same refusal as a benchmark run: a model server that is down is not a
    # degraded run, and on a robot it is a robot driving on a dead detector.
    probe_served_models(cfg)

    env = build_env(cfg)
    components = build_run_components(cfg, env=env)
    env.attach_driver(components["pointnav"])
    target = env.target_category()
    map_path = Path(cfg.ros2.map_dir) / f"{cfg.ros2.map_tag}.json"

    print(f"[robot] {map_mode} run, target={target!r}, map={map_path}")
    frame = env.reset()
    episode = env.current_episode
    profiler = Profiler()
    agent = build_agent(
        cfg, components, target,
        keyframe_dir=str(out_dir / "keyframes" / episode_tag(episode))
        if cfg.eval.save_viz else None,
        profiler=profiler,
    )
    # The operator's floor switch, from the topic through to the map and the
    # scene graph (docs/ROS2.md). Needs floor.source=external to take effect.
    env.on_floor_switch = agent.floors.request_floor

    if map_mode == "search":
        if not map_path.exists():
            raise FileNotFoundError(
                f"no map at {map_path}. Run +experiment=stretch3_map first, "
                f"or point ros2.map_tag at one that exists.")
        n_tracks = apply_map(
            agent, load_map(map_path),
            max_log_odds=float(cfg.scene_graph.presence.reload_max_log_odds),
            initial_floor_y=float(frame.camera_position[1]) - float(cfg.agent.camera_height),
            # The storey is declared, not measured: a height comparison here
            # would be between two synthetic numbers (sim/ros2_env.py).
            initial_floor_key=int(env.floor_key),
        )
        print(f"[robot] restored {n_tracks} tracks on floor {env.floor_key}")

    try:
        outcome = run_episode(cfg, env, agent, episode, target, frame,
                              components["detector"])
    except KeyboardInterrupt:
        print("\n[robot] interrupted -- stopping the base")
        outcome = None
    finally:
        env.close()

    driver = components["pointnav"]
    print(f"[robot] {getattr(driver, 'n_goals_sent', 0)} goals posted to Nav2, "
          f"{getattr(driver, 'n_aborts', 0)} refused; "
          f"floor switches: {env.floor_switches or 'none'}")

    if map_mode == "map":
        path = save_map(map_path, agent, scene=str(cfg.ros2.map_tag))
        print(f"[robot] wrote {path}")

    if outcome is None:
        return

    scorer = components["scorer"]
    verifier = components["verifier"]
    rec = build_episode_record(
        cfg=cfg, episode=episode, env=env, agent=agent, outcome=outcome,
        target=target, detector_identity=detector_identity(cfg),
        metrics=env.metrics(), profiler=profiler, scorer=scorer,
        scorer_before=(0, 0, None),
        verifier=verifier, verifier_before=(0, 0),
    )
    with open(out_dir / "episodes.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    if cfg.eval.save_viz:
        save_topdown(
            str(out_dir / "viz" / f"{episode_tag(episode)}.png"),
            agent.costmap, outcome.trajectory, scene_graph=agent.scene_graph,
            title=f"{cfg.ros2.map_tag} target={target} steps={outcome.steps}",
        )
    print(f"[robot] {outcome.steps} steps, {outcome.wall_time_s}s -> {out_dir}")


if __name__ == "__main__":
    main()
