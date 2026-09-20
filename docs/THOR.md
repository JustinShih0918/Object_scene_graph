# Deploying on a Jetson AGX Thor, beside a Stretch 3

`docs/ROS2.md` describes the layer; this describes where it runs. Three
participants, and the split between them is the whole design:

```
   ┌─ Jetson AGX Thor ──────────────────────────────┐        ┌─ Stretch 3 ────────┐
   │                                                │        │                    │
   │  osg-thor     the pipeline, the scene graph,   │        │  Nav2              │
   │  (L4T, CUDA)  the five perception models,      │        │  stretch_driver    │
   │               ollama. No ROS. No Habitat.      │        │  RealSense driver  │
   │                    ▲                           │        │  TF                │
   │                    │ socket 127.0.0.1:18765    │        │                    │
   │                    ▼                           │  DDS   │                    │
   │  osg-bridge   rclpy, and nothing else.         │◀──────▶│                    │
   │  (ros:humble) numpy + stdlib + ROS messages.   │        │                    │
   └────────────────────────────────────────────────┘        └────────────────────┘
```

**The robot keeps its own low-level control.** This stack never starts a navigator, a
controller or a driver. It publishes a goal pose to the Stretch's existing `NavigateToPose`
and reads back the camera and TF — see *Who owns the base* in `docs/ROS2.md` for the one tick
where that hands over.

## Why two containers on the Thor

Because the bridge genuinely needs nothing. With ROS stubbed out, importing
`osg.ros2.bridge_node` pulls numpy and the standard library and stops — no torch, no OpenCV,
no scipy. `src/osg/ros2/` was written that way because it has to import under two
interpreters, and on this platform that pays three times over:

- **The bridge matches the robot.** A Stretch 3 runs ROS 2 Humble on Ubuntu 22.04.
  `ros:humble-ros-base` *is* Ubuntu 22.04 with Humble's own `rclpy`, so both ends of every
  topic and action are the same distro. No cross-distro DDS, and no Humble compiled from
  source against a Python it was never tested against — which is what a single image on
  JetPack's Ubuntu would have forced.
- **The bridge needs no CUDA.** Nothing in it links against the host's L4T libraries, so a
  jammy userspace on JetPack's Ubuntu is not a compatibility question at all.
- **The Thor image needs no ROS**, and so is free to be whatever JetPack wants: system
  Python, NVIDIA's torch wheels, nothing bent around `rclpy`'s Python version.

They reach each other over the loopback because both take `network_mode: host` — the bridge
needs the host's network for DDS discovery (multicast, which the default bridge network
cannot carry) and the Thor container needs the same loopback to reach the bridge's socket.

## What is NOT in the Thor image

**Habitat.** There is no aarch64 build of habitat-sim — the `aihabitat` conda channel is
linux-64 only — and the robot does not need one. This is verified rather than assumed: with
`habitat`, `habitat_sim` and `magnum` blocked at import, the whole robot path (`Ros2Env` →
`NavAgent` → `run_episode` → `save_map`) runs to completion. `eval.mode=ros2` is the only
mode this image can run; every benchmark mode stays on the x86 image, which is where the
measured numbers come from anyway.

## Build and run

Needs the submodule, same as x86 — the perception servers are built from it:

```bash
git submodule update --init --recursive
cp docker/.env.example docker/.env      # set UID/GID, and the Jetson block

docker compose -f docker/compose.thor.yaml --env-file docker/.env build
docker compose -f docker/compose.thor.yaml --env-file docker/.env up -d
docker exec -it osg-thor bash
```

Then, inside `osg-thor`:

```bash
bash scripts/serve_perception.sh              # the five model servers
python scripts/run_robot.py +experiment=stretch3_map    ros2.map_tag=lab
# ... move something ...
python scripts/run_robot.py +experiment=stretch3_search ros2.map_tag=lab ros2.target=cup
```

`serve_perception.sh` needs no changes here: it already selects its interpreter through
`ASCENT_PYTHON`, which this image points at the models venv.

### Before the robot is in the room

The checks in `docs/ROS2.md` work on the Thor unchanged. Start with the one that needs
neither ROS nor a robot:

```bash
bash scripts/ros2/check_pipeline.sh --loopback     # everything above the bridge
```

Then, with the bridge container up and the fake robot running in it, the ROS path:

```bash
docker exec -it osg-bridge bash -c 'source /opt/ros/humble/setup.bash && ros2 topic list'
docker exec -it osg-bridge bash scripts/ros2/fake_robot.sh --floor-switch-after 6
docker exec -it osg-thor   python scripts/check_ros2_pipeline.py
```

## Set these for your unit

The image is parameterised because none of it can be checked from here. `docker/.env`:

| build arg | what it is | how to find it |
|---|---|---|
| `L4T_BASE` | the L4T image matching the device's JetPack | `cat /etc/nv_tegra_release` on the Thor; a mismatch between the container's CUDA userspace and the host driver is the failure that looks like a broken GPU |
| `CUDA_ARCH` | `TORCH_CUDA_ARCH_LIST` for the compiled kernels | `python3 -c "import torch; print(torch.cuda.get_device_capability())"` once torch is in. Thor is Blackwell-generation; Orin is `8.7`. Wrong value → GroundingDINO silently falls back to CPU, which is a day rather than a run |
| `TORCH_INDEX_URL` | NVIDIA's Jetson wheel index for that JetPack | NVIDIA does not publish Jetson torch on PyPI; the index is keyed by JetPack version |
| `WITH_BLIP2` | whether to install salesforce-lavis | `0` if it will not build on aarch64. It feeds only the value map and ASCENT's commit gate, and `exploration.value_model=clip` replaces it |
| `ROS_DOMAIN_ID` | must match the robot's | `echo $ROS_DOMAIN_ID` on the Stretch |
| `RMW_IMPLEMENTATION` | must match the robot's | the Stretch runs CycloneDDS; mismatched vendors discover each other unreliably and then fail in ways that look like network trouble |

The robot-side items — topic names, the portrait camera's rotation, the camera height, the
head-tilt joint — are all in the `ros2` config group and listed in `docs/ROS2.md`. `bridge.sh`
composes that group and passes it to the node, so settling one is a yaml edit on either
machine.

## If something is wrong

| symptom | first thing to check |
|---|---|
| `BridgeUnavailable: no ROS bridge at 127.0.0.1:18765` | the `osg-bridge` container is up, and **both** containers are on `network_mode: host` |
| `ros2 topic list` in the bridge shows nothing from the robot | `ROS_DOMAIN_ID` and `RMW_IMPLEMENTATION` match the Stretch's, and `ROS_LOCALHOST_ONLY=0` |
| `no camera frame within …s` | the topic names in the `ros2` group, and that TF actually carries `map` → the image header's frame |
| `torch.cuda.is_available()` is False at run time | `L4T_BASE` does not match the device's JetPack |
| GroundingDINO is slow | `CUDA_ARCH` is wrong, so the kernel compiled for another architecture and it fell back to CPU |
| the run refuses to start on a model server | `bash scripts/serve_perception.sh`, and `WITH_BLIP2=0` builds need `exploration.value_model=clip` |

## Files

| file | |
|---|---|
| `docker/Dockerfile.thor` | the pipeline and the models, on L4T |
| `docker/Dockerfile.bridge` | `ros:humble-ros-base` plus the message packages |
| `docker/compose.thor.yaml` | both containers, plus ollama, all on host networking |
| `docker/thor-model-requirements.txt` | the servers' aarch64 dependencies, and why they are not the x86 pins |
| `docker/entrypoint.thor.sh` | no conda wrapper: the pipeline env is the system python |
