#!/usr/bin/env bash
# The ROS 2 half of the pipeline, on the SYSTEM python.
#
#   bash scripts/ros2/bridge.sh [--rgb-topic /a --depth-topic /b ...]
#
# Not the conda python, and not `conda run`: rclpy is built against Ubuntu
# 22.04's python3.10 and both conda envs in this image are 3.9 (habitat-sim
# 0.3.1 pins it). That is the whole reason there is a socket here instead of an
# import -- see src/osg/ros2/__init__.py.
set -eo pipefail
cd "$(dirname "$0")/../.."

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
[ -f "$ROS_SETUP" ] || { echo "no ROS 2 at $ROS_SETUP (rebuild the image)" >&2; exit 1; }
# shellcheck disable=SC1090
source "$ROS_SETUP"

# `src` on the path, not the habitat env's site-packages: osg.ros2 is written
# to import under both interpreters, and nothing else is imported here.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec /usr/bin/python3 -m osg.ros2.bridge_node "$@"
