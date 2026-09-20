#!/usr/bin/env bash
# A robot that does not exist, for checking the pipeline with no robot.
#
#   bash scripts/ros2/fake_robot.sh [--abort-goals] [--floor-switch-after 5]
#
# Publishes the camera topics, the TF chain and a NavigateToPose server, and
# actually moves when it is driven. System python, same reason as bridge.sh.
set -eo pipefail
cd "$(dirname "$0")/../.."

# Overridable because the two images put ROS in different places: the x86
# development image installs it beside the pipeline, while on the Thor the
# bridge has a container to itself (docker/Dockerfile.bridge) and sets both of
# these in its environment.
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
ROS_PYTHON="${ROS_PYTHON:-/usr/bin/python3}"
[ -f "$ROS_SETUP" ] || { echo "no ROS 2 at $ROS_SETUP (rebuild the image)" >&2; exit 1; }
# shellcheck disable=SC1090
source "$ROS_SETUP"

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROS_PYTHON" -m osg.ros2.fake_robot "$@"
