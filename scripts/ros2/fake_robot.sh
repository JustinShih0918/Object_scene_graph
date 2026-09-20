#!/usr/bin/env bash
# A robot that does not exist, for checking the pipeline with no robot.
#
#   bash scripts/ros2/fake_robot.sh [--abort-goals] [--floor-switch-after 5]
#
# Publishes the camera topics, the TF chain and a NavigateToPose server, and
# actually moves when it is driven. System python, same reason as bridge.sh.
set -eo pipefail
cd "$(dirname "$0")/../.."

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
[ -f "$ROS_SETUP" ] || { echo "no ROS 2 at $ROS_SETUP (rebuild the image)" >&2; exit 1; }
# shellcheck disable=SC1090
source "$ROS_SETUP"

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec /usr/bin/python3 -m osg.ros2.fake_robot "$@"
