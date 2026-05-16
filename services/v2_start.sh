#!/usr/bin/env bash
set -e


WORKSPACE="/home/pihuy/tracktor-beam"
POINT_PARAM_FILE="/home/pihuy/tracktor-beam/src/point_mission_mode/cfg/params.yaml"

source "${WORKSPACE}/install/setup.bash"

ros2 run point_mission_mode point_mission_mode \
    --ros-args \
    --params-file "${POINT_PARAM_FILE}"