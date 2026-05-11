#!/usr/bin/env bash

set -e

# =========================
# Cau hinh chung
# =========================
WORKSPACE="$HOME/uav_v1"
ROS_DISTRO_NAME="humble"

# =========================
# Cau hinh camera UDP H264
# Tam thoi chua can camera vi point_mission_mode chua xu ly anh
# Khi can test camera thi doi ENABLE_CAMERA=true
# =========================
ENABLE_CAMERA=true
CAMERA_PACKAGE="udp_h264_camera"
CAMERA_EXECUTABLE="udp_h264_camera_node"

CAMERA_IMAGE_TOPIC="/camera_down/image_raw"
CAMERA_INFO_TOPIC="/camera_down/camera_info"
CAMERA_FRAME_ID="camera_down_link"
CAMERA_UDP_PORT="5600"

# =========================
# Cau hinh point mission mode
# =========================
POINT_PARAM_FILE="${WORKSPACE}/src/point_mission_mode/cfg/params.yaml"

# Neu chay Gazebo/SITL thi doi thanh true
USE_SIM_TIME=false

# =========================
# Source moi truong
# =========================
if [ -f "/opt/ros/${ROS_DISTRO_NAME}/setup.bash" ]; then
    source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
else
    echo "[ERROR] Khong tim thay /opt/ros/${ROS_DISTRO_NAME}/setup.bash"
    exit 1
fi

if [ -f "${WORKSPACE}/install/setup.bash" ]; then
    source "${WORKSPACE}/install/setup.bash"
else
    echo "[ERROR] Khong tim thay ${WORKSPACE}/install/setup.bash"
    echo "Hay build truoc:"
    echo "cd ${WORKSPACE} && colcon build --symlink-install"
    exit 1
fi

if [ ! -f "${POINT_PARAM_FILE}" ]; then
    echo "[ERROR] Khong tim thay file param:"
    echo "${POINT_PARAM_FILE}"
    exit 1
fi

# =========================
# Ham kill process cu neu dang chay
# =========================
kill_process_by_pattern()
{
    local PATTERN="$1"

    local OLD_PIDS
    OLD_PIDS=$(pgrep -f "${PATTERN}" || true)

    if [ -n "${OLD_PIDS}" ]; then
        echo "[INFO] Tim thay process cu: ${PATTERN}"
        echo "[INFO] PID: ${OLD_PIDS}"
        pkill -TERM -f "${PATTERN}" || true
        sleep 1

        OLD_PIDS=$(pgrep -f "${PATTERN}" || true)
        if [ -n "${OLD_PIDS}" ]; then
            echo "[WARN] Process chua tat, kill -9: ${PATTERN}"
            pkill -KILL -f "${PATTERN}" || true
            sleep 1
        fi
    else
        echo "[INFO] Khong co process cu: ${PATTERN}"
    fi
}

kill_old_nodes()
{
    echo "=========================================="
    echo "[INFO] Kiem tra va kill cac node cu neu co..."
    echo "=========================================="

    # Camera cu
    kill_process_by_pattern "ros2 run ${CAMERA_PACKAGE} ${CAMERA_EXECUTABLE}"
    kill_process_by_pattern "${CAMERA_EXECUTABLE}"

    # Ring node cu neu con dang chay
    kill_process_by_pattern "ros2 run ring_pass_mode ring_pass_mode"
    kill_process_by_pattern "ring_pass_mode"
    kill_process_by_pattern "ros2 run ring_detector ring_detector"
    kill_process_by_pattern "ring_detector"

    # Point mission mode cu
    kill_process_by_pattern "ros2 run point_mission_mode point_mission_mode"
    kill_process_by_pattern "ros2 launch point_mission_mode point_mission_mode.launch.py"
    kill_process_by_pattern "point_mission_mode"

    echo "[INFO] Da xu ly xong cac process cu."
}

# =========================
# Ham dung tat ca node khi Ctrl+C
# =========================
PIDS=()

cleanup()
{
    echo ""
    echo "[INFO] Dang dung tat ca node moi vua chay..."

    for PID in "${PIDS[@]}"; do
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" 2>/dev/null || true
        fi
    done

    wait 2>/dev/null || true
    echo "[INFO] Da dung xong."
}

trap cleanup INT TERM EXIT

run_node()
{
    local NAME="$1"
    local CMD="$2"

    echo "=========================================="
    echo "[START] ${NAME}"
    echo "[CMD] ${CMD}"
    echo "=========================================="

    bash -c "${CMD}" &
    PIDS+=("$!")

    sleep 1
}

# =========================
# Kill node cu truoc khi chay node moi
# =========================
kill_old_nodes

# =========================
# Chay camera front node neu can
# =========================
if [ "${ENABLE_CAMERA}" = true ]; then
    run_node "udp_h264_camera_front" \
    "ros2 run ${CAMERA_PACKAGE} ${CAMERA_EXECUTABLE} \
        --ros-args \
        -p image_topic:=${CAMERA_IMAGE_TOPIC} \
        -p camera_info_topic:=${CAMERA_INFO_TOPIC} \
        -p frame_id:=${CAMERA_FRAME_ID} \
        -p udp_port:=${CAMERA_UDP_PORT}"
fi

# =========================
# Chay point mission mode
# Mode: takeoff 3m -> F1 -> F3 -> F2 -> F4 -> F5 -> hold
# =========================
run_node "point_mission_mode" \
"ros2 run point_mission_mode point_mission_mode \
    --ros-args \
    --params-file ${POINT_PARAM_FILE} \
    -p use_sim_time:=${USE_SIM_TIME}"

echo ""
echo "[INFO] Point mission mode da chay."
echo "[INFO] Param file:"
echo "       ${POINT_PARAM_FILE}"
echo "[INFO] Use sim time:"
echo "       ${USE_SIM_TIME}"
echo ""
echo "[INFO] Debug topic:"
echo "       ros2 topic echo /point_mission/state_debug"
echo "       ros2 topic echo /point_mission/points_debug"
echo ""
echo "[INFO] Mode se xuat hien ten:"
echo "       POINT_NAV"
echo ""
echo "[INFO] Bam Ctrl+C de dung tat ca."
echo ""

wait