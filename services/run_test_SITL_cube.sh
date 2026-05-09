#!/usr/bin/env bash

set -e

# =========================
# Cau hinh chung
# =========================
WORKSPACE="$HOME/uav_v1"

# Neu ban dung ROS distro khac humble thi sua dong nay
ROS_DISTRO_NAME="humble"

# =========================
# Cau hinh node camera UDP H264
# Neu package/executable camera cua ban khac ten thi sua 2 dong nay
# =========================
ENABLE_CAMERA=true
CAMERA_PACKAGE="udp_h264_camera"
CAMERA_EXECUTABLE="udp_h264_camera_node"

CAMERA_IMAGE_TOPIC="/camera_down/image_raw"
CAMERA_INFO_TOPIC="/camera_down/camera_info"
CAMERA_FRAME_ID="camera_down_link"
CAMERA_UDP_PORT="5600"

# =========================
# Source moi truong
# =========================
source "${WORKSPACE}/install/setup.bash"

if [ -f "${WORKSPACE}/install/setup.bash" ]; then
    source "${WORKSPACE}/install/setup.bash"
else
    echo "[ERROR] Khong tim thay ${WORKSPACE}/install/setup.bash"
    echo "Hay build truoc:"
    echo "cd ${WORKSPACE} && colcon build --symlink-install"
    exit 1
fi

# =========================
# Ham dung tat ca node khi Ctrl+C
# =========================
PIDS=()

cleanup()
{
    echo ""
    echo "[INFO] Dang dung tat ca node..."

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
    NAME="$1"
    CMD="$2"

    echo "=========================================="
    echo "[START] ${NAME}"
    echo "[CMD] ${CMD}"
    echo "=========================================="

    bash -c "${CMD}" &
    PIDS+=("$!")

    sleep 1
}

# =========================
# Chay camera node
# =========================
if [ "${ENABLE_CAMERA}" = true ]; then
    run_node "udp_h264_camera" \
    "ros2 run ${CAMERA_PACKAGE} ${CAMERA_EXECUTABLE} \
        --ros-args \
        -p image_topic:=${CAMERA_IMAGE_TOPIC} \
        -p camera_info_topic:=${CAMERA_INFO_TOPIC} \
        -p frame_id:=${CAMERA_FRAME_ID} \
        -p udp_port:=${CAMERA_UDP_PORT}"
fi

# =========================
# Chay cube detector
# =========================
run_node "cube_detector" \
"ros2 run cube_detector cube_detector \
    --ros-args \
    --params-file ${WORKSPACE}/src/cube_detector/cfg/params.yaml"

# =========================
# Chay payload gripper
# =========================
run_node "payload_gripper_controller" \
"ros2 run payloadgripper payload_gripper_controller \
    --ros-args \
    --params-file ${WORKSPACE}/src/payloadgripper/cfg/params.yaml"

echo ""
echo "[INFO] Tat ca node da chay."
echo "[INFO] Bam Ctrl+C de dung tat ca."
echo ""

wait