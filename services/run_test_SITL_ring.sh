#!/usr/bin/env bash

set -e

# =========================
# Cau hinh chung
# =========================
WORKSPACE="$HOME/uav_v1"
ROS_DISTRO_NAME="humble"

# =========================
# Cau hinh node camera UDP H264
# Camera front cho ring_detector
# =========================
ENABLE_CAMERA=true
CAMERA_PACKAGE="udp_h264_camera"
CAMERA_EXECUTABLE="udp_h264_camera_node"

CAMERA_IMAGE_TOPIC="/camera_front/image_raw"
CAMERA_INFO_TOPIC="/camera_front/camera_info"
CAMERA_FRAME_ID="camera_front_link"
CAMERA_UDP_PORT="5600"

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
# Chay camera front node
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
# Chay ring pass mode
# =========================
run_node "ring_pass_mode" \
"ros2 run ring_pass_mode ring_pass_mode"

# =========================
# Chay ring detector
# =========================
run_node "ring_detector" \
"ros2 run ring_detector ring_detector \
    --ros-args \
    -p image_topic:=/camera_front/image_raw \
    -p camera_info_topic:=/camera_front/camera_info \
    -p min_area:=0.0 \
    -p min_radius_px:=1.0 \
    -p min_visible_size:=0.0 \
    -p far_min_area:=0.0 \
    -p far_min_radius_px:=1.0 \
    -p far_min_visible_size:=0.0 \
    -p circularity_min:=0.0 \
    -p far_circularity_min:=0.0 \
    -p far_confirm_frames:=3 \
    -p far_confirm_pos_gate_px:=45.0 \
    -p far_confirm_size_ratio:=0.40 \
    -p far_roi_margin_ratio:=0.05"

echo ""
echo "[INFO] Tat ca node mode ring da chay."
echo "[INFO] Camera topic:"
echo "       ${CAMERA_IMAGE_TOPIC}"
echo "       ${CAMERA_INFO_TOPIC}"
echo "[INFO] Bam Ctrl+C de dung tat ca."
echo ""

wait