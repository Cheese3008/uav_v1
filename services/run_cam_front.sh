#!/usr/bin/env bash

set -e

WORKSPACE="/home/pihuy/tracktor-beam"
CAMERA_INFO_FILE="${WORKSPACE}/camera_info.yaml"

source "${WORKSPACE}/install/setup.bash"

ros2 run camera_usb_gst_cpp camera_usb_gst_node \
  --ros-args \
  -r __node:=camera_front_gst_node \
  -p device_path:=/dev/camera_front \
  -p frame_id:=camera_front \
  -p topic_name:=/camera_front/image_raw \
  -p camera_info_topic:=/camera_front/camera_info \
  -p width:=640 \
  -p height:=480 \
  -p fps:=30 \
  -p use_mjpeg:=true \
  -p camera_name:=camera_front \
  -p camera_info_url:=file://${CAMERA_INFO_FILE}