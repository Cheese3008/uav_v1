#!/usr/bin/env bash

set -e

WORKSPACE="/home/pihuy/tracktor-beam"
CAMERA_INFO_FILE="${WORKSPACE}/camera_info.yaml"

source "${WORKSPACE}/install/setup.bash"

ros2 run camera_usb_gst_cpp camera_usb_gst_node \
  --ros-args \
  -r __node:=camera_down_gst_node \
  -p device_path:=/dev/camera_down \
  -p frame_id:=camera_down \
  -p topic_name:=/camera_down/image_raw \
  -p camera_info_topic:=/camera_down/camera_info \
  -p width:=640 \
  -p height:=480 \
  -p fps:=30 \
  -p use_mjpeg:=true \
  -p camera_name:=camera_down \
  -p camera_info_url:=file://${CAMERA_INFO_FILE}