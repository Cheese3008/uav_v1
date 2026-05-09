

#!/bin/bash

source install/setup.bash
source /home/pihuy/tracktor-beam/install/setup.bash

ros2 run camera_usb_gst_cpp camera_usb_gst_node \
  --ros-args \
  -r __node:=camera_front_gst_node \
  -p device_path:=/dev/camera_front \
  -p frame_id:=camera_front \
  -p topic_name:=/camera_front/image_raw \
  -p camera_info_topic:=/camera_front/camera_info \
  -p width:=1280 \
  -p height:=720 \
  -p fps:=30 \
  -p use_mjpeg:=true