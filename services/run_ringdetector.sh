
source install/setup.bash
source /home/pihuy/tracktor-beam/install/setup.bash

ros2 run ring_detector ring_detector --ros-args \
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
  -p far_roi_margin_ratio:=0.05
  # -p min_area:=1200.0 \
  # -p min_radius_px:=18.0 \
  # -p min_visible_size:=65.0 \
  # -p circularity_min:=0.12 \
  # -p aspect_min:=0.45 \
  # -p aspect_max:=2.20 \
  # -p far_min_area:=400.0 \
  # -p far_min_radius_px:=8.0 \
  # -p far_min_visible_size:=35.0 \
  # -p far_circularity_min:=0.06 \
  # -p far_aspect_min:=0.45 \
  # -p far_aspect_max:=2.40 \
  # -p far_confirm_frames:=3 \
  # -p far_confirm_pos_gate_px:=45.0 \
  # -p far_confirm_size_ratio:=0.40 \
  # -p far_roi_margin_ratio:=0.05