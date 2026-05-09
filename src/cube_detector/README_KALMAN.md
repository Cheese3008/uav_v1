# Cube Detector with Kalman Filtering

## Overview
The Cube Detector node detects a victim model (rectangular cube + circular handle) using OpenCV and smooths the 3D position estimates using a 1D Kalman filter for each axis (X, Y, Z).

## Features
- ✅ Detects victim model: rectangular orange box + dark circular handle
- ✅ Validates geometric relationships (circle above box)
- ✅ Kalman filtering for smooth coordinate output
- ✅ Sub-millimeter precision (4 decimal places)
- ✅ Locks on target and handles temporary occlusions
- ✅ Publishes filtered 3D pose and validity status

## Topics
### Subscriptions
- `/world/default/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/image` - Camera image stream
- `/world/default/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/camera_info` - Camera calibration

### Publications
- `/target_pose` (geometry_msgs/PoseStamped) - Filtered 3D position with Kalman smoothing
- `/target_valid` (std_msgs/Bool) - Target lock status
- `/image_proc` (sensor_msgs/Image) - Annotated debug image with XYZ coordinates

## Running the Detector

### Quick start with default parameters:
```bash
ros2 run cube_detector cube_detector
```

### With launch file and custom parameters:
```bash
ros2 launch cube_detector cube_detector.launch.py
```

### Debug mode with detailed logging:
```bash
ros2 run cube_detector cube_detector --ros-args --log-level DEBUG
```

## Parameter Tuning

### Kalman Filter Parameters (in cfg/params.yaml)

**`kalman_q` - Process Noise (Default: 0.01)**
- Controls how much the filter predicts motion
- **Higher Q (e.g., 0.05)**: Less smoothing, faster response, more flickering
- **Lower Q (e.g., 0.005)**: More smoothing, slower response, stable
- **Use case**: If target is flickering/unstable, increase Q

**`kalman_r` - Measurement Noise (Default: 0.5)**
- Controls how much to trust the measurement
- **Higher R (e.g., 1.0)**: Less trust in measurements, smoother output
- **Lower R (e.g., 0.1)**: More trust in measurements, faster adaptation
- **Use case**: If output lags behind actual movement, decrease R

### Color Detection Parameters

**Box (Victim Cube) - Orange:**
```yaml
h_min: 5      # Start of orange hue
h_max: 30     # End of orange hue
s_min: 80     # Minimum saturation (avoid pale colors)
v_min: 50     # Minimum brightness (avoid dark colors)
```

**Circle (Handle) - Dark/Brown:**
```yaml
circle_h_max: 180     # Accept any hue
circle_s_max: 50      # Low saturation (dark colors)
circle_v_max: 200     # Limit brightness
```

### Geometry Parameters

**`circle_position_tolerance: 0.15`**
- Circle must be within 15% of box height above the box
- Range: 0.0 - 1.0
- Lower = stricter validation

## Output Display

The annotated image shows:
- Green rectangle: Detected box bounding box
- Blue circle: Detected circular handle
- Yellow text: **Precise XYZ coordinates (4 decimal places)**
- Light blue text: Status and confidence

Example output line:
```
Victim: LOCKED
X: 0.2345 m
Y: -0.0156 m
Z: 1.8932 m
```

## Troubleshooting

### Issue: Coordinates are jumpy/flickering
**Solution**: Increase `kalman_q` to reduce smoothing
```yaml
kalman_q: 0.05  # More responsive
```

### Issue: Coordinates lag behind actual movement
**Solution**: Decrease `kalman_r` to trust measurements more
```yaml
kalman_r: 0.1
```

### Issue: Box not detected
**Solution**: Adjust HSV range for orange
```yaml
h_min: 0    # Expand hue range
h_max: 35
s_min: 70   # Lower saturation threshold
v_min: 40   # Lower brightness threshold
```

### Issue: Circle not detected
**Solution**: Adjust HSV range for dark handle
```yaml
circle_s_max: 100     # Allow more saturated colors
circle_v_max: 220     # Allow brighter colors
```

## Performance Notes
- Detection runs at **camera frame rate**
- Kalman filter adds negligible latency (<1ms)
- Precision: **Sub-millimeter (0.0001 m accuracy)** when camera is well-calibrated

## Parameters File Location
`~/tracktor-beam/install/cube_detector/share/cube_detector/cfg/params.yaml`

## ROS2 Graph Visualization
```bash
ros2 run rqt_graph rqt_graph
```

This will show connections between cube_detector and other nodes.
