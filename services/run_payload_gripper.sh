

#!/bin/bash

source install/setup.bash
source /home/pihuy/tracktor-beam/install/setup.bash
ros2 run payloadgripper payload_gripper_controller --ros-args --params-file /home/pihuy/tracktor-beam/src/payloadgripper/cfg/params.yaml