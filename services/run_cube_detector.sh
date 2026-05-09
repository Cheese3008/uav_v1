

#!/bin/bash

source install/setup.bash
source /home/pihuy/tracktor-beam/install/setup.bash

ros2 run cube_detector cube_detector --ros-args --params-file ~/tracktor-beam/src/cube_detector/cfg/params.yaml