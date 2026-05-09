cmake_minimum_required(VERSION 3.5)
project(udp_h264_camera)

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(udp_h264_camera_node src/UdpH264CameraNode.cpp)

ament_target_dependencies(udp_h264_camera_node
  rclcpp
  sensor_msgs
  cv_bridge
)

target_link_libraries(udp_h264_camera_node
  ${OpenCV_LIBS}
)

install(TARGETS
  udp_h264_camera_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()