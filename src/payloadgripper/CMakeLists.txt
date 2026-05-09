cmake_minimum_required(VERSION 3.8)
project(payloadgripper)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(
  -Wall
  -Wextra
  -Wpedantic
  -Werror
  -Wno-unused-parameter
)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(px4_ros2_cpp REQUIRED)
find_package(px4_msgs REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

add_executable(payload_gripper_controller
  PayloadGripperController.cpp
  ServoGripper.cpp
)

target_include_directories(payload_gripper_controller PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Eigen3_INCLUDE_DIRS}
)

ament_target_dependencies(payload_gripper_controller
  rclcpp
  Eigen3
  px4_ros2_cpp
  px4_msgs
  geometry_msgs
  std_msgs
  tf2
  tf2_ros
)

target_compile_features(payload_gripper_controller PUBLIC cxx_std_20)

install(TARGETS
  payload_gripper_controller
  DESTINATION lib/${PROJECT_NAME}
)

# Cài toàn bộ thư mục cfg vào install/share/payloadgripper/cfg
install(DIRECTORY
  cfg
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()