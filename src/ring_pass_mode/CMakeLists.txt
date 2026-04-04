cmake_minimum_required(VERSION 3.8)
project(ring_pass_mode)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(px4_ros2_cpp REQUIRED)

add_executable(ring_pass_mode_node
  RingPassMode.cpp
)

target_include_directories(ring_pass_mode_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

ament_target_dependencies(ring_pass_mode_node
  rclcpp
  geometry_msgs
  px4_msgs
  px4_ros2_cpp
)

target_link_libraries(ring_pass_mode_node
  Eigen3::Eigen
)

target_compile_features(ring_pass_mode_node PUBLIC cxx_std_17)

install(TARGETS
  ring_pass_mode_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()