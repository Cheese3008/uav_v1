cmake_minimum_required(VERSION 3.8)
project(ring_pass_mode)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(px4_ros2_cpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(ring_pass_mode
  RingPassController.cpp
)

target_include_directories(ring_pass_mode PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
)

ament_target_dependencies(ring_pass_mode
  rclcpp
  px4_msgs
  px4_ros2_cpp
  geometry_msgs
  std_msgs
)

install(TARGETS ring_pass_mode
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY cfg
  DESTINATION share/${PROJECT_NAME}
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
  )
endif()

ament_package()
