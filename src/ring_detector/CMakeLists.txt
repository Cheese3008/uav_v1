cmake_minimum_required(VERSION 3.5)
project(ring_detector)

add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4.10.0 REQUIRED PATHS /usr/local NO_DEFAULT_PATH)

find_package(Protobuf REQUIRED)
find_package(gz-transport13 REQUIRED)
find_package(gz-msgs10 REQUIRED)

add_executable(ring_detector
  RingDetector.cpp
)

target_include_directories(ring_detector PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(ring_detector
  ${OpenCV_LIBS}
  protobuf::libprotobuf
  gz-transport13::core
  gz-msgs10::core
  Eigen3::Eigen
)

ament_target_dependencies(ring_detector
  rclcpp
  sensor_msgs
  geometry_msgs
  cv_bridge
)

target_compile_features(ring_detector PUBLIC c_std_99 cxx_std_17)

install(TARGETS
  ring_detector
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()