cmake_minimum_required(VERSION 3.5)
project(ring_detector)

add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

# =========================
# ROS2 dependencies
# =========================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

# =========================
# Third-party dependencies
# =========================
find_package(Eigen3 REQUIRED)

# Nếu OpenCV 4.10.0 của bạn build ở /usr/local thì giữ dòng này
find_package(OpenCV 4.10.0 REQUIRED PATHS /usr/local NO_DEFAULT_PATH)

# Nếu dòng trên lỗi, đổi thành dòng này:
# find_package(OpenCV REQUIRED)

# =========================
# Executable
# =========================
add_executable(ring_detector
  RingDetector.cpp
)

target_include_directories(ring_detector PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

ament_target_dependencies(ring_detector
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
)

target_link_libraries(ring_detector
  ${OpenCV_LIBS}
  Eigen3::Eigen
)

target_compile_features(ring_detector PUBLIC cxx_std_17)

# =========================
# Install
# =========================
install(TARGETS
  ring_detector
  DESTINATION lib/${PROJECT_NAME}
)

# =========================
# Testing
# =========================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()
endif()

ament_package()