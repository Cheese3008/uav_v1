cmake_minimum_required(VERSION 3.8)
project(camera_usb_gst_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# =========================
# Dependencies
# =========================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(camera_info_manager REQUIRED)

pkg_check_modules(GSTREAMER REQUIRED
  gstreamer-1.0
  gstreamer-app-1.0
)

# =========================
# Executable
# =========================
add_executable(camera_usb_gst_node
  camera_usb_gst_node.cpp
)

target_include_directories(camera_usb_gst_node PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${GSTREAMER_INCLUDE_DIRS}
)

ament_target_dependencies(camera_usb_gst_node
  rclcpp
  sensor_msgs
  camera_info_manager
)

target_link_libraries(camera_usb_gst_node
  ${GSTREAMER_LIBRARIES}
)

target_compile_features(camera_usb_gst_node PUBLIC cxx_std_17)

# =========================
# Install
# =========================
install(TARGETS
  camera_usb_gst_node
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