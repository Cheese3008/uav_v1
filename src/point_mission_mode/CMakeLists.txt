cmake_minimum_required(VERSION 3.5)
project(point_mission_mode)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(point_mission_mode
    PointMissionMode.cpp
    FrameTransformer.cpp
    ImageTargetDetector.cpp
    Kalman1D.cpp
    ImageErrorKalman.cpp
    ImageErrorVelocityController.cpp
)

target_include_directories(point_mission_mode PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(point_mission_mode
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

ament_target_dependencies(
    point_mission_mode
    rclcpp
    Eigen3
    px4_msgs
    std_msgs
    sensor_msgs
    cv_bridge
)

target_compile_features(point_mission_mode PUBLIC c_std_99 cxx_std_20)

install(TARGETS point_mission_mode DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)
install(DIRECTORY cfg DESTINATION share/${PROJECT_NAME}/)

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
endif()

ament_package()
