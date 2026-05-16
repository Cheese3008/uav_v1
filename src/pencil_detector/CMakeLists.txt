cmake_minimum_required(VERSION 3.8)
project(pencil_detector)

set(CMAKE_CXX_STANDARD 17)
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
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(px4_msgs REQUIRED)

find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(OpenCV 4 QUIET)
if(NOT OpenCV_FOUND)
	find_package(OpenCV 4 REQUIRED HINTS
		/usr/local/lib/cmake/opencv4
		/opt/opencv/lib/cmake/opencv4
		ENV OpenCV_DIR
	)
endif()

# FrameTransformer + DataStructs dùng chung từ package cube_detector (cùng workspace src/)
get_filename_component(CUBE_DETECTOR_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../cube_detector" ABSOLUTE)
set(CUBE_FT_CPP "${CUBE_DETECTOR_SRC}/FrameTransformer.cpp")
if(NOT EXISTS "${CUBE_FT_CPP}")
	message(FATAL_ERROR "Không tìm thấy ${CUBE_FT_CPP}. Cần clone/build cả cube_detector cạnh pencil_detector trong src/.")
endif()

add_executable(pencil_detector
	PencilDetectorNode.cpp
	${CUBE_FT_CPP}
)

target_include_directories(pencil_detector PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}
	${CUBE_DETECTOR_SRC}
	${OpenCV_INCLUDE_DIRS}
	${Eigen3_INCLUDE_DIRS}
)

target_link_libraries(pencil_detector ${OpenCV_LIBS})

ament_target_dependencies(pencil_detector
	rclcpp
	sensor_msgs
	geometry_msgs
	std_msgs
	cv_bridge
	px4_msgs
	Eigen3
)

target_compile_features(pencil_detector PUBLIC cxx_std_17)

install(TARGETS
	pencil_detector
	DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
	cfg
	DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
	launch
	DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
	find_package(ament_lint_auto REQUIRED)
	ament_lint_auto_find_test_dependencies()
endif()

ament_package()
