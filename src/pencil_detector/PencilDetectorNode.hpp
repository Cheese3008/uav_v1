#pragma once

#include <string>

#include <Eigen/Core>

#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <opencv2/opencv.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>

#include "FrameTransformer.hpp"
#include "pencilDetector.hpp"

class PencilDetectorNode : public rclcpp::Node
{
public:
	PencilDetectorNode();

private:
	struct Params
	{
		std::string image_topic = "/camera/image";
		std::string camera_info_topic = "/camera/camera_info";
		std::string vehicle_odometry_topic = "/fmu/out/vehicle_odometry";
		std::string reset_topic = "/pencil_detector/reset";

		std::string debug_topic = "/pencil_detector/algorithm_debug";
		std::string performance_topic = "/pencil_detector/performance";

		std::string target_pose_topic = "/pencil_detector/target_pose";
		std::string target_pose_camera_raw_topic = "/pencil_detector/target_pose_camera_raw";
		std::string target_pose_world_raw_topic = "/pencil_detector/target_pose_world_raw";
		std::string target_pose_world_filtered_topic = "/pencil_detector/target_pose_world_filtered";
		std::string target_velocity_world_filtered_topic = "/pencil_detector/target_velocity_world_filtered";
		std::string target_valid_topic = "/pencil_detector/target_valid";

		std::string camera_frame_id = "camera_optical_frame";
		std::string world_frame_id = "map_ned";

		double cam_offset_x = 0.0;
		double cam_offset_y = 0.0;
		double cam_offset_z = 0.09;

		/** Pinhole: kich thuc vat theo huong bbox lon (dai but), met. */
		double object_extent_m = 0.18;

		int yellow_h_low = 15;
		int yellow_s_low = 60;
		int yellow_v_low = 60;
		int yellow_h_high = 45;
		int yellow_s_high = 255;
		int yellow_v_high = 255;
		int morph_kernel_px = 5;
		double min_area_px = 800.0;

		double publish_hz_min = 0.0; // 0 = publish moi frame sau khi co camera_info
	} _p;

	rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _image_sub;
	rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr _camera_info_sub;
	rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr _reset_sub;
	rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr _vehicle_odometry_sub;

	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _target_pose_pub;
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _target_pose_camera_raw_pub;
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _target_pose_world_raw_pub;
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _target_pose_world_filtered_pub;
	rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr _target_velocity_world_filtered_pub;
	rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr _target_valid_pub;
	rclcpp::Publisher<std_msgs::msg::String>::SharedPtr _perf_pub;
	rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _debug_pub;

	bool _has_camera_info = false;
	cv::Mat _camera_matrix;
	cv::Mat _dist_coeffs;

	frame_transform::FrameTransformer _frame_transformer;
	bool _has_vehicle_pose = false;

	rclcpp::Time _last_publish_time = rclcpp::Time(0, 0, RCL_ROS_TIME);

	using Clock = std::chrono::steady_clock;
	Clock::time_point _perf_last_pub = Clock::now();
	uint64_t _perf_frame_count = 0;
	double _perf_total_ms = 0.0;

	void loadParameters();
	void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
	void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
	void reset_callback(const std_msgs::msg::Bool::SharedPtr msg);
	void vehicle_odometry_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg);

	yellow_pencil::DetectParams buildDetectParams() const;
	void opticalPoseFromBBox(double pixel_extent,
	                         const cv::Point2f &center,
	                         double &x,
	                         double &y,
	                         double &z) const;

	geometry_msgs::msg::PoseStamped build_pose(
		const std_msgs::msg::Header &header,
		const std::string &frame_id,
		double x,
		double y,
		double z) const;

	void publish_outputs(
		const std_msgs::msg::Header &header,
		double ox,
		double oy,
		double oz,
		bool valid);

	void publish_debug_if_needed(
		const cv::Mat &frame,
		const yellow_pencil::DetectResult &det,
		const std_msgs::msg::Header &header);
};
