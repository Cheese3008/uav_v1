#include "RingDetector.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <iomanip>
#include <sstream>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.hpp>

namespace
{
	constexpr double kPi = 3.14159265358979323846;

	std::string normalize_topic_namespace(const std::string &topic_namespace)
	{
		if (topic_namespace.empty() || topic_namespace == "/")
		{
			return "";
		}

		std::string normalized = topic_namespace;

		if (normalized.front() != '/')
		{
			normalized = "/" + normalized;
		}

		while (normalized.size() > 1 && normalized.back() == '/')
		{
			normalized.pop_back();
		}

		return normalized;
	}

	std::string make_namespaced_topic(
		const std::string &topic_namespace,
		const std::string &topic_name)
	{
		const std::string normalized_namespace =
			normalize_topic_namespace(topic_namespace);

		if (normalized_namespace.empty())
		{
			return "/" + topic_name;
		}

		return normalized_namespace + "/" + topic_name;
	}

	int make_odd_kernel_size(int kernel_size)
	{
		if (kernel_size < 3)
		{
			return 3;
		}

		if ((kernel_size % 2) == 0)
		{
			return kernel_size + 1;
		}

		return kernel_size;
	}

	double calc_distance(const cv::Point2f &a, const cv::Point2f &b)
	{
		const double dx = static_cast<double>(a.x - b.x);
		const double dy = static_cast<double>(a.y - b.y);
		return std::sqrt(dx * dx + dy * dy);
	}

	std::vector<cv::Point> offset_contour(
		const std::vector<cv::Point> &contour,
		const cv::Point &offset)
	{
		std::vector<cv::Point> shifted = contour;

		for (auto &point : shifted)
		{
			point.x += offset.x;
			point.y += offset.y;
		}

		return shifted;
	}

	double calc_mean_radial_error(
		const std::vector<cv::Point> &contour,
		const cv::Point2f &center,
		double radius)
	{
		if (contour.empty() || radius <= 1e-6)
		{
			return 1e9;
		}

		double error_sum = 0.0;
		for (const auto &point : contour)
		{
			const double dx = static_cast<double>(point.x) - static_cast<double>(center.x);
			const double dy = static_cast<double>(point.y) - static_cast<double>(center.y);
			const double dist = std::sqrt(dx * dx + dy * dy);
			error_sum += std::abs(dist - radius) / radius;
		}

		return error_sum / static_cast<double>(contour.size());
	}
} // namespace

RingDetectorNode::RingDetectorNode()
	: Node("ring_detector_node")
{
	loadParameters();
	configureBodyKalman();

	auto image_qos = rclcpp::SensorDataQoS();
	auto info_qos = rclcpp::SensorDataQoS();
	auto pub_qos = rclcpp::QoS(10).reliable();

	_image_sub = create_subscription<sensor_msgs::msg::Image>(
		_image_topic,
		image_qos,
		std::bind(&RingDetectorNode::image_callback, this, std::placeholders::_1));

	_camera_info_sub = create_subscription<sensor_msgs::msg::CameraInfo>(
		_camera_info_topic,
		info_qos,
		std::bind(&RingDetectorNode::camera_info_callback, this, std::placeholders::_1));

	_image_pub = create_publisher<sensor_msgs::msg::Image>(
		_processed_image_topic,
		pub_qos);

	_target_valid_pub = create_publisher<std_msgs::msg::Bool>(
		_target_valid_topic,
		pub_qos);

	if (_param_publish_reset_status)
	{
		_reset_status_pub = create_publisher<std_msgs::msg::String>(
			_reset_status_topic,
			pub_qos);
	}

	_target_pose_camera_raw_pub = create_publisher<geometry_msgs::msg::PoseStamped>(
		_target_pose_camera_raw_topic,
		pub_qos);

	_target_error_body_filtered_pub = create_publisher<geometry_msgs::msg::PoseStamped>(
		_target_error_body_filtered_topic,
		pub_qos);

	_target_velocity_body_filtered_pub = create_publisher<geometry_msgs::msg::TwistStamped>(
		_target_velocity_body_filtered_topic,
		pub_qos);

	_ring_detect_reset_sub = create_subscription<std_msgs::msg::String>(
		_ring_detect_reset_topic,
		rclcpp::QoS(10).reliable(),
		std::bind(&RingDetectorNode::ring_detect_reset_callback, this, std::placeholders::_1));

	RCLCPP_INFO(get_logger(), "RingDetector image input              : %s", _image_topic.c_str());
	RCLCPP_INFO(get_logger(), "RingDetector camera info input        : %s", _camera_info_topic.c_str());
	RCLCPP_INFO(get_logger(), "RingDetector processed image out      : %s", _processed_image_topic.c_str());
	RCLCPP_INFO(get_logger(), "RingDetector target valid out         : %s", _target_valid_topic.c_str());
	RCLCPP_INFO(get_logger(), "RingDetector reset status publish     : %s", _param_publish_reset_status ? "true" : "false");
	if (_param_publish_reset_status)
	{
		RCLCPP_INFO(get_logger(), "RingDetector reset status out         : %s", _reset_status_topic.c_str());
	}
	RCLCPP_INFO(get_logger(), "RingDetector reset command input      : %s", _ring_detect_reset_topic.c_str());
	RCLCPP_INFO(get_logger(), "RingDetector camera raw pose out      : %s", _target_pose_camera_raw_topic.c_str());
	RCLCPP_INFO(get_logger(), "RingDetector body filtered error out  : %s", _target_error_body_filtered_topic.c_str());
	RCLCPP_INFO(get_logger(), "RingDetector body filtered vel out    : %s", _target_velocity_body_filtered_topic.c_str());
	RCLCPP_INFO(get_logger(), "RingDetector simple center ROI ratio  : %.2f", _param_center_roi_ratio);
	RCLCPP_INFO(get_logger(), "RingDetector gray min/max             : %d / %d", _param_gray_min, _param_gray_max);
	RCLCPP_INFO(get_logger(), "RingDetector gray color diff max      : %d", _param_gray_color_diff_max);
	RCLCPP_INFO(get_logger(), "RingDetector circle ratio min/max     : %.2f / %.2f", _param_min_circle_area_ratio, _param_max_circle_area_ratio);
	RCLCPP_INFO(get_logger(), "RingDetector max radial error         : %.2f", _param_max_radial_error);
	RCLCPP_INFO(get_logger(), "RingDetector Canny circle detector    : %s", _param_use_canny_circle_detector ? "true" : "false");
	RCLCPP_INFO(get_logger(), "RingDetector Canny low/high           : %.1f / %.1f", _param_canny_low, _param_canny_high);
	RCLCPP_INFO(get_logger(), "RingDetector publish image_proc       : %s", _param_publish_image ? "true" : "false");
	RCLCPP_INFO(get_logger(), "RingDetector pipeline debug           : %s", _param_publish_pipeline_debug ? "true" : "false");
	RCLCPP_INFO(get_logger(), "RingDetector YUV neutral mask         : %s", _param_use_yuv_neutral_mask ? "true" : "false");
	RCLCPP_INFO(get_logger(), "RingDetector area ratio filter        : %s", _param_use_circle_area_ratio_filter ? "true" : "false");
	RCLCPP_INFO(get_logger(), "RingDetector Single Body Kalman q_acc        : %.3f", _param_body_kf_q_acc);
	RCLCPP_INFO(get_logger(), "RingDetector Single Body Kalman R x/y/z      : %.3f / %.3f / %.3f",
		_param_body_kf_r_x,
		_param_body_kf_r_y,
		_param_body_kf_r_z);
	RCLCPP_INFO(get_logger(), "RingDetector Single Body Kalman gate         : %s px=%.1f depth_ratio=%.2f",
		_param_track_use_kf_gate ? "true" : "false",
		_param_track_kf_pixel_gate_px,
		_param_track_kf_depth_ratio_gate);
}

void RingDetectorNode::loadParameters()
{
	declare_parameter<std::string>("image_topic", "/camera_front/image_raw");
	declare_parameter<std::string>("camera_info_topic", "/camera_front/camera_info");

	declare_parameter<std::string>("output_namespace", "/ring_detect");
	declare_parameter<std::string>("processed_image_topic", "");
	declare_parameter<std::string>("color_detect_debug_topic", "");
	declare_parameter<std::string>("target_valid_topic", "");
	declare_parameter<std::string>("reset_status_topic", "");
	declare_parameter<std::string>("ring_detect_reset_topic", "");
	declare_parameter<std::string>("target_pose_camera_raw_topic", "");
	declare_parameter<std::string>("target_error_body_raw_topic", "");
	declare_parameter<std::string>("target_error_body_filtered_topic", "");
	declare_parameter<std::string>("target_velocity_body_filtered_topic", "");
	declare_parameter<std::string>("target_frame_id", "camera_front_optical_frame");
	declare_parameter<std::string>("body_frame_id", "base_link_frd");

	declare_parameter<double>("ring_diameter_m", 1.6);
	declare_parameter<double>("min_area", 250.0);
	declare_parameter<double>("min_radius_px", 8.0);
	declare_parameter<double>("min_visible_size", 20.0);
	declare_parameter<double>("circularity_min", 0.35);
	declare_parameter<double>("aspect_min", 0.65);
	declare_parameter<double>("aspect_max", 1.55);

	declare_parameter<bool>("use_white_mask", true);
	declare_parameter<bool>("publish_image", true);
	declare_parameter<bool>("publish_color_detect_debug", false);
	declare_parameter<bool>("publish_pipeline_debug", false);
	declare_parameter<bool>("debug_draw_rejected_contours", true);
	declare_parameter<int>("debug_panel_width", 520);

	// Detector màu trung tính: nhận từ trắng tới xám đen, không dùng HSV.
	declare_parameter<int>("gray_min", 45);
	declare_parameter<int>("gray_max", 255);
	declare_parameter<int>("gray_color_diff_max", 55);
	declare_parameter<int>("gray_morph_kernel", 3);
	declare_parameter<bool>("use_yuv_neutral_mask", true);
	declare_parameter<int>("yuv_u_center", 128);
	declare_parameter<int>("yuv_v_center", 128);
	declare_parameter<int>("yuv_uv_diff_max", 35);
	declare_parameter<int>("yuv_median_kernel", 5);

	// Giữ tham số cũ để không lỗi khi dùng YAML cũ.
	declare_parameter<int>("white_min", 150);
	declare_parameter<int>("white_color_diff_max", 45);
	declare_parameter<int>("white_morph_kernel", 3);

	// Tham số mới cho detector nhẹ: chỉ xử lý vùng giữa ảnh.
	declare_parameter<double>("center_roi_ratio", 0.70);
	declare_parameter<double>("max_center_distance_ratio", 0.42);
	declare_parameter<double>("min_fill_ratio", 0.12);
	declare_parameter<double>("max_fill_ratio", 1.25);

	// Lọc hình tròn thật để loại hình vuông.
	declare_parameter<double>("min_circle_area_ratio", 0.72);
	declare_parameter<double>("max_circle_area_ratio", 1.20);
	declare_parameter<bool>("use_circle_area_ratio_filter", false);
	declare_parameter<double>("max_radial_error", 0.20);
	declare_parameter<int>("min_approx_vertices", 8);
	declare_parameter<double>("approx_epsilon_ratio", 0.018);

	// Detector cạnh kiểu tham khảo Python:
	// neutral gray mask -> bilateral -> Canny -> contour -> approxPolyDP.
	declare_parameter<bool>("use_canny_circle_detector", true);
	declare_parameter<bool>("use_gray_binary_threshold", false);
	declare_parameter<int>("gray_binary_threshold", 45);
	declare_parameter<bool>("use_bilateral_filter", true);
	declare_parameter<int>("bilateral_d", 5);
	declare_parameter<double>("bilateral_sigma_color", 175.0);
	declare_parameter<double>("bilateral_sigma_space", 175.0);
	declare_parameter<double>("canny_low", 75.0);
	declare_parameter<double>("canny_high", 200.0);
	declare_parameter<int>("edge_dilate_kernel", 3);
	declare_parameter<double>("min_ellipse_axis_ratio", 0.78);
	declare_parameter<double>("max_ellipse_center_shift_ratio", 0.18);

	// Giữ các tham số cũ để YAML cũ không bị lệch, nhưng bản nhẹ không dùng far mode nữa.
	declare_parameter<double>("far_min_area", 120.0);
	declare_parameter<double>("far_min_radius_px", 5.0);
	declare_parameter<double>("far_min_visible_size", 20.0);
	declare_parameter<double>("far_circularity_min", 0.02);
	declare_parameter<double>("far_aspect_min", 0.35);
	declare_parameter<double>("far_aspect_max", 3.00);
	declare_parameter<double>("far_canny_low", 35.0);
	declare_parameter<double>("far_canny_high", 110.0);
	declare_parameter<double>("far_roi_margin_ratio", 0.01);
	declare_parameter<int>("far_confirm_frames", 1);
	declare_parameter<double>("far_confirm_pos_gate_px", 45.0);
	declare_parameter<double>("far_confirm_size_ratio", 0.45);

	declare_parameter<double>("lock_max_dist", 140.0);
	declare_parameter<double>("lock_min_size_ratio", 0.45);
	declare_parameter<int>("max_lock_missed", 15);

	declare_parameter<int>("hold_max_missed", 8);
	declare_parameter<double>("pass_x_thresh", 0.35);
	declare_parameter<double>("pass_min_visible_size", 220.0);
	declare_parameter<double>("body_kf_hold_timeout_s", 0.20);
	declare_parameter<double>("reset_timeout_s", 0.35);
	declare_parameter<bool>("publish_reset_status", false);

	declare_parameter<double>("camera_offset_x", 0.09);
	declare_parameter<double>("camera_offset_y", 0.0);
	declare_parameter<double>("camera_offset_z", 0.0);

	declare_parameter<double>("body_kf_q_acc", 0.80);
	declare_parameter<double>("body_kf_r_x", 0.02);
	declare_parameter<double>("body_kf_r_y", 0.04);
	declare_parameter<double>("body_kf_r_z", 0.02);

	// track_* chi dung de gate khoa vong dua tren projection tu Kalman.
	declare_parameter<bool>("track_use_kf_gate", true);
	declare_parameter<double>("track_kf_pixel_gate_px", 90.0);
	declare_parameter<double>("track_kf_depth_ratio_gate", 0.45);


	get_parameter("image_topic", _image_topic);
	get_parameter("camera_info_topic", _camera_info_topic);

	std::string output_namespace;
	std::string processed_image_topic;
	std::string target_valid_topic;
	std::string reset_status_topic;
	std::string ring_detect_reset_topic;
	std::string target_pose_camera_raw_topic;
	std::string target_error_body_raw_topic;
	std::string target_error_body_filtered_topic;
	std::string target_velocity_body_filtered_topic;

	get_parameter("output_namespace", output_namespace);
	get_parameter("processed_image_topic", processed_image_topic);
	get_parameter("target_valid_topic", target_valid_topic);
	get_parameter("reset_status_topic", reset_status_topic);
	get_parameter("ring_detect_reset_topic", ring_detect_reset_topic);
	get_parameter("target_pose_camera_raw_topic", target_pose_camera_raw_topic);
	get_parameter("target_error_body_raw_topic", target_error_body_raw_topic);
	get_parameter("target_error_body_filtered_topic", target_error_body_filtered_topic);
	get_parameter("target_velocity_body_filtered_topic", target_velocity_body_filtered_topic);
	get_parameter("target_frame_id", _target_frame_id);
	get_parameter("body_frame_id", _body_frame_id);

	_processed_image_topic = processed_image_topic.empty()
								 ? make_namespaced_topic(output_namespace, "image_proc")
								 : processed_image_topic;

	_target_valid_topic = target_valid_topic.empty()
							 ? make_namespaced_topic(output_namespace, "target_valid")
							 : target_valid_topic;

	_reset_status_topic = reset_status_topic.empty()
							  ? make_namespaced_topic(output_namespace, "reset_status")
							  : reset_status_topic;

	_ring_detect_reset_topic = ring_detect_reset_topic.empty()
								   ? make_namespaced_topic(output_namespace, "reset")
								   : ring_detect_reset_topic;

	_target_pose_camera_raw_topic = target_pose_camera_raw_topic.empty()
										? make_namespaced_topic(output_namespace, "target_pose_camera_raw")
										: target_pose_camera_raw_topic;


	_target_error_body_filtered_topic = target_error_body_filtered_topic.empty()
											? make_namespaced_topic(output_namespace, "target_error_body_filtered")
											: target_error_body_filtered_topic;

	_target_velocity_body_filtered_topic = target_velocity_body_filtered_topic.empty()
											   ? make_namespaced_topic(output_namespace, "target_velocity_body_filtered")
											   : target_velocity_body_filtered_topic;

	get_parameter("ring_diameter_m", _param_ring_diameter_m);
	get_parameter("min_area", _param_min_area);
	get_parameter("min_radius_px", _param_min_radius_px);
	get_parameter("min_visible_size", _param_min_visible_size);
	get_parameter("circularity_min", _param_circularity_min);
	get_parameter("aspect_min", _param_aspect_min);
	get_parameter("aspect_max", _param_aspect_max);

	get_parameter("use_white_mask", _param_use_white_mask);
	get_parameter("publish_image", _param_publish_image);
	get_parameter("publish_color_detect_debug", _param_publish_color_detect_debug);
	get_parameter("publish_pipeline_debug", _param_publish_pipeline_debug);
	get_parameter("debug_draw_rejected_contours", _param_debug_draw_rejected_contours);
	get_parameter("debug_panel_width", _param_debug_panel_width);
	get_parameter("gray_min", _param_gray_min);
	get_parameter("gray_max", _param_gray_max);
	get_parameter("gray_color_diff_max", _param_gray_color_diff_max);
	get_parameter("gray_morph_kernel", _param_gray_morph_kernel);
	get_parameter("use_yuv_neutral_mask", _param_use_yuv_neutral_mask);
	get_parameter("yuv_u_center", _param_yuv_u_center);
	get_parameter("yuv_v_center", _param_yuv_v_center);
	get_parameter("yuv_uv_diff_max", _param_yuv_uv_diff_max);
	get_parameter("yuv_median_kernel", _param_yuv_median_kernel);
	get_parameter("white_min", _param_white_min);
	get_parameter("white_color_diff_max", _param_white_color_diff_max);
	get_parameter("white_morph_kernel", _param_white_morph_kernel);

	get_parameter("center_roi_ratio", _param_center_roi_ratio);
	get_parameter("max_center_distance_ratio", _param_max_center_distance_ratio);
	get_parameter("min_fill_ratio", _param_min_fill_ratio);
	get_parameter("max_fill_ratio", _param_max_fill_ratio);
	get_parameter("min_circle_area_ratio", _param_min_circle_area_ratio);
	get_parameter("max_circle_area_ratio", _param_max_circle_area_ratio);
	get_parameter("use_circle_area_ratio_filter", _param_use_circle_area_ratio_filter);
	get_parameter("max_radial_error", _param_max_radial_error);
	get_parameter("min_approx_vertices", _param_min_approx_vertices);
	get_parameter("approx_epsilon_ratio", _param_approx_epsilon_ratio);
	get_parameter("use_canny_circle_detector", _param_use_canny_circle_detector);
	get_parameter("use_gray_binary_threshold", _param_use_gray_binary_threshold);
	get_parameter("gray_binary_threshold", _param_gray_binary_threshold);
	get_parameter("use_bilateral_filter", _param_use_bilateral_filter);
	get_parameter("bilateral_d", _param_bilateral_d);
	get_parameter("bilateral_sigma_color", _param_bilateral_sigma_color);
	get_parameter("bilateral_sigma_space", _param_bilateral_sigma_space);
	get_parameter("canny_low", _param_canny_low);
	get_parameter("canny_high", _param_canny_high);
	get_parameter("edge_dilate_kernel", _param_edge_dilate_kernel);
	get_parameter("min_ellipse_axis_ratio", _param_min_ellipse_axis_ratio);
	get_parameter("max_ellipse_center_shift_ratio", _param_max_ellipse_center_shift_ratio);

	get_parameter("far_min_area", _param_far_min_area);
	get_parameter("far_min_radius_px", _param_far_min_radius_px);
	get_parameter("far_min_visible_size", _param_far_min_visible_size);
	get_parameter("far_circularity_min", _param_far_circularity_min);
	get_parameter("far_aspect_min", _param_far_aspect_min);
	get_parameter("far_aspect_max", _param_far_aspect_max);
	get_parameter("far_canny_low", _param_far_canny_low);
	get_parameter("far_canny_high", _param_far_canny_high);
	get_parameter("far_roi_margin_ratio", _param_far_roi_margin_ratio);
	get_parameter("far_confirm_frames", _param_far_confirm_frames);
	get_parameter("far_confirm_pos_gate_px", _param_far_confirm_pos_gate_px);
	get_parameter("far_confirm_size_ratio", _param_far_confirm_size_ratio);

	get_parameter("lock_max_dist", _param_lock_max_dist);
	get_parameter("lock_min_size_ratio", _param_lock_min_size_ratio);
	get_parameter("max_lock_missed", _param_max_lock_missed);

	get_parameter("hold_max_missed", _param_hold_max_missed);
	get_parameter("pass_x_thresh", _param_pass_x_thresh);
	get_parameter("pass_min_visible_size", _param_pass_min_visible_size);
	get_parameter("body_kf_hold_timeout_s", _param_body_kf_hold_timeout_s);
	get_parameter("reset_timeout_s", _param_reset_timeout_s);
	get_parameter("publish_reset_status", _param_publish_reset_status);

	get_parameter("camera_offset_x", _param_camera_offset_x);
	get_parameter("camera_offset_y", _param_camera_offset_y);
	get_parameter("camera_offset_z", _param_camera_offset_z);

	get_parameter("body_kf_q_acc", _param_body_kf_q_acc);
	get_parameter("body_kf_r_x", _param_body_kf_r_x);
	get_parameter("body_kf_r_y", _param_body_kf_r_y);
	get_parameter("body_kf_r_z", _param_body_kf_r_z);

	get_parameter("track_use_kf_gate", _param_track_use_kf_gate);
	get_parameter("track_kf_pixel_gate_px", _param_track_kf_pixel_gate_px);
	get_parameter("track_kf_depth_ratio_gate", _param_track_kf_depth_ratio_gate);
}

void RingDetectorNode::configureBodyKalman()
{
	_body_kf.init(6, 3, 0, CV_64F);

	_body_kf.transitionMatrix = cv::Mat::eye(6, 6, CV_64F);
	_body_kf.measurementMatrix = cv::Mat::zeros(3, 6, CV_64F);
	_body_kf.measurementMatrix.at<double>(0, 0) = 1.0;
	_body_kf.measurementMatrix.at<double>(1, 1) = 1.0;
	_body_kf.measurementMatrix.at<double>(2, 2) = 1.0;

	_body_kf.statePre = cv::Mat::zeros(6, 1, CV_64F);
	_body_kf.statePost = cv::Mat::zeros(6, 1, CV_64F);
	_body_kf.processNoiseCov = cv::Mat::eye(6, 6, CV_64F) * 1e-6;
	_body_kf.measurementNoiseCov = cv::Mat::zeros(3, 3, CV_64F);
	_body_kf.measurementNoiseCov.at<double>(0, 0) = std::max(_param_body_kf_r_x, 1e-9);
	_body_kf.measurementNoiseCov.at<double>(1, 1) = std::max(_param_body_kf_r_y, 1e-9);
	_body_kf.measurementNoiseCov.at<double>(2, 2) = std::max(_param_body_kf_r_z, 1e-9);

	cv::setIdentity(_body_kf.errorCovPre, cv::Scalar(1.0));
	cv::setIdentity(_body_kf.errorCovPost, cv::Scalar(1.0));

	_body_kf_initialized = false;
	_body_kf_position = Eigen::Vector3d::Zero();
	_body_kf_velocity = Eigen::Vector3d::Zero();
	_body_kf_last_time = rclcpp::Time(0, 0, RCL_ROS_TIME);
}

void RingDetectorNode::initBodyKalman(const Eigen::Vector3d &body_measurement)
{
	configureBodyKalman();

	_body_kf.statePre.at<double>(0, 0) = body_measurement.x();
	_body_kf.statePre.at<double>(1, 0) = body_measurement.y();
	_body_kf.statePre.at<double>(2, 0) = body_measurement.z();
	_body_kf.statePre.at<double>(3, 0) = 0.0;
	_body_kf.statePre.at<double>(4, 0) = 0.0;
	_body_kf.statePre.at<double>(5, 0) = 0.0;

	_body_kf.statePost = _body_kf.statePre.clone();
	_body_kf_position = body_measurement;
	_body_kf_velocity = Eigen::Vector3d::Zero();
	_body_kf_initialized = true;
}

void RingDetectorNode::predictBodyKalman(double dt_s)
{
	if (!_body_kf_initialized)
	{
		return;
	}

	const double dt = std::clamp(dt_s, 1e-3, 0.2);
	const double q_acc = std::max(_param_body_kf_q_acc, 1e-9);

	_body_kf.transitionMatrix = cv::Mat::eye(6, 6, CV_64F);
	_body_kf.transitionMatrix.at<double>(0, 3) = dt;
	_body_kf.transitionMatrix.at<double>(1, 4) = dt;
	_body_kf.transitionMatrix.at<double>(2, 5) = dt;

	_body_kf.processNoiseCov = cv::Mat::zeros(6, 6, CV_64F);
	const double q_pos = 0.25 * dt * dt * dt * dt * q_acc;
	const double q_cross = 0.50 * dt * dt * dt * q_acc;
	const double q_vel = dt * dt * q_acc;

	for (int axis = 0; axis < 3; ++axis)
	{
		const int pos_idx = axis;
		const int vel_idx = axis + 3;
		_body_kf.processNoiseCov.at<double>(pos_idx, pos_idx) = q_pos;
		_body_kf.processNoiseCov.at<double>(pos_idx, vel_idx) = q_cross;
		_body_kf.processNoiseCov.at<double>(vel_idx, pos_idx) = q_cross;
		_body_kf.processNoiseCov.at<double>(vel_idx, vel_idx) = q_vel;
	}

	const cv::Mat prediction = _body_kf.predict();
	_body_kf_position.x() = prediction.at<double>(0, 0);
	_body_kf_position.y() = prediction.at<double>(1, 0);
	_body_kf_position.z() = prediction.at<double>(2, 0);
	_body_kf_velocity.x() = prediction.at<double>(3, 0);
	_body_kf_velocity.y() = prediction.at<double>(4, 0);
	_body_kf_velocity.z() = prediction.at<double>(5, 0);
}

void RingDetectorNode::updateBodyKalman(const Eigen::Vector3d &body_measurement)
{
	if (!_body_kf_initialized)
	{
		initBodyKalman(body_measurement);
		return;
	}

	_body_kf.measurementNoiseCov.at<double>(0, 0) = std::max(_param_body_kf_r_x, 1e-9);
	_body_kf.measurementNoiseCov.at<double>(1, 1) = std::max(_param_body_kf_r_y, 1e-9);
	_body_kf.measurementNoiseCov.at<double>(2, 2) = std::max(_param_body_kf_r_z, 1e-9);

	cv::Mat measurement_vector(3, 1, CV_64F);
	measurement_vector.at<double>(0, 0) = body_measurement.x();
	measurement_vector.at<double>(1, 0) = body_measurement.y();
	measurement_vector.at<double>(2, 0) = body_measurement.z();

	const cv::Mat corrected = _body_kf.correct(measurement_vector);
	_body_kf_position.x() = corrected.at<double>(0, 0);
	_body_kf_position.y() = corrected.at<double>(1, 0);
	_body_kf_position.z() = corrected.at<double>(2, 0);
	_body_kf_velocity.x() = corrected.at<double>(3, 0);
	_body_kf_velocity.y() = corrected.at<double>(4, 0);
	_body_kf_velocity.z() = corrected.at<double>(5, 0);
}

void RingDetectorNode::resetBodyKalman()
{
	configureBodyKalman();
	_last_measurement_time = rclcpp::Time(0, 0, RCL_ROS_TIME);
}

Eigen::Vector3d RingDetectorNode::getBodyKalmanPosition() const
{
	return _body_kf_position;
}

Eigen::Vector3d RingDetectorNode::getBodyKalmanVelocity() const
{
	return _body_kf_velocity;
}

Eigen::Vector3d RingDetectorNode::cameraOpticalToBodyXyz(
	const Eigen::Vector3d &camera_position) const
{
	Eigen::Vector3d body_position;
	body_position.x() = camera_position.z() + _param_camera_offset_x;
	body_position.y() = camera_position.x() + _param_camera_offset_y;
	body_position.z() = camera_position.y() + _param_camera_offset_z;

	return body_position;
}

Eigen::Vector3d RingDetectorNode::bodyXyzToCameraOptical(
	const Eigen::Vector3d &body_position) const
{
	Eigen::Vector3d camera_position;
	camera_position.x() = body_position.y() - _param_camera_offset_y;
	camera_position.y() = body_position.z() - _param_camera_offset_z;
	camera_position.z() = body_position.x() - _param_camera_offset_x;

	return camera_position;
}

bool RingDetectorNode::projectBodyKalmanToImage(cv::Point2f &pixel, double &depth_z) const
{
	if (!_body_kf_initialized || _camera_matrix.empty())
	{
		return false;
	}

	const Eigen::Vector3d body_filtered = getBodyKalmanPosition();
	const Eigen::Vector3d camera_filtered = bodyXyzToCameraOptical(body_filtered);

	const double camera_x = camera_filtered.x();
	const double camera_y = camera_filtered.y();
	const double camera_z = camera_filtered.z();
	if (camera_z <= 1e-6)
	{
		return false;
	}

	const double fx = _camera_matrix.at<double>(0, 0);
	const double fy = _camera_matrix.at<double>(1, 1);
	const double cx0 = _camera_matrix.at<double>(0, 2);
	const double cy0 = _camera_matrix.at<double>(1, 2);

	pixel.x = static_cast<float>((camera_x * fx / camera_z) + cx0);
	pixel.y = static_cast<float>((camera_y * fy / camera_z) + cy0);
	depth_z = camera_z;
	return true;
}

void RingDetectorNode::publishCameraRawTarget(
	const std_msgs::msg::Header &header,
	double camera_x,
	double camera_y,
	double camera_z)
{
	geometry_msgs::msg::PoseStamped camera_raw_msg;
	camera_raw_msg.header = header;
	camera_raw_msg.header.frame_id = _target_frame_id;
	camera_raw_msg.pose.position.x = camera_x;
	camera_raw_msg.pose.position.y = camera_y;
	camera_raw_msg.pose.position.z = camera_z;
	camera_raw_msg.pose.orientation.w = 1.0;

	_target_pose_camera_raw_pub->publish(camera_raw_msg);
}

void RingDetectorNode::publishBodyFilteredTarget(const std_msgs::msg::Header &header)
{
	if (!_body_kf_initialized)
	{
		return;
	}

	const Eigen::Vector3d body_filtered = getBodyKalmanPosition();
	const Eigen::Vector3d body_velocity = getBodyKalmanVelocity();

	_last_body_filtered = body_filtered;
	_has_last_body_filtered = true;

	geometry_msgs::msg::PoseStamped body_filtered_msg;
	body_filtered_msg.header = header;
	body_filtered_msg.header.frame_id = _body_frame_id;
	body_filtered_msg.pose.position.x = body_filtered.x();
	body_filtered_msg.pose.position.y = body_filtered.y();
	body_filtered_msg.pose.position.z = body_filtered.z();
	body_filtered_msg.pose.orientation.w = 1.0;
	_target_error_body_filtered_pub->publish(body_filtered_msg);

	geometry_msgs::msg::TwistStamped body_velocity_msg;
	body_velocity_msg.header = header;
	body_velocity_msg.header.frame_id = _body_frame_id;
	body_velocity_msg.twist.linear.x = body_velocity.x();
	body_velocity_msg.twist.linear.y = body_velocity.y();
	body_velocity_msg.twist.linear.z = body_velocity.z();
	_target_velocity_body_filtered_pub->publish(body_velocity_msg);
}

void RingDetectorNode::publishBodyTargetFromMeasurement(
	const std_msgs::msg::Header &header,
	double camera_x,
	double camera_y,
	double camera_z)
{
	const rclcpp::Time stamp =
		(header.stamp.sec == 0 && header.stamp.nanosec == 0)
			? now()
			: rclcpp::Time(header.stamp);

	const Eigen::Vector3d camera_position(camera_x, camera_y, camera_z);
	const Eigen::Vector3d body_measurement = cameraOpticalToBodyXyz(camera_position);

	// Raw camera pose is kept for debug/diagnostic only.
	// Kalman duy nhất được cập nhật bằng vị trí đo sau khi đổi sang body/drone frame.
	publishCameraRawTarget(header, camera_x, camera_y, camera_z);

	if (!_body_kf_initialized)
	{
		initBodyKalman(body_measurement);
	}
	else
	{
		double dt_s = 1.0 / 30.0;
		if (_body_kf_last_time.nanoseconds() > 0)
		{
			dt_s = (stamp - _body_kf_last_time).seconds();
		}

		predictBodyKalman(dt_s);
		updateBodyKalman(body_measurement);
	}

	_body_kf_last_time = stamp;
	_last_measurement_time = stamp;

	publishBodyFilteredTarget(header);
}

bool RingDetectorNode::publishBodyTargetPredictionOnly(
	const std_msgs::msg::Header &header)
{
	if (!_body_kf_initialized)
	{
		return false;
	}

	const rclcpp::Time stamp =
		(header.stamp.sec == 0 && header.stamp.nanosec == 0)
			? now()
			: rclcpp::Time(header.stamp);

	double dt_s = 1.0 / 30.0;
	if (_body_kf_last_time.nanoseconds() > 0)
	{
		dt_s = (stamp - _body_kf_last_time).seconds();
	}

	predictBodyKalman(dt_s);
	_body_kf_last_time = stamp;
	publishBodyFilteredTarget(header);

	return true;
}

void RingDetectorNode::camera_info_callback(
	const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
	_camera_matrix = cv::Mat(3, 3, CV_64F, const_cast<double *>(msg->k.data())).clone();

	if (!msg->d.empty())
	{
		_dist_coeffs = cv::Mat(
						   static_cast<int>(msg->d.size()),
						   1,
						   CV_64F,
						   const_cast<double *>(msg->d.data()))
						   .clone();
	}
	else
	{
		_dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
	}

	_has_camera_info = true;
}

void RingDetectorNode::ring_detect_reset_callback(
	const std_msgs::msg::String::SharedPtr msg)
{
	if (!msg)
	{
		return;
	}

	std::string command = msg->data;
	std::transform(command.begin(), command.end(), command.begin(),
		[](unsigned char c) { return static_cast<char>(std::tolower(c)); });

	if (command == "reset" || command == "1" || command == "true" || command == "clear")
	{
		reset_lock_state();

		std_msgs::msg::Bool valid_msg;
		valid_msg.data = false;
		_target_valid_pub->publish(valid_msg);

		if (_reset_status_pub)
		{
			std_msgs::msg::String reset_msg;
			reset_msg.data = "RESET";
			_reset_status_pub->publish(reset_msg);
		}

		RCLCPP_WARN(
			get_logger(),
			"[RingDetector] received reset command from %s",
			_ring_detect_reset_topic.c_str());
	}
}

cv::Mat RingDetectorNode::build_neutral_gray_mask(const cv::Mat &bgr) const
{
	if (bgr.empty())
	{
		return cv::Mat();
	}

	const int gray_min = std::clamp(_param_gray_min, 0, 255);
	const int gray_max = std::clamp(_param_gray_max, gray_min, 255);
	cv::Mat mask;

	if (_param_use_yuv_neutral_mask)
	{
		// Cách này gần với code Python bạn gửi, nhưng thay vì bắt màu cam bằng U threshold,
		// ta bắt vật trung tính trắng/xám bằng điều kiện U,V gần 128.
		// Y giữ thông tin sáng/tối nên vòng trắng bị bóng đổ thành xám vẫn được nhận.
		cv::Mat yuv;
		cv::cvtColor(bgr, yuv, cv::COLOR_BGR2YUV);

		const int median_kernel = make_odd_kernel_size(_param_yuv_median_kernel);
		if (median_kernel >= 3)
		{
			cv::medianBlur(yuv, yuv, median_kernel);
		}

		std::vector<cv::Mat> yuv_channels;
		cv::split(yuv, yuv_channels);
		const cv::Mat &y = yuv_channels[0];
		const cv::Mat &u = yuv_channels[1];
		const cv::Mat &v = yuv_channels[2];

		cv::Mat y_min_mask;
		cv::Mat y_max_mask;
		cv::compare(y, gray_min, y_min_mask, cv::CMP_GE);
		cv::compare(y, gray_max, y_max_mask, cv::CMP_LE);
		cv::Mat brightness_mask = y_min_mask & y_max_mask;

		const int u_center = std::clamp(_param_yuv_u_center, 0, 255);
		const int v_center = std::clamp(_param_yuv_v_center, 0, 255);
		const int uv_diff_max = std::clamp(_param_yuv_uv_diff_max, 0, 127);

		cv::Mat u_diff;
		cv::Mat v_diff;
		cv::absdiff(u, cv::Scalar(u_center), u_diff);
		cv::absdiff(v, cv::Scalar(v_center), v_diff);

		cv::Mat u_mask;
		cv::Mat v_mask;
		cv::compare(u_diff, uv_diff_max, u_mask, cv::CMP_LE);
		cv::compare(v_diff, uv_diff_max, v_mask, cv::CMP_LE);

		mask = brightness_mask & u_mask & v_mask;
	}
	else
	{
		// Fallback BGR cũ: B/G/R gần nhau và độ sáng nằm trong ngưỡng.
		std::vector<cv::Mat> channels;
		cv::split(bgr, channels);

		const cv::Mat &blue = channels[0];
		const cv::Mat &green = channels[1];
		const cv::Mat &red = channels[2];
		const int color_diff_max = std::clamp(_param_gray_color_diff_max, 0, 255);

		cv::Mat gray;
		cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

		cv::Mat gray_min_mask;
		cv::Mat gray_max_mask;
		cv::compare(gray, gray_min, gray_min_mask, cv::CMP_GE);
		cv::compare(gray, gray_max, gray_max_mask, cv::CMP_LE);
		cv::Mat brightness_mask = gray_min_mask & gray_max_mask;

		cv::Mat max_bg;
		cv::Mat max_bgr;
		cv::Mat min_bg;
		cv::Mat min_bgr;
		cv::max(blue, green, max_bg);
		cv::max(max_bg, red, max_bgr);
		cv::min(blue, green, min_bg);
		cv::min(min_bg, red, min_bgr);

		cv::Mat diff;
		cv::subtract(max_bgr, min_bgr, diff);

		cv::Mat neutral_mask;
		cv::compare(diff, color_diff_max, neutral_mask, cv::CMP_LE);

		mask = brightness_mask & neutral_mask;
	}

	const int kernel_size = make_odd_kernel_size(_param_gray_morph_kernel);
	const cv::Mat kernel = cv::getStructuringElement(
		cv::MORPH_ELLIPSE,
		cv::Size(kernel_size, kernel_size));

	cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
	cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

	return mask;
}

cv::Mat RingDetectorNode::make_color_detect_debug_image(const cv::Mat &bgr_frame) const
{
	if (bgr_frame.empty())
	{
		return cv::Mat();
	}

	cv::Mat white_mask = build_neutral_gray_mask(bgr_frame);

	if (white_mask.empty())
	{
		return cv::Mat();
	}

	cv::Mat color_debug = cv::Mat::zeros(
		bgr_frame.size(),
		bgr_frame.type());

	bgr_frame.copyTo(color_debug, white_mask);

	cv::putText(
		color_debug,
		"BGR GRAY MASK DEBUG",
		cv::Point(20, 35),
		cv::FONT_HERSHEY_SIMPLEX,
		0.8,
		cv::Scalar(0, 255, 255),
		2,
		cv::LINE_AA);

	return color_debug;
}

cv::Mat RingDetectorNode::make_side_by_side_debug_image(
	const cv::Mat &left_image,
	const cv::Mat &right_image) const
{
	if (left_image.empty() && right_image.empty())
	{
		return cv::Mat();
	}

	if (left_image.empty())
	{
		return right_image.clone();
	}

	if (right_image.empty())
	{
		return left_image.clone();
	}

	cv::Mat left_bgr;
	cv::Mat right_bgr;

	if (left_image.channels() == 1)
	{
		cv::cvtColor(left_image, left_bgr, cv::COLOR_GRAY2BGR);
	}
	else
	{
		left_bgr = left_image.clone();
	}

	if (right_image.channels() == 1)
	{
		cv::cvtColor(right_image, right_bgr, cv::COLOR_GRAY2BGR);
	}
	else
	{
		right_bgr = right_image.clone();
	}

	if (right_bgr.size() != left_bgr.size())
	{
		cv::resize(right_bgr, right_bgr, left_bgr.size());
	}

	// Không dùng cv::hconcat để tránh lỗi size/type ở runtime.
	// Tự tạo canvas rồi copy từng ảnh vào ROI.
	cv::Mat combined(
		left_bgr.rows,
		left_bgr.cols + right_bgr.cols,
		CV_8UC3,
		cv::Scalar(0, 0, 0));

	left_bgr.copyTo(combined(cv::Rect(0, 0, left_bgr.cols, left_bgr.rows)));
	right_bgr.copyTo(combined(cv::Rect(left_bgr.cols, 0, right_bgr.cols, right_bgr.rows)));

	return combined;
}

cv::Mat RingDetectorNode::make_labeled_debug_panel(
	const cv::Mat &image,
	const std::string &label) const
{
	cv::Mat panel;

	if (image.empty())
	{
		panel = cv::Mat::zeros(cv::Size(640, 360), CV_8UC3);
	}
	else if (image.channels() == 1)
	{
		cv::cvtColor(image, panel, cv::COLOR_GRAY2BGR);
	}
	else if (image.channels() == 4)
	{
		cv::cvtColor(image, panel, cv::COLOR_BGRA2BGR);
	}
	else
	{
		panel = image.clone();
	}

	if (panel.depth() != CV_8U)
	{
		cv::Mat converted_panel;
		panel.convertTo(converted_panel, CV_8U);
		panel = converted_panel;
	}

	const int target_width = std::clamp(_param_debug_panel_width, 240, 960);
	const double scale = static_cast<double>(target_width) / static_cast<double>(std::max(1, panel.cols));
	const int target_height = std::max(1, static_cast<int>(std::round(static_cast<double>(panel.rows) * scale)));

	cv::resize(panel, panel, cv::Size(target_width, target_height));

	cv::rectangle(
		panel,
		cv::Point(0, 0),
		cv::Point(panel.cols, 34),
		cv::Scalar(0, 0, 0),
		cv::FILLED);

	cv::putText(
		panel,
		label,
		cv::Point(10, 24),
		cv::FONT_HERSHEY_SIMPLEX,
		0.65,
		cv::Scalar(0, 255, 255),
		2,
		cv::LINE_AA);

	return panel;
}

cv::Mat RingDetectorNode::make_pipeline_debug_image(
	const cv::Mat &annotated_frame) const
{
	// Ghép tất cả lớp debug vào đúng một topic /ring_detect/image_proc.
	// Không dùng hconcat/vconcat để tránh lỗi khác rows/type/channels khi xem bằng rqt.
	std::ostringstream overlay_label;
	overlay_label << "6 contour overlay raw=" << _debug_raw_contour_count
			  << " accepted=" << _debug_accepted_candidate_count;

	std::vector<cv::Mat> panels;
	panels.emplace_back(make_labeled_debug_panel(annotated_frame, "1 image_proc / selected target"));
	panels.emplace_back(make_labeled_debug_panel(_debug_y_channel, "2 Y channel"));
	panels.emplace_back(make_labeled_debug_panel(_debug_neutral_mask, "3 neutral mask"));
	panels.emplace_back(make_labeled_debug_panel(_debug_detector_binary, "4 detector input"));
	panels.emplace_back(make_labeled_debug_panel(_debug_edge, "5 Canny edge"));
	panels.emplace_back(make_labeled_debug_panel(_debug_candidate_overlay, overlay_label.str()));

	int cell_width = 1;
	int cell_height = 1;
	for (const auto &panel : panels)
	{
		cell_width = std::max(cell_width, panel.cols);
		cell_height = std::max(cell_height, panel.rows);
	}

	const int column_count = 2;
	const int row_count = static_cast<int>((panels.size() + column_count - 1) / column_count);
	cv::Mat canvas = cv::Mat::zeros(
		row_count * cell_height,
		column_count * cell_width,
		CV_8UC3);

	for (size_t i = 0; i < panels.size(); ++i)
	{
		cv::Mat panel = panels[i];
		if (panel.empty())
		{
			continue;
		}

		if (panel.channels() == 1)
		{
			cv::cvtColor(panel, panel, cv::COLOR_GRAY2BGR);
		}
		else if (panel.channels() == 4)
		{
			cv::cvtColor(panel, panel, cv::COLOR_BGRA2BGR);
		}

		if (panel.depth() != CV_8U)
		{
			cv::Mat converted;
			panel.convertTo(converted, CV_8U);
			panel = converted;
		}

		const int row = static_cast<int>(i) / column_count;
		const int col = static_cast<int>(i) % column_count;
		const int x = col * cell_width;
		const int y = row * cell_height;

		const cv::Rect dst_roi(x, y, panel.cols, panel.rows);
		panel.copyTo(canvas(dst_roi));
	}

	return canvas;
}

std::vector<RingDetectorNode::RingCandidate>
RingDetectorNode::detect_ring_candidates(const cv::Mat &frame, cv::Mat &edges)
{
	std::vector<RingCandidate> candidates;

	if (frame.empty())
	{
		return candidates;
	}

	_debug_neutral_mask.release();
	_debug_y_channel.release();
	_debug_detector_binary.release();
	_debug_edge.release();
	_debug_candidate_overlay = frame.clone();
	_debug_raw_contour_count = 0;
	_debug_accepted_candidate_count = 0;

	const int image_width = frame.cols;
	const int image_height = frame.rows;

	cv::Point2f image_center(
		static_cast<float>(image_width) * 0.5f,
		static_cast<float>(image_height) * 0.5f);

	if (!_camera_matrix.empty())
	{
		image_center.x = static_cast<float>(_camera_matrix.at<double>(0, 2));
		image_center.y = static_cast<float>(_camera_matrix.at<double>(1, 2));
	}

	const double roi_ratio = std::clamp(_param_center_roi_ratio, 0.20, 1.00);
	const int roi_width = std::max(32, static_cast<int>(static_cast<double>(image_width) * roi_ratio));
	const int roi_height = std::max(32, static_cast<int>(static_cast<double>(image_height) * roi_ratio));

	int roi_x = static_cast<int>(std::round(image_center.x - static_cast<float>(roi_width) * 0.5f));
	int roi_y = static_cast<int>(std::round(image_center.y - static_cast<float>(roi_height) * 0.5f));
	roi_x = std::clamp(roi_x, 0, std::max(0, image_width - roi_width));
	roi_y = std::clamp(roi_y, 0, std::max(0, image_height - roi_height));

	const cv::Rect roi_rect(roi_x, roi_y, roi_width, roi_height);
	const cv::Mat roi = frame(roi_rect);

	if (!_debug_candidate_overlay.empty())
	{
		cv::rectangle(
			_debug_candidate_overlay,
			roi_rect,
			cv::Scalar(255, 255, 0),
			2);
	}

	// Bước 1: giữ lại vùng màu trung tính từ trắng -> xám -> xám đen.
	// Đây là phần thay cho threshold gray đơn giản trong code Python tham khảo,
	// vì vòng trắng ngoài thực tế có thể bị bóng đổ thành xám tối.
	cv::Mat neutral_mask_roi = build_neutral_gray_mask(roi);

	edges = cv::Mat::zeros(frame.size(), CV_8UC1);
	_debug_neutral_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
	if (!neutral_mask_roi.empty())
	{
		neutral_mask_roi.copyTo(_debug_neutral_mask(roi_rect));
	}

	if (neutral_mask_roi.empty())
	{
		return candidates;
	}

	// Bước 2: tạo ảnh input cho Canny.
	// Không dùng trực tiếp neutral_mask_roi dạng 0/255, vì nền mô phỏng cũng màu xám.
	// Nếu Canny chạy trên mask nhị phân thì cạnh giữa vòng xám đen và nền xám sẽ biến mất.
	// Cách này giữ lại cường độ thật bên trong vùng màu trung tính, nên vẫn thấy cạnh vòng.
	cv::Mat gray_roi;
	if (_param_use_yuv_neutral_mask)
	{
		cv::Mat yuv_roi;
		cv::cvtColor(roi, yuv_roi, cv::COLOR_BGR2YUV);

		const int median_kernel = make_odd_kernel_size(_param_yuv_median_kernel);
		if (median_kernel >= 3)
		{
			cv::medianBlur(yuv_roi, yuv_roi, median_kernel);
		}

		std::vector<cv::Mat> yuv_channels;
		cv::split(yuv_roi, yuv_channels);
		gray_roi = yuv_channels[0];
	}
	else
	{
		cv::cvtColor(roi, gray_roi, cv::COLOR_BGR2GRAY);
	}

	_debug_y_channel = cv::Mat::zeros(frame.size(), CV_8UC1);
	gray_roi.copyTo(_debug_y_channel(roi_rect));

	cv::Mat detector_binary_roi = cv::Mat::zeros(gray_roi.size(), CV_8UC1);
	gray_roi.copyTo(detector_binary_roi, neutral_mask_roi);

	if (_param_use_gray_binary_threshold)
	{
		cv::Mat threshold_roi;
		const int gray_threshold = std::clamp(_param_gray_binary_threshold, 0, 255);
		cv::threshold(detector_binary_roi, threshold_roi, gray_threshold, 255, cv::THRESH_BINARY);
		detector_binary_roi = threshold_roi;
	}

	_debug_detector_binary = cv::Mat::zeros(frame.size(), CV_8UC1);
	detector_binary_roi.copyTo(_debug_detector_binary(roi_rect));

	cv::Mat edge_roi;

	if (_param_use_canny_circle_detector)
	{
		cv::Mat edge_input_roi;
		if (_param_use_bilateral_filter)
		{
			const int bilateral_d = std::clamp(_param_bilateral_d, 1, 15);
			cv::bilateralFilter(
				detector_binary_roi,
				edge_input_roi,
				bilateral_d,
				std::max(1.0, _param_bilateral_sigma_color),
				std::max(1.0, _param_bilateral_sigma_space));
		}
		else
		{
			edge_input_roi = detector_binary_roi;
		}

		// Bước 3: Canny giống code tham khảo.
		const double canny_low = std::max(0.0, _param_canny_low);
		const double canny_high = std::max(canny_low + 1.0, _param_canny_high);
		cv::Canny(edge_input_roi, edge_roi, canny_low, canny_high);

		const int edge_kernel_size = make_odd_kernel_size(_param_edge_dilate_kernel);
		if (edge_kernel_size >= 3)
		{
			const cv::Mat edge_kernel = cv::getStructuringElement(
				cv::MORPH_ELLIPSE,
				cv::Size(edge_kernel_size, edge_kernel_size));

			// Dilation nhẹ để nối các cạnh bị đứt do bóng đổ hoặc nén H264.
			cv::dilate(edge_roi, edge_roi, edge_kernel);
		}
	}
	else
	{
		// Fallback: tìm contour trực tiếp trên mask màu trung tính, nhẹ hơn nhưng kém phân biệt cạnh trong/ngoài.
		edge_roi = detector_binary_roi;
	}

	edge_roi.copyTo(edges(roi_rect));
	_debug_edge = edges.clone();

	std::vector<std::vector<cv::Point>> contours;
	// RETR_TREE giống code mẫu: vòng tròn/ring thường có biên ngoài và biên trong.
	// CHAIN_APPROX_NONE giữ đủ điểm biên để tính radial_error ổn hơn.
	cv::findContours(edge_roi, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
	_debug_raw_contour_count = static_cast<int>(contours.size());

	const double max_center_distance =
		static_cast<double>(std::min(image_width, image_height)) *
		std::clamp(_param_max_center_distance_ratio, 0.05, 0.70);

	for (const auto &contour : contours)
	{
		const std::vector<cv::Point> contour_full_for_debug =
			offset_contour(contour, cv::Point(roi_x, roi_y));

		if (_param_debug_draw_rejected_contours && !_debug_candidate_overlay.empty())
		{
			cv::drawContours(
				_debug_candidate_overlay,
				std::vector<std::vector<cv::Point>>{contour_full_for_debug},
				-1,
				cv::Scalar(0, 0, 255),
				1);
		}

		if (contour.size() < 8)
		{
			continue;
		}

		const double area = std::abs(cv::contourArea(contour));
		if (area < _param_min_area)
		{
			continue;
		}

		const double perimeter = cv::arcLength(contour, true);
		if (perimeter < 1e-6)
		{
			continue;
		}

		std::vector<cv::Point> approx;
		const double epsilon_ratio = std::clamp(_param_approx_epsilon_ratio, 0.005, 0.080);
		cv::approxPolyDP(contour, approx, perimeter * epsilon_ratio, true);

		// Đây là ý chính từ code Python tham khảo:
		// hình tròn sau approx thường có nhiều đỉnh; hình vuông thường còn 4 đỉnh.
		if (static_cast<int>(approx.size()) < _param_min_approx_vertices)
		{
			continue;
		}

		cv::Point2f center_roi;
		float radius = 0.0f;
		cv::minEnclosingCircle(contour, center_roi, radius);

		if (radius < static_cast<float>(_param_min_radius_px))
		{
			continue;
		}

		const cv::Rect bbox = cv::boundingRect(contour);
		if (bbox.width <= 0 || bbox.height <= 0)
		{
			continue;
		}

		const double aspect = static_cast<double>(bbox.width) / static_cast<double>(bbox.height);
		if (aspect < _param_aspect_min || aspect > _param_aspect_max)
		{
			continue;
		}

		const double circularity = 4.0 * kPi * area / (perimeter * perimeter);
		if (circularity < _param_circularity_min)
		{
			continue;
		}

		const double circle_area = kPi * static_cast<double>(radius) * static_cast<double>(radius);
		const double circle_area_ratio = area / std::max(1.0, circle_area);

		// Khi dùng Canny, contour của vòng có thể là viền mỏng hoặc biên trong/ngoài,
		// nên area/circle_area thường nhỏ hơn vòng tròn đặc. Nếu bật filter này quá chặt,
		// vòng to trong mô phỏng sẽ dễ bị loại dù nhìn bằng mắt rất rõ.
		if (_param_use_circle_area_ratio_filter)
		{
			if (circle_area_ratio < _param_min_circle_area_ratio ||
				circle_area_ratio > _param_max_circle_area_ratio)
			{
				continue;
			}

			const double fill_ratio = circle_area_ratio;
			if (fill_ratio < _param_min_fill_ratio || fill_ratio > _param_max_fill_ratio)
			{
				continue;
			}
		}

		const double radial_error = calc_mean_radial_error(contour, center_roi, radius);
		if (radial_error > _param_max_radial_error)
		{
			continue;
		}

		// Fit ellipse giúp loại các contour méo/đa giác dài.
		// Hình tròn thật có major/minor gần nhau; hình vuông hoặc cạnh gãy thường lệch hơn.
		if (contour.size() >= 5)
		{
			const cv::RotatedRect ellipse = cv::fitEllipse(contour);
			const double major_axis = std::max(ellipse.size.width, ellipse.size.height);
			const double minor_axis = std::min(ellipse.size.width, ellipse.size.height);

			if (major_axis < 1e-6)
			{
				continue;
			}

			const double ellipse_axis_ratio = minor_axis / major_axis;
			if (ellipse_axis_ratio < _param_min_ellipse_axis_ratio)
			{
				continue;
			}

			const double center_shift_ratio =
				calc_distance(ellipse.center, center_roi) / std::max(1.0f, radius);
			if (center_shift_ratio > _param_max_ellipse_center_shift_ratio)
			{
				continue;
			}
		}

		const cv::Point2f center_full(
			center_roi.x + static_cast<float>(roi_x),
			center_roi.y + static_cast<float>(roi_y));

		if (calc_distance(center_full, image_center) > max_center_distance)
		{
			continue;
		}

		const float visible_size = 2.0f * radius;
		if (visible_size < static_cast<float>(_param_min_visible_size))
		{
			continue;
		}

		RingCandidate candidate;
		candidate.center = center_full;
		candidate.radius = radius;
		candidate.area = static_cast<float>(area);
		candidate.circularity = static_cast<float>(circularity);
		candidate.bbox_w = static_cast<float>(bbox.width);
		candidate.bbox_h = static_cast<float>(bbox.height);
		candidate.bbox_area = static_cast<float>(bbox.width * bbox.height);
		candidate.visible_size = visible_size;
		candidate.circle_area_ratio = static_cast<float>(circle_area_ratio);
		candidate.radial_error = static_cast<float>(radial_error);
		candidate.approx_vertices = static_cast<int>(approx.size());
		candidate.contour = contour_full_for_debug;

		candidates.push_back(candidate);
		_debug_accepted_candidate_count = static_cast<int>(candidates.size());

		if (!_debug_candidate_overlay.empty())
		{
			cv::drawContours(
				_debug_candidate_overlay,
				std::vector<std::vector<cv::Point>>{candidate.contour},
				-1,
				cv::Scalar(0, 255, 0),
				2);

			cv::circle(
				_debug_candidate_overlay,
				candidate.center,
				static_cast<int>(std::round(candidate.radius)),
				cv::Scalar(0, 255, 255),
				2);

			std::ostringstream candidate_text;
			candidate_text << "r=" << std::fixed << std::setprecision(0) << candidate.radius
						   << " v=" << candidate.approx_vertices
						   << " re=" << std::setprecision(2) << candidate.radial_error;

			cv::putText(
				_debug_candidate_overlay,
				candidate_text.str(),
				cv::Point(
					static_cast<int>(std::round(candidate.center.x + 8.0f)),
					static_cast<int>(std::round(candidate.center.y - 8.0f))),
				cv::FONT_HERSHEY_SIMPLEX,
				0.5,
				cv::Scalar(0, 255, 255),
				1,
				cv::LINE_AA);
		}
	}

	if (!_debug_candidate_overlay.empty())
	{
		std::ostringstream debug_text;
		debug_text << "raw contours=" << _debug_raw_contour_count
				   << " accepted=" << _debug_accepted_candidate_count
				   << " roi=" << std::fixed << std::setprecision(2) << roi_ratio
				   << " grayMin=" << _param_gray_min
				   << " areaRatioFilter=" << (_param_use_circle_area_ratio_filter ? "on" : "off");

		cv::putText(
			_debug_candidate_overlay,
			debug_text.str(),
			cv::Point(20, 35),
			cv::FONT_HERSHEY_SIMPLEX,
			0.65,
			cv::Scalar(0, 255, 255),
			2,
			cv::LINE_AA);
	}

	return candidates;
}

std::vector<RingDetectorNode::RingCandidate>
RingDetectorNode::detect_ring_candidates_far(const cv::Mat &, cv::Mat &edges)
{
	// Bản tối ưu không dùng far detector riêng để tránh xử lý 2 lần mỗi frame.
	edges.release();
	return {};
}

bool RingDetectorNode::select_main_ring(
	const std::vector<RingCandidate> &candidates,
	RingCandidate &best) const
{
	if (candidates.empty())
	{
		return false;
	}

	cv::Point2f image_center(640.0f, 480.0f);
	if (!_camera_matrix.empty())
	{
		image_center.x = static_cast<float>(_camera_matrix.at<double>(0, 2));
		image_center.y = static_cast<float>(_camera_matrix.at<double>(1, 2));
	}

	bool found = false;
	double best_score = -1e18;

	for (const auto &candidate : candidates)
	{
		const double center_distance = calc_distance(candidate.center, image_center);

		// Mục tiêu chính: vòng tròn trung tính gần tâm nhất.
		// Ưu tiên gần tâm, sau đó mới xét độ tròn và kích thước.
		const double circle_score =
			120.0 * static_cast<double>(candidate.circularity) +
			80.0 * static_cast<double>(candidate.circle_area_ratio) -
			120.0 * static_cast<double>(candidate.radial_error);

		const double score =
			-1.00 * center_distance +
			0.15 * static_cast<double>(candidate.visible_size) +
			circle_score;

		if (!found || score > best_score)
		{
			best = candidate;
			best_score = score;
			found = true;
		}
	}

	return found;
}

bool RingDetectorNode::select_far_ring(
	const std::vector<RingCandidate> &candidates,
	RingCandidate &best) const
{
	return select_main_ring(candidates, best);
}

bool RingDetectorNode::find_locked_ring(
	const std::vector<RingCandidate> &candidates,
	RingCandidate &best) const
{
	if (!_locked || candidates.empty())
	{
		return false;
	}

	cv::Point2f kf_expected_center = _lock_center;
	double kf_expected_depth = -1.0;
	const bool has_kf_projection =
		_param_track_use_kf_gate && projectBodyKalmanToImage(kf_expected_center, kf_expected_depth);

	bool found = false;
	double best_score = -1e18;

	for (const auto &candidate : candidates)
	{
		const double lock_distance = calc_distance(candidate.center, _lock_center);
		if (lock_distance > _param_lock_max_dist)
		{
			continue;
		}

		if (has_kf_projection)
		{
			const double kf_pixel_distance = calc_distance(candidate.center, kf_expected_center);
			if (kf_pixel_distance > _param_track_kf_pixel_gate_px)
			{
				continue;
			}

			double candidate_camera_x = 0.0;
			double candidate_camera_y = 0.0;
			double candidate_camera_z = 0.0;
			estimate_ring_pose_from_image(
				candidate,
				0,
				0,
				candidate_camera_x,
				candidate_camera_y,
				candidate_camera_z);

			if (kf_expected_depth > 1e-6)
			{
				const double depth_ratio_error =
					std::abs(candidate_camera_z - kf_expected_depth) / kf_expected_depth;
				if (depth_ratio_error > _param_track_kf_depth_ratio_gate)
				{
					continue;
				}
			}
		}

		if (_lock_visible_size > 1e-6f &&
			candidate.visible_size < static_cast<float>(_lock_visible_size * _param_lock_min_size_ratio))
		{
			continue;
		}

		const double size_ratio =
			static_cast<double>(candidate.visible_size) / std::max(1.0f, _lock_visible_size);
		const double kf_distance = has_kf_projection
			? calc_distance(candidate.center, kf_expected_center)
			: lock_distance;

		const double score =
			-1.25 * kf_distance -
			0.55 * lock_distance -
			std::abs(1.0 - size_ratio) * 60.0 +
			80.0 * static_cast<double>(candidate.circularity) +
			50.0 * static_cast<double>(candidate.circle_area_ratio) -
			80.0 * static_cast<double>(candidate.radial_error);

		if (!found || score > best_score)
		{
			best = candidate;
			best_score = score;
			found = true;
		}
	}

	return found;
}

bool RingDetectorNode::confirm_far_candidate(const RingCandidate &candidate)
{
	if (_param_far_confirm_frames <= 1)
	{
		return true;
	}

	if (!_has_far_candidate_memory)
	{
		_has_far_candidate_memory = true;
		_far_candidate_center = candidate.center;
		_far_candidate_size = candidate.visible_size;
		_far_candidate_count = 1;
		return (_far_candidate_count >= _param_far_confirm_frames);
	}

	const double dist = calc_distance(candidate.center, _far_candidate_center);
	const double size_ref = std::max(1.0f, _far_candidate_size);
	const double size_ratio_err =
		std::abs(static_cast<double>(candidate.visible_size) - size_ref) / size_ref;

	if (dist <= _param_far_confirm_pos_gate_px &&
		size_ratio_err <= _param_far_confirm_size_ratio)
	{
		_far_candidate_count++;
	}
	else
	{
		_far_candidate_count = 1;
	}

	_far_candidate_center = candidate.center;
	_far_candidate_size = candidate.visible_size;

	return (_far_candidate_count >= _param_far_confirm_frames);
}

void RingDetectorNode::estimate_ring_pose_from_image(
	const RingCandidate &target,
	int /*image_width*/,
	int /*image_height*/,
	double &x,
	double &y,
	double &z) const
{
	const double fx = _camera_matrix.at<double>(0, 0);
	const double fy = _camera_matrix.at<double>(1, 1);
	const double cx0 = _camera_matrix.at<double>(0, 2);
	const double cy0 = _camera_matrix.at<double>(1, 2);

	const double cx = static_cast<double>(target.center.x);
	const double cy = static_cast<double>(target.center.y);
	const double diameter_px = std::max(2.0 * static_cast<double>(target.radius), 1e-6);

	z = fx * _param_ring_diameter_m / diameter_px;
	x = (cx - cx0) * z / fx;
	y = (cy - cy0) * z / fy;
}

bool RingDetectorNode::is_passed_ring() const
{
	if (!_has_last_body_filtered)
	{
		return false;
	}

	if (_last_body_filtered.x() > 0.0 &&
		_last_body_filtered.x() < _param_pass_x_thresh)
	{
		return true;
	}

	if (_has_last_good_target &&
		_last_good_target.visible_size >= static_cast<float>(_param_pass_min_visible_size))
	{
		return true;
	}

	return false;
}

bool RingDetectorNode::can_hold_with_body_kalman(const rclcpp::Time &now_ts) const
{
	if (!_body_kf_initialized || _last_measurement_time.nanoseconds() == 0)
	{
		return false;
	}

	const double lost_time_s = (now_ts - _last_measurement_time).seconds();

	return lost_time_s <= _param_body_kf_hold_timeout_s &&
		   _lock_missed <= _param_max_lock_missed &&
		   _hold_missed <= _param_hold_max_missed;
}

void RingDetectorNode::reset_lock_state()
{
	_locked = false;
	_lock_center = cv::Point2f(0.0f, 0.0f);
	_lock_visible_size = 0.0f;
	_lock_missed = 0;
	_hold_missed = 0;

	_has_last_good_target = false;
	_last_camera_x = 0.0;
	_last_camera_y = 0.0;
	_last_camera_z = 0.0;
	_last_body_filtered = Eigen::Vector3d::Zero();
	_has_last_body_filtered = false;

	_has_far_candidate_memory = false;
	_far_candidate_center = cv::Point2f(0.0f, 0.0f);
	_far_candidate_size = 0.0f;
	_far_candidate_count = 0;

	resetBodyKalman();
}

void RingDetectorNode::annotate_image(
	cv_bridge::CvImagePtr image,
	const RingCandidate &target,
	double /*camera_x*/,
	double /*camera_y*/,
	double /*camera_z*/,
	const std::string &state_text) const
{
	auto &frame = image->image;

	const int w = frame.cols;
	const int h = frame.rows;

	const cv::Point image_center(w / 2, h / 2);
	const cv::Point ring_center(
		static_cast<int>(std::round(target.center.x)),
		static_cast<int>(std::round(target.center.y)));

	if (!target.contour.empty())
	{
		cv::drawContours(
			frame,
			std::vector<std::vector<cv::Point>>{target.contour},
			-1,
			cv::Scalar(0, 255, 0),
			2);
	}

	cv::circle(frame, ring_center, 4, cv::Scalar(0, 0, 255), -1);
	cv::circle(
		frame,
		ring_center,
		static_cast<int>(std::round(target.radius)),
		cv::Scalar(255, 255, 0),
		2);

	cv::circle(frame, image_center, 4, cv::Scalar(255, 0, 0), -1);
	cv::line(frame, image_center, ring_center, cv::Scalar(0, 255, 255), 2);

	std::ostringstream line1;
	std::ostringstream line2;
	std::ostringstream line3;

	line1 << std::fixed << std::setprecision(1)
		  << "cx=" << target.center.x
		  << " cy=" << target.center.y
		  << " r=" << target.radius
		  << " circ=" << std::setprecision(2) << target.circularity;

	line2 << std::fixed << std::setprecision(2)
		  << "circleRatio=" << target.circle_area_ratio
		  << " radialErr=" << target.radial_error
		  << " vtx=" << target.approx_vertices;

	if (_has_last_body_filtered)
	{
		line3 << std::fixed << std::setprecision(3)
			  << "body filtered x=" << _last_body_filtered.x()
			  << " y=" << _last_body_filtered.y()
			  << " z=" << _last_body_filtered.z();
	}
	else
	{
		line3 << "body filtered unavailable";
	}

	cv::putText(frame, state_text, cv::Point(20, 35),
				cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

	cv::putText(frame, line1.str(), cv::Point(20, 70),
				cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

	cv::putText(frame, line2.str(), cv::Point(20, 105),
				cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

	cv::putText(frame, line3.str(), cv::Point(20, 140),
				cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
}

void RingDetectorNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
	try
	{
		cv_bridge::CvImagePtr cv_ptr =
			cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

		cv::Mat color_debug_source;
		if (_param_publish_color_detect_debug)
		{
			color_debug_source = cv_ptr->image.clone();
		}

		cv::Mat mask_debug;
		const auto candidates = detect_ring_candidates(cv_ptr->image, mask_debug);

		std_msgs::msg::Bool valid_msg;
		std_msgs::msg::String reset_msg;
		valid_msg.data = false;
		reset_msg.data = "ACTIVE";

		const rclcpp::Time now_ts =
			(msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0)
				? now()
				: rclcpp::Time(msg->header.stamp);

		if (!_has_camera_info)
		{
			cv::putText(
				cv_ptr->image,
				"WAITING CAMERA INFO",
				cv::Point(20, 35),
				cv::FONT_HERSHEY_SIMPLEX,
				0.8,
				cv::Scalar(0, 255, 255),
				2,
				cv::LINE_AA);
		}
		else
		{
			RingCandidate target;
			bool found = false;

			if (_locked)
			{
				// Khi đã lock một vòng, chỉ tìm quanh vị trí dự đoán từ Single Body Kalman đã chiếu ngược về ảnh.
				// Không fallback sang select_main_ring để tránh nhảy sang vòng kế bên.
				found = find_locked_ring(candidates, target);
			}
			else
			{
				found = select_main_ring(candidates, target);
				if (found)
				{
					found = confirm_far_candidate(target);
				}
			}

			if (found)
			{
				double camera_x = 0.0;
				double camera_y = 0.0;
				double camera_z = 0.0;

				estimate_ring_pose_from_image(
					target,
					cv_ptr->image.cols,
					cv_ptr->image.rows,
					camera_x,
					camera_y,
					camera_z);

				_locked = true;
				_lock_center = target.center;
				_lock_visible_size = target.visible_size;
				_lock_missed = 0;
				_hold_missed = 0;

				_last_good_target = target;
				_has_last_good_target = true;

				_last_camera_x = camera_x;
				_last_camera_y = camera_y;
				_last_camera_z = camera_z;

				publishBodyTargetFromMeasurement(
					msg->header,
					camera_x,
					camera_y,
					camera_z);

				valid_msg.data = true;
				reset_msg.data = "ACTIVE";

				annotate_image(
					cv_ptr,
					target,
					camera_x,
					camera_y,
					camera_z,
					"RING DETECTED");
			}
			else
			{
				_lock_missed++;
				_hold_missed++;

				if (can_hold_with_body_kalman(now_ts) &&
					publishBodyTargetPredictionOnly(msg->header))
				{
					valid_msg.data = true;
					reset_msg.data = "ACTIVE";

					RingCandidate pred_target = _has_last_good_target ? _last_good_target : RingCandidate{};

					annotate_image(
						cv_ptr,
						pred_target,
						_last_camera_x,
						_last_camera_y,
						_last_camera_z,
						"SINGLE BODY KF HOLD");
				}
				else
				{
					const bool passed = is_passed_ring();

					if (passed)
					{
						cv::putText(
							cv_ptr->image,
							"PASSED RING",
							cv::Point(20, 35),
							cv::FONT_HERSHEY_SIMPLEX,
							0.8,
							cv::Scalar(0, 255, 0),
							2,
							cv::LINE_AA);

						reset_msg.data = "RESET";
						valid_msg.data = false;
						reset_lock_state();
					}
					else
					{
						cv::putText(
							cv_ptr->image,
							"RING LOST / SEARCHING",
							cv::Point(20, 35),
							cv::FONT_HERSHEY_SIMPLEX,
							0.8,
							cv::Scalar(0, 0, 255),
							2,
							cv::LINE_AA);

						const bool reset_by_missed =
							_lock_missed > _param_max_lock_missed ||
							_hold_missed > _param_hold_max_missed;

						const bool reset_by_time =
							_last_measurement_time.nanoseconds() > 0 &&
							(now_ts - _last_measurement_time).seconds() > _param_reset_timeout_s;

						if (reset_by_missed || reset_by_time)
						{
							reset_msg.data = "RESET";
							reset_lock_state();
						}
					}
				}
			}
		}

		_target_valid_pub->publish(valid_msg);
		if (_reset_status_pub)
		{
			_reset_status_pub->publish(reset_msg);
		}

		if (_param_publish_image)
		{
			cv::Mat final_debug_image;

			if (_param_publish_pipeline_debug)
			{
				final_debug_image = make_pipeline_debug_image(cv_ptr->image);
			}
			else
			{
				cv::Mat color_debug_image;
				if (_param_publish_color_detect_debug)
				{
					color_debug_image = make_color_detect_debug_image(color_debug_source);
				}

				final_debug_image = make_side_by_side_debug_image(
					cv_ptr->image,
					color_debug_image);
			}

			cv_bridge::CvImage out_msg;
			out_msg.header = msg->header;
			out_msg.encoding = sensor_msgs::image_encodings::BGR8;
			out_msg.image = final_debug_image.empty() ? cv_ptr->image : final_debug_image;

			_image_pub->publish(*out_msg.toImageMsg());
		}
	}
	catch (const cv_bridge::Exception &error)
	{
		RCLCPP_ERROR_THROTTLE(
			get_logger(),
			*get_clock(),
			2000,
			"cv_bridge error in image_callback: %s",
			error.what());
	}
	catch (const std::exception &error)
	{
		RCLCPP_ERROR_THROTTLE(
			get_logger(),
			*get_clock(),
			2000,
			"std::exception in image_callback: %s",
			error.what());
	}
}

int main(int argc, char **argv)
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<RingDetectorNode>());
	rclcpp::shutdown();
	return 0;
}