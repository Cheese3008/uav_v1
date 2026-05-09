#include "RingDetector.hpp"

#include <algorithm>
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

	// ============================================================
	// Complete-circle-only logic ported from CanMV K230 version.
	// Bản này đã nới ngưỡng so với bản strict cũ để dễ bắt vòng thật hơn;
	// các giá trị cũ được giữ bằng comment OLD STRICT ngay phía dưới.
	// These helpers are intentionally kept outside RingDetectorNode so the existing
	// RingDetector.hpp does not need new method declarations.
	// ============================================================
	constexpr int kCircleCheckSamples = 48;
	constexpr int kCircleSearchDr = 4;
	constexpr int kCircleEdgeOffset = 3;
	// OLD STRICT: constexpr double kCircleMinEdgeStrength = 18.0;
	constexpr double kCircleMinEdgeStrength = 8.0;
	// OLD STRICT: constexpr double kCircleMinSupportRatio = 0.80;
	constexpr double kCircleMinSupportRatio = 0.58;
	// OLD STRICT: constexpr int kCircleMaxGapSamples = 6;
	constexpr int kCircleMaxGapSamples = 16;
	// OLD STRICT: constexpr double kCircleMaxRadiusStd = 2.8;
	constexpr double kCircleMaxRadiusStd = 10.0;
	constexpr int kCompleteCircleHardEdgeMarginPx = 2;
	// OLD STRICT: constexpr double kCompleteCircleMinConfidence = 0.40;
	constexpr double kCompleteCircleMinConfidence = 0.20;

	constexpr double kNearHoughDp = 1.2;
	constexpr double kNearHoughMinDist = 20.0;
	constexpr double kNearHoughParam1 = 120.0;
	// OLD STRICT: constexpr double kNearHoughParam2 = 34.0;
	constexpr double kNearHoughParam2 = 18.0;

	constexpr double kFarHoughDp = 1.2;
	constexpr double kFarHoughMinDist = 16.0;
	constexpr double kFarHoughParam1 = 100.0;
	// OLD STRICT: constexpr double kFarHoughParam2 = 26.0;
	constexpr double kFarHoughParam2 = 14.0;

	struct CompleteCircleStats
	{
		double support_ratio = 0.0;
		int max_gap = 999;
		double radius_std = 999.0;
		double edge_mean = 0.0;
		double confidence = 0.0;
	};

	struct CompleteCircleCandidateData
	{
		cv::Point2f center;
		float radius = 0.0f;
		CompleteCircleStats stats;
	};

	double get_complete_circle_metric_support(const float area_field)
	{
		return static_cast<double>(area_field);
	}

	int get_complete_circle_metric_gap(const float bbox_w_field)
	{
		return static_cast<int>(std::round(static_cast<double>(bbox_w_field)));
	}

	double get_complete_circle_metric_radius_std(const float bbox_h_field)
	{
		return static_cast<double>(bbox_h_field);
	}

	double get_complete_circle_metric_confidence(const float circularity_field)
	{
		return static_cast<double>(circularity_field);
	}

	uchar get_gray_value_safe(const cv::Mat &gray, double x, double y)
	{
		const int xi = static_cast<int>(std::round(x));
		const int yi = static_cast<int>(std::round(y));

		if (xi < 0 || yi < 0 || xi >= gray.cols || yi >= gray.rows)
		{
			return 0;
		}

		return gray.at<uchar>(yi, xi);
	}

	struct EdgeStrengthResult
	{
		double strength = -1.0;
		double radius = 0.0;
	};

	EdgeStrengthResult estimate_edge_strength_on_radius(
		const cv::Mat &gray,
		double cx,
		double cy,
		double radius,
		double theta)
	{
		EdgeStrengthResult result;
		result.strength = -1.0;
		result.radius = radius;

		const double cos_theta = std::cos(theta);
		const double sin_theta = std::sin(theta);

		for (int dr = -kCircleSearchDr; dr <= kCircleSearchDr; ++dr)
		{
			const double rr = radius + static_cast<double>(dr);
			const double rin = rr - static_cast<double>(kCircleEdgeOffset);
			const double rout = rr + static_cast<double>(kCircleEdgeOffset);

			const double xi = cx + rin * cos_theta;
			const double yi = cy + rin * sin_theta;

			const double xm = cx + rr * cos_theta;
			const double ym = cy + rr * sin_theta;

			const double xo = cx + rout * cos_theta;
			const double yo = cy + rout * sin_theta;

			const double gi = static_cast<double>(get_gray_value_safe(gray, xi, yi));
			const double gm = static_cast<double>(get_gray_value_safe(gray, xm, ym));
			const double go = static_cast<double>(get_gray_value_safe(gray, xo, yo));

			// Same geometric edge test as the K230 port:
			// the pixel at the expected circumference must differ from inner/outer samples.
			const double strength = std::abs(gm - 0.5 * (gi + go)) + 0.35 * std::abs(go - gi);

			if (strength > result.strength)
			{
				result.strength = strength;
				result.radius = rr;
			}
		}

		return result;
	}

	bool validate_complete_circle_candidate(
		const cv::Mat &gray,
		const cv::Point2f &center,
		double radius,
		CompleteCircleStats &stats)
	{
		if (gray.empty())
		{
			return false;
		}

		if (center.x - radius < kCompleteCircleHardEdgeMarginPx ||
			center.y - radius < kCompleteCircleHardEdgeMarginPx ||
			center.x + radius > gray.cols - kCompleteCircleHardEdgeMarginPx ||
			center.y + radius > gray.rows - kCompleteCircleHardEdgeMarginPx)
		{
			return false;
		}

		std::vector<int> supports(kCircleCheckSamples, 0);
		std::vector<double> edge_strengths;
		std::vector<double> best_radii;
		edge_strengths.reserve(kCircleCheckSamples);
		best_radii.reserve(kCircleCheckSamples);

		for (int i = 0; i < kCircleCheckSamples; ++i)
		{
			const double theta = 2.0 * kPi * static_cast<double>(i) / static_cast<double>(kCircleCheckSamples);
			const EdgeStrengthResult edge_result = estimate_edge_strength_on_radius(
				gray,
				static_cast<double>(center.x),
				static_cast<double>(center.y),
				radius,
				theta);

			edge_strengths.push_back(edge_result.strength);
			best_radii.push_back(edge_result.radius);

			if (edge_result.strength >= kCircleMinEdgeStrength)
			{
				supports[i] = 1;
			}
		}

		int support_count = 0;
		for (const int support : supports)
		{
			support_count += support;
		}

		stats.support_ratio = static_cast<double>(support_count) / static_cast<double>(kCircleCheckSamples);

		int max_gap = 0;
		int current_gap = 0;

		for (int i = 0; i < kCircleCheckSamples * 2; ++i)
		{
			const int support = supports[i % kCircleCheckSamples];

			if (support == 0)
			{
				current_gap++;
				max_gap = std::max(max_gap, current_gap);
			}
			else
			{
				current_gap = 0;
			}
		}

		stats.max_gap = std::min(max_gap, kCircleCheckSamples);

		std::vector<double> valid_radii;
		std::vector<double> valid_edges;
		valid_radii.reserve(kCircleCheckSamples);
		valid_edges.reserve(kCircleCheckSamples);

		for (int i = 0; i < kCircleCheckSamples; ++i)
		{
			if (supports[i] == 1)
			{
				valid_radii.push_back(best_radii[i]);
				valid_edges.push_back(edge_strengths[i]);
			}
		}

		if (valid_radii.size() < 4)
		{
			return false;
		}

		double mean_radius = 0.0;
		for (const double radius_value : valid_radii)
		{
			mean_radius += radius_value;
		}
		mean_radius /= static_cast<double>(valid_radii.size());

		double radius_variance = 0.0;
		for (const double radius_value : valid_radii)
		{
			const double error = radius_value - mean_radius;
			radius_variance += error * error;
		}
		radius_variance /= static_cast<double>(valid_radii.size());
		stats.radius_std = std::sqrt(radius_variance);

		stats.edge_mean = 0.0;
		for (const double edge_value : valid_edges)
		{
			stats.edge_mean += edge_value;
		}
		stats.edge_mean /= static_cast<double>(valid_edges.size());

		const double magnitude_score = 0.5; // OpenCV Hough does not expose CanMV magnitude.
		const double support_score = std::clamp(stats.support_ratio, 0.0, 1.0);
		const double gap_score = 1.0 - std::min(
										   1.0,
										   static_cast<double>(stats.max_gap) / static_cast<double>(std::max(1, kCircleMaxGapSamples)));
		// OLD STRICT:
		// const double radius_std_score = 1.0 - std::min(
		// 	1.0,
		// 	stats.radius_std / std::max(0.1, kCircleMaxRadiusStd));
		//
		// New: cho phép radius_std tăng theo bán kính vòng.
		// Ở ảnh 1280x720 hoặc vòng to, ngưỡng cố định 2.8 px quá gắt.
		const double radius_std_limit = std::max(kCircleMaxRadiusStd, radius * 0.080);
		const double radius_std_score = 1.0 - std::min(
												  1.0,
												  stats.radius_std / std::max(0.1, radius_std_limit));
		const double edge_score = std::clamp(stats.edge_mean / 35.0, 0.0, 1.0);

		stats.confidence =
			0.18 * magnitude_score +
			0.34 * support_score +
			0.20 * gap_score +
			0.18 * radius_std_score +
			0.10 * edge_score;

		if (stats.support_ratio < kCircleMinSupportRatio)
		{
			return false;
		}

		if (stats.max_gap > kCircleMaxGapSamples)
		{
			return false;
		}

		// OLD STRICT:
		// if (stats.radius_std > kCircleMaxRadiusStd) {
		// 	return false;
		// }
		if (stats.radius_std > radius_std_limit)
		{
			return false;
		}

		if (stats.confidence < kCompleteCircleMinConfidence)
		{
			return false;
		}

		return true;
	}

	std::vector<cv::Point> make_circle_contour(
		const cv::Point2f &center,
		float radius,
		int samples = 96)
	{
		std::vector<cv::Point> contour;
		contour.reserve(samples);

		for (int i = 0; i < samples; ++i)
		{
			const double theta = 2.0 * kPi * static_cast<double>(i) / static_cast<double>(samples);
			const int x = static_cast<int>(std::round(center.x + radius * std::cos(theta)));
			const int y = static_cast<int>(std::round(center.y + radius * std::sin(theta)));
			contour.emplace_back(x, y);
		}

		return contour;
	}

	void append_complete_circle_candidates_from_hough(
		const cv::Mat &gray_for_hough,
		const cv::Mat &gray_for_validation,
		double dp,
		double min_dist,
		double param1,
		double param2,
		int min_radius,
		int max_radius,
		std::vector<CompleteCircleCandidateData> &candidates)
	{
		if (gray_for_hough.empty() || gray_for_validation.empty())
		{
			return;
		}

		std::vector<cv::Vec3f> circles;
		cv::HoughCircles(
			gray_for_hough,
			circles,
			cv::HOUGH_GRADIENT,
			dp,
			min_dist,
			param1,
			param2,
			min_radius,
			max_radius);

		for (const auto &circle : circles)
		{
			const cv::Point2f center(circle[0], circle[1]);
			const float radius = circle[2];

			if (radius <= 1.0f)
			{
				continue;
			}

			CompleteCircleStats stats;
			if (!validate_complete_circle_candidate(
					gray_for_validation,
					center,
					static_cast<double>(radius),
					stats))
			{
				continue;
			}

			CompleteCircleCandidateData candidate;
			candidate.center = center;
			candidate.radius = radius;
			candidate.stats = stats;
			candidates.push_back(candidate);
		}
	}
} // namespace

RingDetectorNode::RingDetectorNode()
	: Node("ring_detector_node")
{
	loadParameters();
	initBodyKalmanMatrices();

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

	_reset_status_pub = create_publisher<std_msgs::msg::String>(
		_reset_status_topic,
		pub_qos);

	_target_pose_camera_raw_pub = create_publisher<geometry_msgs::msg::PoseStamped>(
		_target_pose_camera_raw_topic,
		pub_qos);

	_target_error_body_raw_pub = create_publisher<geometry_msgs::msg::PoseStamped>(
		_target_error_body_raw_topic,
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
	RCLCPP_INFO(get_logger(), "RingDetector reset status out         : %s", _reset_status_topic.c_str());
	RCLCPP_INFO(get_logger(), "RingDetector reset command input      : %s", _ring_detect_reset_topic.c_str());
	RCLCPP_INFO(get_logger(), "RingDetector camera raw pose out      : %s", _target_pose_camera_raw_topic.c_str());
	RCLCPP_INFO(get_logger(), "RingDetector body raw error out       : %s", _target_error_body_raw_topic.c_str());
	RCLCPP_INFO(get_logger(), "RingDetector body filtered error out  : %s", _target_error_body_filtered_topic.c_str());
	RCLCPP_INFO(get_logger(), "RingDetector body filtered vel out    : %s", _target_velocity_body_filtered_topic.c_str());
	RCLCPP_INFO(get_logger(), "RingDetector target frame id          : %s", _target_frame_id.c_str());
	RCLCPP_INFO(get_logger(), "RingDetector body frame id            : %s", _body_frame_id.c_str());
	RCLCPP_INFO(get_logger(), "RingDetector white mask enabled       : %s", _param_use_white_mask ? "true" : "false");
	RCLCPP_INFO(get_logger(), "RingDetector side-by-side debug       : %s", _param_publish_color_detect_debug ? "true" : "false");
}

void RingDetectorNode::loadParameters()
{
	declare_parameter<std::string>("image_topic", "/camera_front/image_raw");
	declare_parameter<std::string>("camera_info_topic", "/camera_front/camera_info");

	declare_parameter<std::string>("output_namespace", "/ring_detect");
	declare_parameter<std::string>("processed_image_topic", "");

	// Giữ lại param này để không lỗi nếu YAML cũ còn khai báo,
	// nhưng node sẽ không publish topic riêng nữa.
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
	declare_parameter<double>("min_area", 700.0);
	declare_parameter<double>("min_radius_px", 10.0);
	declare_parameter<double>("min_visible_size", 28.0);
	declare_parameter<double>("circularity_min", 0.05);
	declare_parameter<double>("aspect_min", 0.30);
	declare_parameter<double>("aspect_max", 3.50);

	declare_parameter<bool>("use_white_mask", true);

	// true: /ring_detect/image_proc sẽ ghép 2 khung ngang:
	// left = detection debug, right = color detect debug.
	declare_parameter<bool>("publish_color_detect_debug", true);

	declare_parameter<int>("white_s_max", 85);
	declare_parameter<int>("white_v_min", 115);
	declare_parameter<int>("white_morph_kernel", 3);

	declare_parameter<double>("far_min_area", 120.0);
	declare_parameter<double>("far_min_radius_px", 5.0);
	declare_parameter<double>("far_min_visible_size", 20.0);
	declare_parameter<double>("far_circularity_min", 0.02);
	declare_parameter<double>("far_aspect_min", 0.35);
	declare_parameter<double>("far_aspect_max", 3.00);
	declare_parameter<double>("far_canny_low", 35.0);
	declare_parameter<double>("far_canny_high", 110.0);
	declare_parameter<double>("far_roi_margin_ratio", 0.01);
	declare_parameter<int>("far_confirm_frames", 3);
	declare_parameter<double>("far_confirm_pos_gate_px", 45.0);
	declare_parameter<double>("far_confirm_size_ratio", 0.45);

	declare_parameter<double>("lock_max_dist", 180.0);
	declare_parameter<double>("lock_min_size_ratio", 0.45);
	declare_parameter<int>("max_lock_missed", 30);

	declare_parameter<int>("hold_max_missed", 15);
	declare_parameter<double>("pass_forward_thresh", 0.35);
	declare_parameter<double>("pass_min_visible_size", 220.0);
	declare_parameter<double>("body_kf_hold_timeout_s", 0.30);
	declare_parameter<double>("reset_timeout_s", 0.50);

	declare_parameter<double>("camera_offset_x", 0.09);
	declare_parameter<double>("camera_offset_y", 0.0);
	declare_parameter<double>("camera_offset_z", 0.0);

	declare_parameter<double>("body_kf_q_pos", 0.02);
	declare_parameter<double>("body_kf_q_vel", 0.10);
	declare_parameter<double>("body_kf_r_x", 0.04);
	declare_parameter<double>("body_kf_r_y", 0.02);
	declare_parameter<double>("body_kf_r_z", 0.02);

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
								   ? make_namespaced_topic(output_namespace, "reset_cmd")
								   : ring_detect_reset_topic;

	_target_pose_camera_raw_topic = target_pose_camera_raw_topic.empty()
										? make_namespaced_topic(output_namespace, "target_pose_camera_raw")
										: target_pose_camera_raw_topic;

	_target_error_body_raw_topic = target_error_body_raw_topic.empty()
									   ? make_namespaced_topic(output_namespace, "target_error_body_raw")
									   : target_error_body_raw_topic;

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
	get_parameter("publish_color_detect_debug", _param_publish_color_detect_debug);
	get_parameter("white_s_max", _param_white_s_max);
	get_parameter("white_v_min", _param_white_v_min);
	get_parameter("white_morph_kernel", _param_white_morph_kernel);

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
	get_parameter("pass_forward_thresh", _param_pass_forward_thresh);
	get_parameter("pass_min_visible_size", _param_pass_min_visible_size);
	get_parameter("body_kf_hold_timeout_s", _param_body_kf_hold_timeout_s);
	get_parameter("reset_timeout_s", _param_reset_timeout_s);

	get_parameter("camera_offset_x", _param_camera_offset_x);
	get_parameter("camera_offset_y", _param_camera_offset_y);
	get_parameter("camera_offset_z", _param_camera_offset_z);

	get_parameter("body_kf_q_pos", _param_body_kf_q_pos);
	get_parameter("body_kf_q_vel", _param_body_kf_q_vel);
	get_parameter("body_kf_r_x", _param_body_kf_r_x);
	get_parameter("body_kf_r_y", _param_body_kf_r_y);
	get_parameter("body_kf_r_z", _param_body_kf_r_z);
}

void RingDetectorNode::initBodyKalmanMatrices()
{
	_body_kf_state.setZero();
	_body_kf_P.setIdentity();
	_body_kf_Q.setZero();
	_body_kf_R.setZero();
	_body_kf_H.setZero();

	_body_kf_Q(0, 0) = _param_body_kf_q_pos;
	_body_kf_Q(1, 1) = _param_body_kf_q_pos;
	_body_kf_Q(2, 2) = _param_body_kf_q_pos;
	_body_kf_Q(3, 3) = _param_body_kf_q_vel;
	_body_kf_Q(4, 4) = _param_body_kf_q_vel;
	_body_kf_Q(5, 5) = _param_body_kf_q_vel;

	_body_kf_R(0, 0) = _param_body_kf_r_x;
	_body_kf_R(1, 1) = _param_body_kf_r_y;
	_body_kf_R(2, 2) = _param_body_kf_r_z;

	_body_kf_H(0, 0) = 1.0;
	_body_kf_H(1, 1) = 1.0;
	_body_kf_H(2, 2) = 1.0;

	_body_kf_initialized = false;
	_body_kf_last_time = rclcpp::Time(0, 0, RCL_ROS_TIME);
}

void RingDetectorNode::initBodyKalman(const Eigen::Vector3d &measurement)
{
	_body_kf_state.setZero();
	_body_kf_state(0) = measurement.x();
	_body_kf_state(1) = measurement.y();
	_body_kf_state(2) = measurement.z();

	_body_kf_P.setIdentity();
	_body_kf_P *= 0.10;

	_body_kf_initialized = true;
}

void RingDetectorNode::predictBodyKalman(double dt_s)
{
	if (!_body_kf_initialized)
	{
		return;
	}

	const double dt = std::clamp(dt_s, 1e-3, 0.2);

	Eigen::Matrix<double, kBodyKalmanStateDim, kBodyKalmanStateDim> F;
	F.setIdentity();
	F(0, 3) = dt;
	F(1, 4) = dt;
	F(2, 5) = dt;

	_body_kf_state = F * _body_kf_state;
	_body_kf_P = F * _body_kf_P * F.transpose() + _body_kf_Q;
}

void RingDetectorNode::updateBodyKalman(const Eigen::Vector3d &measurement)
{
	if (!_body_kf_initialized)
	{
		initBodyKalman(measurement);
		return;
	}

	Eigen::Matrix<double, kBodyKalmanMeasDim, 1> z;
	z(0) = measurement.x();
	z(1) = measurement.y();
	z(2) = measurement.z();

	const Eigen::Matrix<double, kBodyKalmanMeasDim, 1> innovation =
		z - _body_kf_H * _body_kf_state;

	const Eigen::Matrix<double, kBodyKalmanMeasDim, kBodyKalmanMeasDim> S =
		_body_kf_H * _body_kf_P * _body_kf_H.transpose() + _body_kf_R;

	const Eigen::Matrix<double, kBodyKalmanStateDim, kBodyKalmanMeasDim> K =
		_body_kf_P * _body_kf_H.transpose() * S.inverse();

	_body_kf_state = _body_kf_state + K * innovation;

	const Eigen::Matrix<double, kBodyKalmanStateDim, kBodyKalmanStateDim> I =
		Eigen::Matrix<double, kBodyKalmanStateDim, kBodyKalmanStateDim>::Identity();

	_body_kf_P = (I - K * _body_kf_H) * _body_kf_P;
}

void RingDetectorNode::resetBodyKalman()
{
	initBodyKalmanMatrices();
	_last_measurement_time = rclcpp::Time(0, 0, RCL_ROS_TIME);
}

Eigen::Vector3d RingDetectorNode::cameraOpticalToFrontBody(
	const Eigen::Vector3d &camera_position) const
{
	Eigen::Vector3d body_position;
	body_position.x() = camera_position.z() + _param_camera_offset_x;
	body_position.y() = camera_position.x() + _param_camera_offset_y;
	body_position.z() = camera_position.y() + _param_camera_offset_z;

	return body_position;
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

void RingDetectorNode::publishBodyRawTarget(
	const std_msgs::msg::Header &header,
	const Eigen::Vector3d &body_raw)
{
	geometry_msgs::msg::PoseStamped body_raw_msg;
	body_raw_msg.header = header;
	body_raw_msg.header.frame_id = _body_frame_id;
	body_raw_msg.pose.position.x = body_raw.x();
	body_raw_msg.pose.position.y = body_raw.y();
	body_raw_msg.pose.position.z = body_raw.z();
	body_raw_msg.pose.orientation.w = 1.0;

	_target_error_body_raw_pub->publish(body_raw_msg);
}

void RingDetectorNode::publishBodyFilteredTarget(const std_msgs::msg::Header &header)
{
	if (!_body_kf_initialized)
	{
		return;
	}

	geometry_msgs::msg::PoseStamped body_filtered_msg;
	body_filtered_msg.header = header;
	body_filtered_msg.header.frame_id = _body_frame_id;
	body_filtered_msg.pose.position.x = _body_kf_state(0);
	body_filtered_msg.pose.position.y = _body_kf_state(1);
	body_filtered_msg.pose.position.z = _body_kf_state(2);
	body_filtered_msg.pose.orientation.w = 1.0;
	_target_error_body_filtered_pub->publish(body_filtered_msg);

	geometry_msgs::msg::TwistStamped body_velocity_msg;
	body_velocity_msg.header = header;
	body_velocity_msg.header.frame_id = _body_frame_id;
	body_velocity_msg.twist.linear.x = _body_kf_state(3);
	body_velocity_msg.twist.linear.y = _body_kf_state(4);
	body_velocity_msg.twist.linear.z = _body_kf_state(5);
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
	const Eigen::Vector3d body_raw = cameraOpticalToFrontBody(camera_position);

	publishCameraRawTarget(header, camera_x, camera_y, camera_z);
	publishBodyRawTarget(header, body_raw);

	if (!_body_kf_initialized)
	{
		initBodyKalman(body_raw);
	}
	else
	{
		double dt_s = 1.0 / 30.0;

		if (_body_kf_last_time.nanoseconds() > 0)
		{
			dt_s = (stamp - _body_kf_last_time).seconds();
		}

		predictBodyKalman(dt_s);
		updateBodyKalman(body_raw);
	}

	_body_kf_last_time = stamp;
	_last_measurement_time = stamp;
	_last_body_raw = body_raw;
	_has_last_body_raw = true;

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

	if (msg->data == "RESET")
	{
		reset_lock_state();

		RCLCPP_WARN(
			get_logger(),
			"[RingDetector] received reset command from %s",
			_ring_detect_reset_topic.c_str());
	}
}

cv::Mat RingDetectorNode::build_white_mask(const cv::Mat &bgr) const
{
	if (bgr.empty())
	{
		return cv::Mat();
	}

	cv::Mat hsv;
	cv::Mat hsv_mask;
	cv::Mat gray_shadow_mask;
	cv::Mat combined_mask;
	cv::Mat clean_mask;

	cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

	const int s_max = std::clamp(_param_white_s_max, 0, 255);
	const int v_min = std::clamp(_param_white_v_min, 0, 255);

	// Lớp 1: trắng/xám theo HSV
	cv::inRange(
		hsv,
		cv::Scalar(0, 0, v_min),
		cv::Scalar(179, s_max, 255),
		hsv_mask);

	// Lớp 2: trắng bị bóng -> thành xám, BGR gần nhau nhưng tối hơn
	std::vector<cv::Mat> bgr_channels;
	cv::split(bgr, bgr_channels);

	cv::Mat max_bg;
	cv::Mat min_bg;
	cv::max(bgr_channels[0], bgr_channels[1], max_bg);
	cv::max(max_bg, bgr_channels[2], max_bg);
	cv::min(bgr_channels[0], bgr_channels[1], min_bg);
	cv::min(min_bg, bgr_channels[2], min_bg);

	cv::Mat channel_delta;
	cv::subtract(max_bg, min_bg, channel_delta);

	const int shadow_v_min = std::clamp(v_min - 35, 0, 255);
	const int gray_delta_max = std::clamp(std::max(35, s_max), 0, 120);

	cv::Mat shadow_brightness_mask;
	cv::Mat gray_balance_mask;
	cv::threshold(max_bg, shadow_brightness_mask, shadow_v_min, 255, cv::THRESH_BINARY);
	cv::threshold(channel_delta, gray_balance_mask, gray_delta_max, 255, cv::THRESH_BINARY_INV);
	cv::bitwise_and(shadow_brightness_mask, gray_balance_mask, gray_shadow_mask);

	cv::bitwise_or(hsv_mask, gray_shadow_mask, combined_mask);

	const int kernel_size = make_odd_kernel_size(_param_white_morph_kernel);

	cv::Mat kernel = cv::getStructuringElement(
		cv::MORPH_ELLIPSE,
		cv::Size(kernel_size, kernel_size));

	// CLOSE trước để nối đoạn vòng bị bóng làm đứt mask
	cv::morphologyEx(combined_mask, clean_mask, cv::MORPH_CLOSE, kernel);
	cv::dilate(clean_mask, clean_mask, kernel, cv::Point(-1, -1), 1);
	cv::morphologyEx(clean_mask, clean_mask, cv::MORPH_OPEN, kernel);

	return clean_mask;
}

cv::Mat RingDetectorNode::make_color_detect_debug_image(const cv::Mat &bgr_frame) const
{
	if (bgr_frame.empty())
	{
		return cv::Mat();
	}

	cv::Mat white_mask = build_white_mask(bgr_frame);

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
		"COLOR DETECT DEBUG",
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

	cv::putText(
		left_bgr,
		"DETECTION DEBUG",
		cv::Point(20, 35),
		cv::FONT_HERSHEY_SIMPLEX,
		0.8,
		cv::Scalar(0, 255, 255),
		2,
		cv::LINE_AA);

	cv::Mat combined;
	cv::hconcat(left_bgr, right_bgr, combined);

	return combined;
}

// ============================================================
// OLD LOGIC KEPT FOR REVIEW - detect_ring_candidates contour/white-mask
// Lý do comment: giữ lại logic cũ để so sánh/rollback,
// nhưng logic active bên dưới dùng complete-circle validation từ K230.
// ============================================================
// OLD: std::vector<RingDetectorNode::RingCandidate>
// OLD: RingDetectorNode::detect_ring_candidates(const cv::Mat &frame, cv::Mat &edges)
// OLD: {
// OLD: 	std::vector<RingCandidate> candidates;
// OLD:
// OLD: 	if (frame.empty()) {
// OLD: 		return candidates;
// OLD: 	}
// OLD:
// OLD: 	const int h = frame.rows;
// OLD: 	const int w = frame.cols;
// OLD:
// OLD: 	const int margin_x = static_cast<int>(w * 0.04);
// OLD: 	const int margin_y = static_cast<int>(h * 0.04);
// OLD:
// OLD: 	if (w - 2 * margin_x <= 0 || h - 2 * margin_y <= 0) {
// OLD: 		return candidates;
// OLD: 	}
// OLD:
// OLD: 	cv::Rect roi_rect(margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y);
// OLD: 	cv::Mat roi = frame(roi_rect);
// OLD:
// OLD: 	cv::Mat gray;
// OLD: 	cv::Mat masked_gray;
// OLD: 	cv::Mat blur;
// OLD:
// OLD: 	cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
// OLD:
// OLD: 	if (_param_use_white_mask) {
// OLD: 		cv::Mat white_mask = build_white_mask(roi);
// OLD:
// OLD: 		if (!white_mask.empty()) {
// OLD: 			cv::bitwise_and(gray, gray, masked_gray, white_mask);
// OLD: 		} else {
// OLD: 			masked_gray = gray;
// OLD: 		}
// OLD: 	} else {
// OLD: 		masked_gray = gray;
// OLD: 	}
// OLD:
// OLD: 	cv::GaussianBlur(masked_gray, blur, cv::Size(7, 7), 1.5);
// OLD: 	cv::Canny(blur, edges, 80, 160);
// OLD:
// OLD: 	std::vector<std::vector<cv::Point>> contours;
// OLD: 	cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
// OLD:
// OLD: 	for (const auto &contour : contours) {
// OLD: 		const double area = cv::contourArea(contour);
// OLD: 		if (area < _param_min_area) {
// OLD: 			continue;
// OLD: 		}
// OLD:
// OLD: 		const double perimeter = cv::arcLength(contour, true);
// OLD: 		if (perimeter < 1e-6) {
// OLD: 			continue;
// OLD: 		}
// OLD:
// OLD: 		const cv::Rect bbox = cv::boundingRect(contour);
// OLD: 		if (bbox.width <= 0 || bbox.height <= 0) {
// OLD: 			continue;
// OLD: 		}
// OLD:
// OLD: 		const double aspect = static_cast<double>(bbox.width) / static_cast<double>(bbox.height);
// OLD: 		if (aspect < _param_aspect_min || aspect > _param_aspect_max) {
// OLD: 			continue;
// OLD: 		}
// OLD:
// OLD: 		const double circularity = 4.0 * kPi * area / (perimeter * perimeter);
// OLD: 		if (circularity < _param_circularity_min) {
// OLD: 			continue;
// OLD: 		}
// OLD:
// OLD: 		cv::Point2f center_roi;
// OLD: 		float radius = 0.0f;
// OLD: 		cv::minEnclosingCircle(contour, center_roi, radius);
// OLD:
// OLD: 		if (radius < _param_min_radius_px) {
// OLD: 			continue;
// OLD: 		}
// OLD:
// OLD: 		const cv::Point2f center_full(center_roi.x + margin_x, center_roi.y + margin_y);
// OLD:
// OLD: 		if (center_full.x - radius < 2.0f ||
// OLD: 			center_full.y - radius < 2.0f ||
// OLD: 			center_full.x + radius > static_cast<float>(w - 2) ||
// OLD: 			center_full.y + radius > static_cast<float>(h - 2)) {
// OLD: 			continue;
// OLD: 		}
// OLD:
// OLD: 		std::vector<cv::Point> contour_full = contour;
// OLD: 		for (auto &p : contour_full) {
// OLD: 			p.x += margin_x;
// OLD: 			p.y += margin_y;
// OLD: 		}
// OLD:
// OLD: 		RingCandidate c;
// OLD: 		c.center = center_full;
// OLD: 		c.radius = radius;
// OLD: 		c.area = static_cast<float>(area);
// OLD: 		c.circularity = static_cast<float>(circularity);
// OLD: 		c.bbox_w = static_cast<float>(bbox.width);
// OLD: 		c.bbox_h = static_cast<float>(bbox.height);
// OLD: 		c.bbox_area = static_cast<float>(bbox.width * bbox.height);
// OLD: 		c.visible_size = static_cast<float>(std::max(bbox.width, bbox.height));
// OLD: 		c.contour = contour_full;
// OLD:
// OLD: 		candidates.push_back(c);
// OLD: 	}
// OLD:
// OLD: 	return candidates;
// OLD: }
// ============================================================
std::vector<RingDetectorNode::RingCandidate>
RingDetectorNode::detect_ring_candidates(const cv::Mat &frame, cv::Mat &edges)
{
	std::vector<RingCandidate> candidates;

	if (frame.empty())
	{
		return candidates;
	}

	cv::Mat gray;
	cv::Mat clahe_img;
	cv::Mat complete_circle_input;
	cv::Mat blur;

	cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
	clahe->apply(gray, clahe_img);

	// NEW LARGE-RING LOGIC:
	// Không lọc theo kích thước target nữa, nhưng vẫn dùng white mask nếu bật
	// để giảm cạnh nền/trần/cửa sinh ra circle giả.
	// OLD ACTIVE: dùng trực tiếp clahe_img cho Hough + validate nên rất dễ ăn cạnh background.
	if (_param_use_white_mask)
	{
		cv::Mat white_mask = build_white_mask(frame);

		if (!white_mask.empty())
		{
			cv::Mat kernel = cv::getStructuringElement(
				cv::MORPH_ELLIPSE,
				cv::Size(5, 5));

			// Nối các đoạn bị đứt trên vòng lớn.
			cv::morphologyEx(white_mask, white_mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);
			cv::dilate(white_mask, white_mask, kernel, cv::Point(-1, -1), 1);

			cv::bitwise_and(clahe_img, clahe_img, complete_circle_input, white_mask);
		}
		else
		{
			complete_circle_input = clahe_img;
		}
	}
	else
	{
		complete_circle_input = clahe_img;
	}

	cv::GaussianBlur(complete_circle_input, blur, cv::Size(5, 5), 1.2);
	cv::Canny(blur, edges, 60, 140);

	const int max_radius = std::max(
		static_cast<int>(std::min(frame.cols, frame.rows) * 0.49),
		static_cast<int>(_param_min_radius_px + 2.0));

	std::vector<CompleteCircleCandidateData> raw_candidates;
	append_complete_circle_candidates_from_hough(
		blur,
		complete_circle_input,
		kNearHoughDp,
		kNearHoughMinDist,
		kNearHoughParam1,
		kNearHoughParam2,
		static_cast<int>(std::max(1.0, _param_min_radius_px)),
		max_radius,
		raw_candidates);

	for (const auto &raw : raw_candidates)
	{
		RingCandidate candidate;
		candidate.center = raw.center;
		candidate.radius = raw.radius;

		// Store complete-circle metrics in existing fields to avoid changing RingDetector.hpp:
		// area        -> support_ratio
		// circularity -> confidence
		// bbox_w      -> max_gap
		// bbox_h      -> radius_std
		candidate.area = static_cast<float>(raw.stats.support_ratio);
		candidate.circularity = static_cast<float>(raw.stats.confidence);
		candidate.bbox_w = static_cast<float>(raw.stats.max_gap);
		candidate.bbox_h = static_cast<float>(raw.stats.radius_std);
		candidate.bbox_area = static_cast<float>(4.0f * raw.radius * raw.radius);
		candidate.visible_size = static_cast<float>(2.0f * raw.radius);
		candidate.contour = make_circle_contour(raw.center, raw.radius);

		candidates.push_back(candidate);
	}

	return candidates;
}

// ============================================================
// OLD LOGIC KEPT FOR REVIEW - detect_ring_candidates_far contour/white-mask
// Lý do comment: giữ lại logic cũ để so sánh/rollback,
// nhưng logic active bên dưới dùng complete-circle validation từ K230.
// ============================================================
// OLD: std::vector<RingDetectorNode::RingCandidate>
// OLD: RingDetectorNode::detect_ring_candidates_far(const cv::Mat &frame, cv::Mat &edges)
// OLD: {
// OLD: 	std::vector<RingCandidate> candidates;
// OLD:
// OLD: 	if (frame.empty()) {
// OLD: 		return candidates;
// OLD: 	}
// OLD:
// OLD: 	const int h = frame.rows;
// OLD: 	const int w = frame.cols;
// OLD:
// OLD: 	const int margin_x = static_cast<int>(w * _param_far_roi_margin_ratio);
// OLD: 	const int margin_y = static_cast<int>(h * _param_far_roi_margin_ratio);
// OLD:
// OLD: 	if (w - 2 * margin_x <= 0 || h - 2 * margin_y <= 0) {
// OLD: 		return candidates;
// OLD: 	}
// OLD:
// OLD: 	cv::Rect roi_rect(margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y);
// OLD: 	cv::Mat roi = frame(roi_rect);
// OLD:
// OLD: 	cv::Mat gray;
// OLD: 	cv::Mat clahe_img;
// OLD: 	cv::Mat masked_clahe;
// OLD: 	cv::Mat blur;
// OLD:
// OLD: 	cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
// OLD:
// OLD: 	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.5, cv::Size(8, 8));
// OLD: 	clahe->apply(gray, clahe_img);
// OLD:
// OLD: 	if (_param_use_white_mask) {
// OLD: 		cv::Mat white_mask = build_white_mask(roi);
// OLD:
// OLD: 		if (!white_mask.empty()) {
// OLD: 			cv::bitwise_and(clahe_img, clahe_img, masked_clahe, white_mask);
// OLD: 		} else {
// OLD: 			masked_clahe = clahe_img;
// OLD: 		}
// OLD: 	} else {
// OLD: 		masked_clahe = clahe_img;
// OLD: 	}
// OLD:
// OLD: 	cv::GaussianBlur(masked_clahe, blur, cv::Size(5, 5), 1.0);
// OLD: 	cv::Canny(blur, edges, _param_far_canny_low, _param_far_canny_high);
// OLD:
// OLD: 	cv::dilate(edges, edges, cv::Mat(), cv::Point(-1, -1), 1);
// OLD:
// OLD: 	std::vector<std::vector<cv::Point>> contours;
// OLD: 	cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
// OLD:
// OLD: 	for (const auto &contour : contours) {
// OLD: 		const double area = cv::contourArea(contour);
// OLD: 		if (area < _param_far_min_area) {
// OLD: 			continue;
// OLD: 		}
// OLD:
// OLD: 		const double perimeter = cv::arcLength(contour, true);
// OLD: 		if (perimeter < 1e-6) {
// OLD: 			continue;
// OLD: 		}
// OLD:
// OLD: 		const cv::Rect bbox = cv::boundingRect(contour);
// OLD: 		if (bbox.width <= 0 || bbox.height <= 0) {
// OLD: 			continue;
// OLD: 		}
// OLD:
// OLD: 		const double aspect = static_cast<double>(bbox.width) / static_cast<double>(bbox.height);
// OLD: 		if (aspect < _param_far_aspect_min || aspect > _param_far_aspect_max) {
// OLD: 			continue;
// OLD: 		}
// OLD:
// OLD: 		const double circularity = 4.0 * kPi * area / (perimeter * perimeter);
// OLD: 		if (circularity < _param_far_circularity_min) {
// OLD: 			continue;
// OLD: 		}
// OLD:
// OLD: 		cv::Point2f center_roi;
// OLD: 		float radius = 0.0f;
// OLD: 		cv::minEnclosingCircle(contour, center_roi, radius);
// OLD:
// OLD: 		if (radius < _param_far_min_radius_px) {
// OLD: 			continue;
// OLD: 		}
// OLD:
// OLD: 		const cv::Point2f center_full(center_roi.x + margin_x, center_roi.y + margin_y);
// OLD:
// OLD: 		if (center_full.x - radius < 1.0f ||
// OLD: 			center_full.y - radius < 1.0f ||
// OLD: 			center_full.x + radius > static_cast<float>(w - 1) ||
// OLD: 			center_full.y + radius > static_cast<float>(h - 1)) {
// OLD: 			continue;
// OLD: 		}
// OLD:
// OLD: 		std::vector<cv::Point> contour_full = contour;
// OLD: 		for (auto &p : contour_full) {
// OLD: 			p.x += margin_x;
// OLD: 			p.y += margin_y;
// OLD: 		}
// OLD:
// OLD: 		RingCandidate c;
// OLD: 		c.center = center_full;
// OLD: 		c.radius = radius;
// OLD: 		c.area = static_cast<float>(area);
// OLD: 		c.circularity = static_cast<float>(circularity);
// OLD: 		c.bbox_w = static_cast<float>(bbox.width);
// OLD: 		c.bbox_h = static_cast<float>(bbox.height);
// OLD: 		c.bbox_area = static_cast<float>(bbox.width * bbox.height);
// OLD: 		c.visible_size = static_cast<float>(std::max(bbox.width, bbox.height));
// OLD: 		c.contour = contour_full;
// OLD:
// OLD: 		candidates.push_back(c);
// OLD: 	}
// OLD:
// OLD: 	return candidates;
// OLD: }
// ============================================================
std::vector<RingDetectorNode::RingCandidate>
RingDetectorNode::detect_ring_candidates_far(const cv::Mat &frame, cv::Mat &edges)
{
	std::vector<RingCandidate> candidates;

	if (frame.empty())
	{
		return candidates;
	}

	cv::Mat gray;
	cv::Mat clahe_img;
	cv::Mat complete_circle_input;
	cv::Mat blur;

	cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.8, cv::Size(8, 8));
	clahe->apply(gray, clahe_img);

	// NEW LARGE-RING LOGIC:
	// Far mode cũng dùng cùng pre-process với near mode để bắt vòng lớn/trắng rõ hơn.
	if (_param_use_white_mask)
	{
		cv::Mat white_mask = build_white_mask(frame);

		if (!white_mask.empty())
		{
			cv::Mat kernel = cv::getStructuringElement(
				cv::MORPH_ELLIPSE,
				cv::Size(5, 5));
			cv::morphologyEx(white_mask, white_mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);
			cv::dilate(white_mask, white_mask, kernel, cv::Point(-1, -1), 1);
			cv::bitwise_and(clahe_img, clahe_img, complete_circle_input, white_mask);
		}
		else
		{
			complete_circle_input = clahe_img;
		}
	}
	else
	{
		complete_circle_input = clahe_img;
	}

	cv::GaussianBlur(complete_circle_input, blur, cv::Size(5, 5), 1.0);
	cv::Canny(blur, edges, 50, 130);

	const int min_radius = static_cast<int>(std::max(1.0, _param_far_min_radius_px));
	const int max_radius = std::max(
		static_cast<int>(std::min(frame.cols, frame.rows) * 0.49),
		min_radius + 2);

	std::vector<CompleteCircleCandidateData> raw_candidates;
	append_complete_circle_candidates_from_hough(
		blur,
		complete_circle_input,
		kFarHoughDp,
		kFarHoughMinDist,
		kFarHoughParam1,
		kFarHoughParam2,
		min_radius,
		max_radius,
		raw_candidates);

	for (const auto &raw : raw_candidates)
	{
		RingCandidate candidate;
		candidate.center = raw.center;
		candidate.radius = raw.radius;
		candidate.area = static_cast<float>(raw.stats.support_ratio);
		candidate.circularity = static_cast<float>(raw.stats.confidence);
		candidate.bbox_w = static_cast<float>(raw.stats.max_gap);
		candidate.bbox_h = static_cast<float>(raw.stats.radius_std);
		candidate.bbox_area = static_cast<float>(4.0f * raw.radius * raw.radius);
		candidate.visible_size = static_cast<float>(2.0f * raw.radius);
		candidate.contour = make_circle_contour(raw.center, raw.radius);

		candidates.push_back(candidate);
	}

	return candidates;
}

// ============================================================
// OLD LOGIC KEPT FOR REVIEW - select_main_ring old scoring
// Lý do comment: giữ lại logic cũ để so sánh/rollback,
// nhưng logic active bên dưới dùng complete-circle validation từ K230.
// ============================================================
// OLD: bool RingDetectorNode::select_main_ring(
// OLD: 	const std::vector<RingCandidate> &candidates,
// OLD: 	RingCandidate &best) const
// OLD: {
// OLD: 	if (candidates.empty()) {
// OLD: 		return false;
// OLD: 	}
// OLD:
// OLD: 	float cx_img = 640.0f;
// OLD: 	float cy_img = 480.0f;
// OLD:
// OLD: 	if (!_camera_matrix.empty()) {
// OLD: 		cx_img = static_cast<float>(_camera_matrix.at<double>(0, 2));
// OLD: 		cy_img = static_cast<float>(_camera_matrix.at<double>(1, 2));
// OLD: 	}
// OLD:
// OLD: 	bool found = false;
// OLD: 	double best_score = -1e18;
// OLD:
// OLD: 	for (const auto &c : candidates) {
// OLD: 		if (c.visible_size < _param_min_visible_size) {
// OLD: 			continue;
// OLD: 		}
// OLD:
// OLD: 		const double dx = static_cast<double>(c.center.x - cx_img);
// OLD: 		const double dy = static_cast<double>(c.center.y - cy_img);
// OLD: 		const double dist_center = std::sqrt(dx * dx + dy * dy);
// OLD:
// OLD: 		const double score =
// OLD: 			-2.0 * dist_center +
// OLD: 			0.8 * static_cast<double>(c.visible_size) +
// OLD: 			50.0 * static_cast<double>(c.circularity);
// OLD:
// OLD: 		if (!found || score > best_score) {
// OLD: 			best = c;
// OLD: 			best_score = score;
// OLD: 			found = true;
// OLD: 		}
// OLD: 	}
// OLD:
// OLD: 	return found;
// OLD: }
// ============================================================
bool RingDetectorNode::select_main_ring(
	const std::vector<RingCandidate> &candidates,
	RingCandidate &best) const
{
	if (candidates.empty())
	{
		return false;
	}

	float cx_img = 640.0f;
	float cy_img = 480.0f;

	if (!_camera_matrix.empty())
	{
		cx_img = static_cast<float>(_camera_matrix.at<double>(0, 2));
		cy_img = static_cast<float>(_camera_matrix.at<double>(1, 2));
	}

	bool found = false;
	double best_score = -1e18;

	for (const auto &c : candidates)
	{
		// OLD SIZE FILTER:
		// if (c.visible_size < _param_min_visible_size) {
		// 	continue;
		// }
		// New: bỏ lọc kích thước, chỉ đánh giá theo độ nguyên vẹn hình tròn.

		const double support_ratio = get_complete_circle_metric_support(c.area);
		const int max_gap = get_complete_circle_metric_gap(c.bbox_w);
		const double radius_std = get_complete_circle_metric_radius_std(c.bbox_h);
		const double confidence = get_complete_circle_metric_confidence(c.circularity);

		const double dx = static_cast<double>(c.center.x - cx_img);
		const double dy = static_cast<double>(c.center.y - cy_img);
		const double dist_center = std::sqrt(dx * dx + dy * dy);

		// Same score as the K230 complete-circle-only detector:
		// prefer full 360-degree support, small gap, stable radius, and near image center.
		const double score =
			// OLD SCORE: + 0.55 * visible_size làm circle lớn giả dễ thắng.
			// New: ưu tiên vòng đầy đủ và ổn định; chỉ cộng size rất nhẹ để chọn vòng lớn khi cùng chất lượng.
			-0.60 * dist_center +
			0.08 * static_cast<double>(c.visible_size) +
			220.0 * support_ratio -
			10.0 * radius_std -
			8.0 * static_cast<double>(max_gap) +
			180.0 * confidence;

		if (!found || score > best_score)
		{
			best = c;
			best_score = score;
			found = true;
		}
	}

	return found;
}

// ============================================================
// OLD LOGIC KEPT FOR REVIEW - select_far_ring old scoring
// Lý do comment: giữ lại logic cũ để so sánh/rollback,
// nhưng logic active bên dưới dùng complete-circle validation từ K230.
// ============================================================
// OLD: bool RingDetectorNode::select_far_ring(
// OLD: 	const std::vector<RingCandidate> &candidates,
// OLD: 	RingCandidate &best) const
// OLD: {
// OLD: 	if (candidates.empty()) {
// OLD: 		return false;
// OLD: 	}
// OLD:
// OLD: 	float cx_img = 640.0f;
// OLD: 	float cy_img = 480.0f;
// OLD:
// OLD: 	if (!_camera_matrix.empty()) {
// OLD: 		cx_img = static_cast<float>(_camera_matrix.at<double>(0, 2));
// OLD: 		cy_img = static_cast<float>(_camera_matrix.at<double>(1, 2));
// OLD: 	}
// OLD:
// OLD: 	bool found = false;
// OLD: 	double best_score = -1e18;
// OLD:
// OLD: 	for (const auto &c : candidates) {
// OLD: 		if (c.visible_size < _param_far_min_visible_size) {
// OLD: 			continue;
// OLD: 		}
// OLD:
// OLD: 		const double dx = static_cast<double>(c.center.x - cx_img);
// OLD: 		const double dy = static_cast<double>(c.center.y - cy_img);
// OLD: 		const double dist_center = std::sqrt(dx * dx + dy * dy);
// OLD:
// OLD: 		const double score =
// OLD: 			-3.0 * dist_center +
// OLD: 			0.25 * static_cast<double>(c.visible_size) +
// OLD: 			40.0 * static_cast<double>(c.circularity) +
// OLD: 			0.003 * static_cast<double>(c.bbox_area);
// OLD:
// OLD: 		if (!found || score > best_score) {
// OLD: 			best = c;
// OLD: 			best_score = score;
// OLD: 			found = true;
// OLD: 		}
// OLD: 	}
// OLD:
// OLD: 	return found;
// OLD: }
// ============================================================
bool RingDetectorNode::select_far_ring(
	const std::vector<RingCandidate> &candidates,
	RingCandidate &best) const
{
	if (candidates.empty())
	{
		return false;
	}

	float cx_img = 640.0f;
	float cy_img = 480.0f;

	if (!_camera_matrix.empty())
	{
		cx_img = static_cast<float>(_camera_matrix.at<double>(0, 2));
		cy_img = static_cast<float>(_camera_matrix.at<double>(1, 2));
	}

	bool found = false;
	double best_score = -1e18;

	for (const auto &c : candidates)
	{
		// OLD SIZE FILTER:
		// if (c.visible_size < _param_far_min_visible_size) {
		// 	continue;
		// }
		// New: bỏ lọc kích thước, chỉ đánh giá theo độ nguyên vẹn hình tròn.

		const double support_ratio = get_complete_circle_metric_support(c.area);
		const int max_gap = get_complete_circle_metric_gap(c.bbox_w);
		const double radius_std = get_complete_circle_metric_radius_std(c.bbox_h);
		const double confidence = get_complete_circle_metric_confidence(c.circularity);

		const double dx = static_cast<double>(c.center.x - cx_img);
		const double dy = static_cast<double>(c.center.y - cy_img);
		const double dist_center = std::sqrt(dx * dx + dy * dy);

		const double score =
			// OLD SCORE: + 0.55 * visible_size làm circle lớn giả dễ thắng.
			// New: ưu tiên vòng đầy đủ và ổn định; chỉ cộng size rất nhẹ để chọn vòng lớn khi cùng chất lượng.
			-0.60 * dist_center +
			0.08 * static_cast<double>(c.visible_size) +
			220.0 * support_ratio -
			10.0 * radius_std -
			8.0 * static_cast<double>(max_gap) +
			180.0 * confidence;

		if (!found || score > best_score)
		{
			best = c;
			best_score = score;
			found = true;
		}
	}

	return found;
}

// ============================================================
// OLD LOGIC KEPT FOR REVIEW - find_locked_ring old scoring
// Lý do comment: giữ lại logic cũ để so sánh/rollback,
// nhưng logic active bên dưới dùng complete-circle validation từ K230.
// ============================================================
// OLD: bool RingDetectorNode::find_locked_ring(
// OLD: 	const std::vector<RingCandidate> &candidates,
// OLD: 	RingCandidate &best) const
// OLD: {
// OLD: 	if (!_locked || candidates.empty()) {
// OLD: 		return false;
// OLD: 	}
// OLD:
// OLD: 	bool found = false;
// OLD: 	double best_score = -1e18;
// OLD:
// OLD: 	for (const auto &c : candidates) {
// OLD: 		const double dx = static_cast<double>(c.center.x - _lock_center.x);
// OLD: 		const double dy = static_cast<double>(c.center.y - _lock_center.y);
// OLD: 		const double dist = std::sqrt(dx * dx + dy * dy);
// OLD:
// OLD: 		if (dist > _param_lock_max_dist) {
// OLD: 			continue;
// OLD: 		}
// OLD:
// OLD: 		if (_lock_visible_size > 1e-6f) {
// OLD: 			if (c.visible_size < static_cast<float>(_lock_visible_size * _param_lock_min_size_ratio)) {
// OLD: 				continue;
// OLD: 			}
// OLD: 		}
// OLD:
// OLD: 		const double size_ratio =
// OLD: 			static_cast<double>(c.visible_size) / std::max(1.0f, _lock_visible_size);
// OLD:
// OLD: 		const double score =
// OLD: 			-dist -
// OLD: 			std::abs(1.0 - size_ratio) * 120.0 +
// OLD: 			0.05 * static_cast<double>(c.bbox_area);
// OLD:
// OLD: 		if (!found || score > best_score) {
// OLD: 			best = c;
// OLD: 			best_score = score;
// OLD: 			found = true;
// OLD: 		}
// OLD: 	}
// OLD:
// OLD: 	return found;
// OLD: }
// ============================================================
bool RingDetectorNode::find_locked_ring(
	const std::vector<RingCandidate> &candidates,
	RingCandidate &best) const
{
	if (!_locked || candidates.empty())
	{
		return false;
	}

	bool found = false;
	double best_score = -1e18;

	for (const auto &c : candidates)
	{
		const double dx = static_cast<double>(c.center.x - _lock_center.x);
		const double dy = static_cast<double>(c.center.y - _lock_center.y);
		const double dist = std::sqrt(dx * dx + dy * dy);

		if (dist > _param_lock_max_dist)
		{
			continue;
		}

		if (_lock_visible_size > 1e-6f)
		{
			if (c.visible_size < static_cast<float>(_lock_visible_size * _param_lock_min_size_ratio))
			{
				continue;
			}
		}

		const double size_ratio =
			static_cast<double>(c.visible_size) / std::max(1.0f, _lock_visible_size);

		const double support_ratio = get_complete_circle_metric_support(c.area);
		const int max_gap = get_complete_circle_metric_gap(c.bbox_w);
		const double radius_std = get_complete_circle_metric_radius_std(c.bbox_h);
		const double confidence = get_complete_circle_metric_confidence(c.circularity);

		const double score =
			-dist -
			std::abs(1.0 - size_ratio) * 80.0 +
			220.0 * support_ratio -
			10.0 * radius_std -
			8.0 * static_cast<double>(max_gap) +
			180.0 * confidence;

		if (!found || score > best_score)
		{
			best = c;
			best_score = score;
			found = true;
		}
	}

	return found;
}

bool RingDetectorNode::confirm_far_candidate(const RingCandidate &candidate)
{
	if (!_has_far_candidate_memory)
	{
		_has_far_candidate_memory = true;
		_far_candidate_center = candidate.center;
		_far_candidate_size = candidate.visible_size;
		_far_candidate_count = 1;
		return (_far_candidate_count >= _param_far_confirm_frames);
	}

	const double dx = static_cast<double>(candidate.center.x - _far_candidate_center.x);
	const double dy = static_cast<double>(candidate.center.y - _far_candidate_center.y);
	const double dist = std::sqrt(dx * dx + dy * dy);

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
	if (!_has_last_body_raw)
	{
		return false;
	}

	if (_last_body_raw.x() > 0.0 &&
		_last_body_raw.x() < _param_pass_forward_thresh)
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
	_last_body_raw = Eigen::Vector3d::Zero();
	_has_last_body_raw = false;

	_has_far_candidate_memory = false;
	_far_candidate_center = cv::Point2f(0.0f, 0.0f);
	_far_candidate_size = 0.0f;
	_far_candidate_count = 0;

	resetBodyKalman();
}

void RingDetectorNode::annotate_image(
	cv_bridge::CvImagePtr image,
	const RingCandidate &target,
	double camera_x,
	double camera_y,
	double camera_z,
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
	std::ostringstream line4;

	line1 << std::fixed << std::setprecision(2)
		  << "cx=" << target.center.x
		  << " cy=" << target.center.y
		  << " r=" << target.radius
		  << " conf=" << get_complete_circle_metric_confidence(target.circularity);

	line2 << std::fixed << std::setprecision(2)
		  << "support=" << get_complete_circle_metric_support(target.area)
		  << " gap=" << get_complete_circle_metric_gap(target.bbox_w)
		  << " std=" << get_complete_circle_metric_radius_std(target.bbox_h);

	if (_has_last_body_raw)
	{
		line3 << std::fixed << std::setprecision(3)
			  << "body raw fwd=" << _last_body_raw.x()
			  << " right=" << _last_body_raw.y()
			  << " down=" << _last_body_raw.z();
	}
	else
	{
		line3 << "body raw unavailable";
	}

	if (_body_kf_initialized)
	{
		line4 << std::fixed << std::setprecision(3)
			  << "body kf x=" << _body_kf_state(0)
			  << " y=" << _body_kf_state(1)
			  << " z=" << _body_kf_state(2);
	}
	else
	{
		line4 << "body kf uninitialized";
	}

	cv::putText(frame, state_text, cv::Point(20, 35),
				cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

	cv::putText(frame, line1.str(), cv::Point(20, 70),
				cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

	cv::putText(frame, line2.str(), cv::Point(20, 105),
				cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

	cv::putText(frame, line3.str(), cv::Point(20, 140),
				cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);

	cv::putText(frame, line4.str(), cv::Point(20, 175),
				cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 180, 180), 2, cv::LINE_AA);

	if (_far_candidate_count > 0 && !_locked)
	{
		std::ostringstream far_ss;
		far_ss << "FAR_CONFIRM=" << _far_candidate_count << "/" << _param_far_confirm_frames;
		cv::putText(frame, far_ss.str(), cv::Point(20, 210),
					cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 165, 255), 2, cv::LINE_AA);
	}
}

void RingDetectorNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
	try
	{
		cv_bridge::CvImagePtr cv_ptr =
			cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

		const cv::Mat original_frame = cv_ptr->image.clone();

		cv::Mat edges_near;
		cv::Mat edges_far;

		const auto candidates_near = detect_ring_candidates(cv_ptr->image, edges_near);
		const auto candidates_far = detect_ring_candidates_far(cv_ptr->image, edges_far);

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
				found = find_locked_ring(candidates_near, target);

				if (!found)
				{
					found = find_locked_ring(candidates_far, target);
				}
			}

			if (!found)
			{
				found = select_main_ring(candidates_near, target);
			}

			if (!found)
			{
				RingCandidate far_target;
				const bool far_found = select_far_ring(candidates_far, far_target);

				if (far_found && confirm_far_candidate(far_target))
				{
					target = far_target;
					found = true;
				}
				else if (!far_found)
				{
					_has_far_candidate_memory = false;
					_far_candidate_center = cv::Point2f(0.0f, 0.0f);
					_far_candidate_size = 0.0f;
					_far_candidate_count = 0;
				}
			}

			// K230 complete-circle logic requires a new target to be stable for several frames
			// before it is accepted. Reuse the existing far-candidate memory as a general
			// temporal confirmation gate while the tracker is not locked.
			if (found && !_locked)
			{
				if (!confirm_far_candidate(target))
				{
					found = false;
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
						"BODY KF HOLD");
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
		_reset_status_pub->publish(reset_msg);

		cv::Mat color_debug_image;

		if (_param_publish_color_detect_debug)
		{
			color_debug_image = make_color_detect_debug_image(original_frame);
		}

		const cv::Mat combined_debug_image = make_side_by_side_debug_image(
			cv_ptr->image,
			color_debug_image);

		cv_bridge::CvImage out_msg;
		out_msg.header = msg->header;
		out_msg.encoding = sensor_msgs::image_encodings::BGR8;

		if (!combined_debug_image.empty())
		{
			out_msg.image = combined_debug_image;
		}
		else
		{
			out_msg.image = cv_ptr->image;
		}

		_image_pub->publish(*out_msg.toImageMsg());
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