#include "RingDetector.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <vector>

#include <Eigen/Dense>
#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.hpp>

RingDetectorNode::RingDetectorNode()
: Node("ring_detector_node")
{
	loadParameters();
	initSsmMatrices();

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

	_image_pub = create_publisher<sensor_msgs::msg::Image>("/image_proc", pub_qos);
	_target_pose_pub = create_publisher<geometry_msgs::msg::PoseStamped>("/target_pose", pub_qos);
	_target_valid_pub = create_publisher<std_msgs::msg::Bool>("/target_valid", pub_qos);
	_reset_pub = create_publisher<std_msgs::msg::String>("/reset", pub_qos);
}

void RingDetectorNode::loadParameters()
{
	declare_parameter<std::string>("image_topic", "/camera/image");
	declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");

	declare_parameter<double>("soft_hold_timeout_s", 0.4);
	declare_parameter<double>("reset_timeout_s", 0.3);
	declare_parameter<double>("last_seen_hold_timeout_s", 1.5);

	declare_parameter<double>("ring_diameter_m", 1.8);
	declare_parameter<double>("min_area", 700.0);
	declare_parameter<double>("min_radius_px", 12.0);
	declare_parameter<double>("min_visible_size", 60.0);
	declare_parameter<double>("circularity_min", 0.05);
	declare_parameter<double>("aspect_min", 0.30);
	declare_parameter<double>("aspect_max", 3.50);

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
	declare_parameter<double>("lock_alpha", 0.75);

	declare_parameter<int>("hold_max_missed", 15);
	declare_parameter<double>("pass_z_thresh", 1.2);
	declare_parameter<double>("pass_min_visible_size", 220.0);

	declare_parameter<double>("reacquire_gate_px", 75.0);
	declare_parameter<double>("reacquire_size_ratio", 0.20);
	declare_parameter<double>("reacquire_cost_pos", 4.0);
	declare_parameter<double>("reacquire_cost_size", 60.0);
	declare_parameter<double>("reacquire_cost_area", 0.01);

	declare_parameter<double>("pass_commit_timeout_s", 0.8);
	declare_parameter<double>("pass_commit_z_thresh", 1.8);
	declare_parameter<double>("pass_commit_visible_size", 180.0);
	declare_parameter<double>("pass_commit_center_px", 90.0);
	declare_parameter<int>("pass_commit_max_missed", 30);
	declare_parameter<double>("pass_commit_forward_speed_mps", 1.2);

	// ===== SSM parameters =====
	declare_parameter<double>("ssm_q_pos", 20.0);
	declare_parameter<double>("ssm_q_vel", 80.0);
	declare_parameter<double>("ssm_q_size", 25.0);
	declare_parameter<double>("ssm_q_size_vel", 60.0);
	declare_parameter<double>("ssm_q_pose_xy", 0.15);

	declare_parameter<double>("ssm_r_pos", 18.0);
	declare_parameter<double>("ssm_r_size", 12.0);
	declare_parameter<double>("ssm_r_pose_x", 0.10);

	declare_parameter<double>("track_confidence_decay", 0.90);
	declare_parameter<double>("track_confidence_gain", 0.25);
	declare_parameter<double>("track_confidence_min_for_hold", 0.15);

	get_parameter("image_topic", _image_topic);
	get_parameter("camera_info_topic", _camera_info_topic);

	get_parameter("soft_hold_timeout_s", _param_soft_hold_timeout_s);
	get_parameter("reset_timeout_s", _param_reset_timeout_s);
	get_parameter("last_seen_hold_timeout_s", _param_last_seen_hold_timeout_s);

	get_parameter("ring_diameter_m", _param_ring_diameter_m);
	get_parameter("min_area", _param_min_area);
	get_parameter("min_radius_px", _param_min_radius_px);
	get_parameter("min_visible_size", _param_min_visible_size);
	get_parameter("circularity_min", _param_circularity_min);
	get_parameter("aspect_min", _param_aspect_min);
	get_parameter("aspect_max", _param_aspect_max);

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
	get_parameter("lock_alpha", _param_lock_alpha);

	get_parameter("hold_max_missed", _param_hold_max_missed);
	get_parameter("pass_z_thresh", _param_pass_z_thresh);
	get_parameter("pass_min_visible_size", _param_pass_min_visible_size);

	get_parameter("reacquire_gate_px", _param_reacquire_gate_px);
	get_parameter("reacquire_size_ratio", _param_reacquire_size_ratio);
	get_parameter("reacquire_cost_pos", _param_reacquire_cost_pos);
	get_parameter("reacquire_cost_size", _param_reacquire_cost_size);
	get_parameter("reacquire_cost_area", _param_reacquire_cost_area);

	get_parameter("pass_commit_timeout_s", _param_pass_commit_timeout_s);
	get_parameter("pass_commit_z_thresh", _param_pass_commit_z_thresh);
	get_parameter("pass_commit_visible_size", _param_pass_commit_visible_size);
	get_parameter("pass_commit_center_px", _param_pass_commit_center_px);
	get_parameter("pass_commit_max_missed", _param_pass_commit_max_missed);
	get_parameter("pass_commit_forward_speed_mps", _param_pass_commit_forward_speed_mps);

	get_parameter("ssm_q_pos", _param_ssm_q_pos);
	get_parameter("ssm_q_vel", _param_ssm_q_vel);
	get_parameter("ssm_q_size", _param_ssm_q_size);
	get_parameter("ssm_q_size_vel", _param_ssm_q_size_vel);
	get_parameter("ssm_q_pose_xy", _param_ssm_q_pose_xy);

	get_parameter("ssm_r_pos", _param_ssm_r_pos);
	get_parameter("ssm_r_size", _param_ssm_r_size);
	get_parameter("ssm_r_pose_x", _param_ssm_r_pose_x);

	get_parameter("track_confidence_decay", _param_track_confidence_decay);
	get_parameter("track_confidence_gain", _param_track_confidence_gain);
	get_parameter("track_confidence_min_for_hold", _param_track_confidence_min_for_hold);
}

void RingDetectorNode::initSsmMatrices()
{
	_ssm_state.setZero();
	_ssm_P.setIdentity();
	_ssm_Q.setZero();
	_ssm_R.setZero();
	_ssm_H.setZero();

	// state = [cx, cy, vx, vy, size, vsize, x_pose, y_pose]
	// meas  = [cx, cy, size, x_pose]
	_ssm_H(0, 0) = 1.0;
	_ssm_H(1, 1) = 1.0;
	_ssm_H(2, 4) = 1.0;
	_ssm_H(3, 6) = 1.0;

	_ssm_Q(0, 0) = _param_ssm_q_pos;
	_ssm_Q(1, 1) = _param_ssm_q_pos;
	_ssm_Q(2, 2) = _param_ssm_q_vel;
	_ssm_Q(3, 3) = _param_ssm_q_vel;
	_ssm_Q(4, 4) = _param_ssm_q_size;
	_ssm_Q(5, 5) = _param_ssm_q_size_vel;
	_ssm_Q(6, 6) = _param_ssm_q_pose_xy;
	_ssm_Q(7, 7) = _param_ssm_q_pose_xy;

	_ssm_R(0, 0) = _param_ssm_r_pos;
	_ssm_R(1, 1) = _param_ssm_r_pos;
	_ssm_R(2, 2) = _param_ssm_r_size;
	_ssm_R(3, 3) = _param_ssm_r_pose_x;

	_ssm_initialized = false;
	_track_confidence = 0.0;
}

void RingDetectorNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
	_camera_matrix = cv::Mat(3, 3, CV_64F, const_cast<double *>(msg->k.data())).clone();

	if (!msg->d.empty()) {
		_dist_coeffs = cv::Mat(
			static_cast<int>(msg->d.size()),
			1,
			CV_64F,
			const_cast<double *>(msg->d.data())).clone();
	} else {
		_dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);
	}

	_has_camera_info = true;
}

std::vector<RingDetectorNode::RingCandidate>
RingDetectorNode::detect_ring_candidates(const cv::Mat &frame, cv::Mat &edges)
{
	std::vector<RingCandidate> candidates;

	if (frame.empty()) {
		return candidates;
	}

	const int h = frame.rows;
	const int w = frame.cols;

	const int margin_x = static_cast<int>(w * 0.04);
	const int margin_y = static_cast<int>(h * 0.04);

	if (w - 2 * margin_x <= 0 || h - 2 * margin_y <= 0) {
		return candidates;
	}

	cv::Rect roi_rect(margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y);
	cv::Mat roi = frame(roi_rect);

	cv::Mat gray;
	cv::Mat blur;

	cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(gray, blur, cv::Size(7, 7), 1.5);
	cv::Canny(blur, edges, 80, 160);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	for (const auto &contour : contours) {
		const double area = cv::contourArea(contour);
		if (area < _param_min_area) {
			continue;
		}

		const double perimeter = cv::arcLength(contour, true);
		if (perimeter < 1e-6) {
			continue;
		}

		const cv::Rect bbox = cv::boundingRect(contour);
		if (bbox.width <= 0 || bbox.height <= 0) {
			continue;
		}

		const double aspect = static_cast<double>(bbox.width) / static_cast<double>(bbox.height);
		if (aspect < _param_aspect_min || aspect > _param_aspect_max) {
			continue;
		}

		const double circularity = 4.0 * M_PI * area / (perimeter * perimeter);
		if (circularity < _param_circularity_min) {
			continue;
		}

		cv::Point2f center_roi;
		float radius = 0.0f;
		cv::minEnclosingCircle(contour, center_roi, radius);

		if (radius < _param_min_radius_px) {
			continue;
		}

		const cv::Point2f center_full(center_roi.x + margin_x, center_roi.y + margin_y);

		if (center_full.x - radius < 2.0f ||
			center_full.y - radius < 2.0f ||
			center_full.x + radius > static_cast<float>(w - 2) ||
			center_full.y + radius > static_cast<float>(h - 2)) {
			continue;
		}

		std::vector<cv::Point> contour_full = contour;
		for (auto &p : contour_full) {
			p.x += margin_x;
			p.y += margin_y;
		}

		RingCandidate c;
		c.center = center_full;
		c.radius = radius;
		c.area = static_cast<float>(area);
		c.circularity = static_cast<float>(circularity);
		c.bbox_w = static_cast<float>(bbox.width);
		c.bbox_h = static_cast<float>(bbox.height);
		c.bbox_area = static_cast<float>(bbox.width * bbox.height);
		c.visible_size = static_cast<float>(std::max(bbox.width, bbox.height));
		c.contour = contour_full;

		candidates.push_back(c);
	}

	return candidates;
}

std::vector<RingDetectorNode::RingCandidate>
RingDetectorNode::detect_ring_candidates_far(const cv::Mat &frame, cv::Mat &edges)
{
	std::vector<RingCandidate> candidates;

	if (frame.empty()) {
		return candidates;
	}

	const int h = frame.rows;
	const int w = frame.cols;

	const int margin_x = static_cast<int>(w * _param_far_roi_margin_ratio);
	const int margin_y = static_cast<int>(h * _param_far_roi_margin_ratio);

	if (w - 2 * margin_x <= 0 || h - 2 * margin_y <= 0) {
		return candidates;
	}

	cv::Rect roi_rect(margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y);
	cv::Mat roi = frame(roi_rect);

	cv::Mat gray;
	cv::Mat clahe_img;
	cv::Mat blur;

	cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);

	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.5, cv::Size(8, 8));
	clahe->apply(gray, clahe_img);

	cv::GaussianBlur(clahe_img, blur, cv::Size(5, 5), 1.0);
	cv::Canny(blur, edges, _param_far_canny_low, _param_far_canny_high);

	cv::dilate(edges, edges, cv::Mat(), cv::Point(-1, -1), 1);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	for (const auto &contour : contours) {
		const double area = cv::contourArea(contour);
		if (area < _param_far_min_area) {
			continue;
		}

		const double perimeter = cv::arcLength(contour, true);
		if (perimeter < 1e-6) {
			continue;
		}

		const cv::Rect bbox = cv::boundingRect(contour);
		if (bbox.width <= 0 || bbox.height <= 0) {
			continue;
		}

		const double aspect = static_cast<double>(bbox.width) / static_cast<double>(bbox.height);
		if (aspect < _param_far_aspect_min || aspect > _param_far_aspect_max) {
			continue;
		}

		const double circularity = 4.0 * M_PI * area / (perimeter * perimeter);
		if (circularity < _param_far_circularity_min) {
			continue;
		}

		cv::Point2f center_roi;
		float radius = 0.0f;
		cv::minEnclosingCircle(contour, center_roi, radius);

		if (radius < _param_far_min_radius_px) {
			continue;
		}

		const cv::Point2f center_full(center_roi.x + margin_x, center_roi.y + margin_y);

		if (center_full.x - radius < 1.0f ||
			center_full.y - radius < 1.0f ||
			center_full.x + radius > static_cast<float>(w - 1) ||
			center_full.y + radius > static_cast<float>(h - 1)) {
			continue;
		}

		std::vector<cv::Point> contour_full = contour;
		for (auto &p : contour_full) {
			p.x += margin_x;
			p.y += margin_y;
		}

		RingCandidate c;
		c.center = center_full;
		c.radius = radius;
		c.area = static_cast<float>(area);
		c.circularity = static_cast<float>(circularity);
		c.bbox_w = static_cast<float>(bbox.width);
		c.bbox_h = static_cast<float>(bbox.height);
		c.bbox_area = static_cast<float>(bbox.width * bbox.height);
		c.visible_size = static_cast<float>(std::max(bbox.width, bbox.height));
		c.contour = contour_full;

		candidates.push_back(c);
	}

	return candidates;
}

bool RingDetectorNode::select_main_ring(const std::vector<RingCandidate> &candidates, RingCandidate &best) const
{
	if (candidates.empty()) {
		return false;
	}

	std::vector<RingCandidate> filtered;
	for (const auto &c : candidates) {
		if (c.visible_size >= _param_min_visible_size) {
			filtered.push_back(c);
		}
	}

	if (filtered.empty()) {
		return false;
	}

	float cx_img = 640.0f;
	float cy_img = 480.0f;

	if (!_camera_matrix.empty()) {
		cx_img = static_cast<float>(_camera_matrix.at<double>(0, 2));
		cy_img = static_cast<float>(_camera_matrix.at<double>(1, 2));
	}

	bool found = false;
	double best_score = -1e18;

	for (const auto &c : filtered) {
		const double dx = static_cast<double>(c.center.x - cx_img);
		const double dy = static_cast<double>(c.center.y - cy_img);
		const double dist_center = std::sqrt(dx * dx + dy * dy);

		const double score =
			-2.0 * dist_center +
			0.8 * static_cast<double>(c.visible_size) +
			50.0 * static_cast<double>(c.circularity);

		if (!found || score > best_score) {
			best = c;
			best_score = score;
			found = true;
		}
	}

	return found;
}

bool RingDetectorNode::select_far_ring(const std::vector<RingCandidate> &candidates, RingCandidate &best) const
{
	if (candidates.empty()) {
		return false;
	}

	std::vector<RingCandidate> filtered;
	for (const auto &c : candidates) {
		if (c.visible_size >= _param_far_min_visible_size) {
			filtered.push_back(c);
		}
	}

	if (filtered.empty()) {
		return false;
	}

	float cx_img = 640.0f;
	float cy_img = 480.0f;

	if (!_camera_matrix.empty()) {
		cx_img = static_cast<float>(_camera_matrix.at<double>(0, 2));
		cy_img = static_cast<float>(_camera_matrix.at<double>(1, 2));
	}

	bool found = false;
	double best_score = -1e18;

	for (const auto &c : filtered) {
		const double dx = static_cast<double>(c.center.x - cx_img);
		const double dy = static_cast<double>(c.center.y - cy_img);
		const double dist_center = std::sqrt(dx * dx + dy * dy);

		const double score =
			-3.0 * dist_center +
			0.25 * static_cast<double>(c.visible_size) +
			40.0 * static_cast<double>(c.circularity) +
			0.003 * static_cast<double>(c.bbox_area);

		if (!found || score > best_score) {
			best = c;
			best_score = score;
			found = true;
		}
	}

	return found;
}

bool RingDetectorNode::find_locked_ring(const std::vector<RingCandidate> &candidates, RingCandidate &best) const
{
	if (!_locked || candidates.empty()) {
		return false;
	}

	bool found = false;
	double best_score = -1e18;

	for (const auto &c : candidates) {
		const double dx = static_cast<double>(c.center.x - _lock_center.x);
		const double dy = static_cast<double>(c.center.y - _lock_center.y);
		const double dist = std::sqrt(dx * dx + dy * dy);

		if (dist > _param_lock_max_dist) {
			continue;
		}

		if (_lock_visible_size > 1e-6f) {
			if (c.visible_size < static_cast<float>(_lock_visible_size * _param_lock_min_size_ratio)) {
				continue;
			}
		}

		const double size_ratio =
			static_cast<double>(c.visible_size) / std::max(1.0f, _lock_visible_size);

		const double dist_score = -dist;
		const double size_score = -std::abs(1.0 - size_ratio) * 120.0;
		const double area_score = 0.05 * static_cast<double>(c.bbox_area);
		const double score = dist_score + size_score + area_score;

		if (!found || score > best_score) {
			best = c;
			best_score = score;
			found = true;
		}
	}

	return found;
}

bool RingDetectorNode::find_predicted_ring(const std::vector<RingCandidate> &candidates, RingCandidate &best) const
{
	if (!_has_track_memory || !_ssm_initialized || candidates.empty()) {
		return false;
	}

	cv::Point2f pred_center;
	float pred_size;
	get_predicted_track_measurement(pred_center, pred_size);

	bool found = false;
	double best_cost = 1e18;

	for (const auto &c : candidates) {
		if (c.visible_size < _param_far_min_visible_size) {
			continue;
		}

		const double dx = static_cast<double>(c.center.x - pred_center.x);
		const double dy = static_cast<double>(c.center.y - pred_center.y);
		const double dist = std::sqrt(dx * dx + dy * dy);

		const double dynamic_gate =
			std::max(_param_reacquire_gate_px, static_cast<double>(pred_size) * 1.5);

		if (dist > dynamic_gate) {
			continue;
		}

		const double size_ref = std::max(1.0, static_cast<double>(pred_size));
		const double size_ratio_err =
			std::abs(static_cast<double>(c.visible_size) - size_ref) / size_ref;

		const double allowed_size_ratio =
			(pred_size < static_cast<float>(_param_min_visible_size))
				? std::max(_param_reacquire_size_ratio, 0.55)
				: _param_reacquire_size_ratio;

		if (size_ratio_err > allowed_size_ratio) {
			continue;
		}

		const double cost =
			_param_reacquire_cost_pos * dist +
			_param_reacquire_cost_size * size_ratio_err -
			_param_reacquire_cost_area * static_cast<double>(c.bbox_area);

		if (!found || cost < best_cost) {
			best = c;
			best_cost = cost;
			found = true;
		}
	}

	return found;
}

bool RingDetectorNode::confirm_far_candidate(const RingCandidate &candidate)
{
	if (!_has_far_candidate_memory) {
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
		size_ratio_err <= _param_far_confirm_size_ratio) {
		_far_candidate_count++;
	} else {
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

void RingDetectorNode::estimate_pose_from_ssm_state(double &x, double &y, double &z) const
{
	if (_camera_matrix.empty() || !_ssm_initialized) {
		x = 0.0;
		y = 0.0;
		z = 0.0;
		return;
	}

	const double fx = _camera_matrix.at<double>(0, 0);
	const double fy = _camera_matrix.at<double>(1, 1);
	const double cx0 = _camera_matrix.at<double>(0, 2);
	const double cy0 = _camera_matrix.at<double>(1, 2);

	const double cx = _ssm_state(0);
	const double cy = _ssm_state(1);
	const double size = std::max(1.0, _ssm_state(4));

	z = fx * _param_ring_diameter_m / size;
	x = (cx - cx0) * z / fx;
	y = (cy - cy0) * z / fy;
}

bool RingDetectorNode::is_passed_ring() const
{
	if (!_has_last_good_target) {
		return false;
	}

	if (_last_z > 0.0 && _last_z < _param_pass_z_thresh) {
		return true;
	}

	if (_last_good_target.visible_size >= static_cast<float>(_param_pass_min_visible_size)) {
		return true;
	}

	return false;
}

bool RingDetectorNode::should_commit_pass_through() const
{
	if (!_has_last_good_target || _camera_matrix.empty()) {
		return false;
	}

	const float cx_img = static_cast<float>(_camera_matrix.at<double>(0, 2));
	const float cy_img = static_cast<float>(_camera_matrix.at<double>(1, 2));

	const double dx = static_cast<double>(_last_good_target.center.x - cx_img);
	const double dy = static_cast<double>(_last_good_target.center.y - cy_img);
	const double dist_to_center = std::sqrt(dx * dx + dy * dy);

	const bool close_enough = (_last_z > 0.0 && _last_z < _param_pass_commit_z_thresh);
	const bool large_enough =
		(_last_good_target.visible_size >= static_cast<float>(_param_pass_commit_visible_size));
	const bool centered_enough = (dist_to_center <= _param_pass_commit_center_px);

	return close_enough && large_enough && centered_enough;
}

bool RingDetectorNode::has_last_seen_pose() const
{
	return _has_last_seen_pose;
}

void RingDetectorNode::init_track_state(const RingCandidate &target, const rclcpp::Time &now_ts)
{
	double x = 0.0;
	double y = 0.0;
	double z = 0.0;
	estimate_ring_pose_from_image(target, 0, 0, x, y, z);

	_ssm_state.setZero();
	_ssm_state(0) = static_cast<double>(target.center.x);
	_ssm_state(1) = static_cast<double>(target.center.y);
	_ssm_state(2) = 0.0;
	_ssm_state(3) = 0.0;
	_ssm_state(4) = static_cast<double>(target.visible_size);
	_ssm_state(5) = 0.0;
	_ssm_state(6) = x;
	_ssm_state(7) = y;

	_ssm_P.setIdentity();
	_ssm_P *= 20.0;

	_ssm_initialized = true;
	_track_confidence = 1.0;

	_track_center = target.center;
	_track_velocity = cv::Point2f(0.0f, 0.0f);
	_track_size = target.visible_size;
	_track_size_velocity = 0.0f;
	_has_track_memory = true;
	_soft_lost = false;
	_last_track_time = now_ts;
	_last_match_time = now_ts;
}

void RingDetectorNode::predict_track_state(const rclcpp::Time &now_ts)
{
	if (!_has_track_memory || !_ssm_initialized) {
		return;
	}

	if (_last_track_time.nanoseconds() == 0) {
		_last_track_time = now_ts;
		return;
	}

	double dt = (now_ts - _last_track_time).seconds();
	if (dt <= 0.0) {
		return;
	}

	dt = std::min(dt, 0.2);

	Eigen::Matrix<double, 8, 8> F;
	F.setIdentity();
	F(0, 2) = dt;
	F(1, 3) = dt;
	F(4, 5) = dt;

	_ssm_state = F * _ssm_state;
	_ssm_P = F * _ssm_P * F.transpose() + _ssm_Q;

	_track_center.x = static_cast<float>(_ssm_state(0));
	_track_center.y = static_cast<float>(_ssm_state(1));
	_track_velocity.x = static_cast<float>(_ssm_state(2));
	_track_velocity.y = static_cast<float>(_ssm_state(3));
	_track_size = static_cast<float>(std::max(1.0, _ssm_state(4)));
	_track_size_velocity = static_cast<float>(_ssm_state(5));

	_track_confidence *= _param_track_confidence_decay;
	_track_confidence = std::max(0.0, _track_confidence);

	_last_track_time = now_ts;
}

void RingDetectorNode::update_track_state_from_measurement(
	const RingCandidate &target,
	const rclcpp::Time &now_ts)
{
	if (!_has_track_memory || !_ssm_initialized || _last_match_time.nanoseconds() == 0) {
		init_track_state(target, now_ts);
		return;
	}

	double meas_x = 0.0;
	double meas_y = 0.0;
	double meas_z = 0.0;
	estimate_ring_pose_from_image(target, 0, 0, meas_x, meas_y, meas_z);

	Eigen::Matrix<double, 4, 1> z;
	z(0) = static_cast<double>(target.center.x);
	z(1) = static_cast<double>(target.center.y);
	z(2) = static_cast<double>(target.visible_size);
	z(3) = meas_x;

	const Eigen::Matrix<double, 4, 1> innovation = z - _ssm_H * _ssm_state;
	const Eigen::Matrix<double, 4, 4> S = _ssm_H * _ssm_P * _ssm_H.transpose() + _ssm_R;
	const Eigen::Matrix<double, 8, 4> K = _ssm_P * _ssm_H.transpose() * S.inverse();

	_ssm_state = _ssm_state + K * innovation;

	const Eigen::Matrix<double, 8, 8> I = Eigen::Matrix<double, 8, 8>::Identity();
	_ssm_P = (I - K * _ssm_H) * _ssm_P;

	_ssm_state(6) = meas_x;
	_ssm_state(7) = meas_y;

	_track_center.x = static_cast<float>(_ssm_state(0));
	_track_center.y = static_cast<float>(_ssm_state(1));
	_track_velocity.x = static_cast<float>(_ssm_state(2));
	_track_velocity.y = static_cast<float>(_ssm_state(3));
	_track_size = static_cast<float>(std::max(1.0, _ssm_state(4)));
	_track_size_velocity = static_cast<float>(_ssm_state(5));

	_track_confidence = std::min(1.0, _track_confidence + _param_track_confidence_gain);

	_last_track_time = now_ts;
	_last_match_time = now_ts;
}

void RingDetectorNode::get_predicted_track_measurement(
	cv::Point2f &center,
	float &visible_size) const
{
	center.x = static_cast<float>(_ssm_state(0));
	center.y = static_cast<float>(_ssm_state(1));
	visible_size = static_cast<float>(std::max(1.0, _ssm_state(4)));
}

void RingDetectorNode::reset_lock_state()
{
	_locked = false;
	_in_hold = false;
	_lock_center = cv::Point2f(0.0f, 0.0f);
	_lock_visible_size = 0.0f;
	_lock_missed = 0;
	_hold_missed = 0;

	_has_last_good_target = false;
	_last_x = 0.0;
	_last_y = 0.0;
	_last_z = 0.0;

	_has_detect_time = false;
	_last_detect_time = rclcpp::Time(0, 0, RCL_ROS_TIME);

	_has_track_memory = false;
	_soft_lost = false;
	_track_center = cv::Point2f(0.0f, 0.0f);
	_track_velocity = cv::Point2f(0.0f, 0.0f);
	_track_size = 0.0f;
	_track_size_velocity = 0.0f;
	_last_track_time = rclcpp::Time(0, 0, RCL_ROS_TIME);
	_last_match_time = rclcpp::Time(0, 0, RCL_ROS_TIME);

	_ssm_state.setZero();
	_ssm_P.setIdentity();
	_ssm_initialized = false;
	_track_confidence = 0.0;

	_last_seen_timer_started = false;
	_last_seen_lost_start_time = rclcpp::Time(0, 0, RCL_ROS_TIME);
	_last_seen_x = 0.0;
	_last_seen_y = 0.0;
	_last_seen_z = 0.0;
	_has_last_seen_pose = false;

	_has_far_candidate_memory = false;
	_far_candidate_center = cv::Point2f(0.0f, 0.0f);
	_far_candidate_size = 0.0f;
	_far_candidate_count = 0;
}

void RingDetectorNode::annotate_image(
	cv_bridge::CvImagePtr image,
	const RingCandidate &target,
	double x,
	double y,
	double z) const
{
	auto &frame = image->image;

	const int w = frame.cols;
	const int h = frame.rows;

	const cv::Point image_center(w / 2, h / 2);
	const cv::Point ring_center(
		static_cast<int>(std::round(target.center.x)),
		static_cast<int>(std::round(target.center.y)));

	if (!target.contour.empty()) {
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

	if (_has_track_memory) {
		const cv::Point pred_center(
			static_cast<int>(std::round(_track_center.x)),
			static_cast<int>(std::round(_track_center.y)));

		cv::circle(frame, pred_center, 6, cv::Scalar(255, 0, 255), 2);
		cv::circle(
			frame,
			pred_center,
			static_cast<int>(std::round(
				std::max(_param_reacquire_gate_px, static_cast<double>(_track_size) * 1.5))),
			cv::Scalar(255, 0, 255),
			1);
	}

	std::ostringstream line1;
	std::ostringstream line2;
	std::ostringstream line3;
	std::ostringstream line4;

	line1 << std::fixed << std::setprecision(2)
	      << "cx=" << target.center.x
	      << " cy=" << target.center.y
	      << " r=" << target.radius;

	line2 << std::fixed << std::setprecision(3)
	      << "x=" << x
	      << " y=" << y
	      << " z=" << z;

	if (_soft_lost) {
		line3 << "STATE=SOFT_LOST hold=" << _hold_missed;
	} else if (_locked) {
		line3 << "STATE=LOCKED missed=" << _lock_missed;
	} else if (_has_track_memory) {
		line3 << "STATE=REACQUIRE";
	} else {
		line3 << "STATE=SEARCH_INIT";
	}

	line4 << std::fixed << std::setprecision(2)
	      << "SSM_CONF=" << _track_confidence
	      << " vx=" << _track_velocity.x
	      << " vy=" << _track_velocity.y;

	cv::putText(frame, "RING DETECTED", cv::Point(20, 35),
		cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

	cv::putText(frame, line1.str(), cv::Point(20, 70),
		cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

	cv::putText(frame, line2.str(), cv::Point(20, 105),
		cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

	cv::putText(frame, line3.str(), cv::Point(20, 140),
		cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);

	cv::putText(frame, line4.str(), cv::Point(20, 175),
		cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 180, 180), 2, cv::LINE_AA);

	if (_far_candidate_count > 0 && !_has_track_memory) {
		std::ostringstream far_ss;
		far_ss << "FAR_CONFIRM=" << _far_candidate_count << "/" << _param_far_confirm_frames;
		cv::putText(frame, far_ss.str(), cv::Point(20, 210),
			cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 165, 255), 2, cv::LINE_AA);
	}
}

void RingDetectorNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
	try {
		cv_bridge::CvImagePtr cv_ptr =
			cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

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

		if (!_has_camera_info) {
			cv::putText(
				cv_ptr->image,
				"WAITING CAMERA INFO",
				cv::Point(20, 35),
				cv::FONT_HERSHEY_SIMPLEX,
				0.8,
				cv::Scalar(0, 255, 255),
				2,
				cv::LINE_AA);
		} else {
			RingCandidate target;
			bool found = false;

			if (_has_track_memory) {
				predict_track_state(now_ts);

				found = find_predicted_ring(candidates_near, target);

				if (!found) {
					found = find_predicted_ring(candidates_far, target);
				}

				if (!found) {
					found = find_locked_ring(candidates_near, target);
				}

				if (!found) {
					found = find_locked_ring(candidates_far, target);
				}

				if (found) {
					_locked = true;
					_soft_lost = false;
					_in_hold = false;
					_lock_missed = 0;
					_hold_missed = 0;

					update_track_state_from_measurement(target, now_ts);

					_lock_center = _track_center;
					_lock_visible_size = _track_size;

					target.center = _track_center;
					target.visible_size = _track_size;
					target.radius = _track_size * 0.5f;

					_has_far_candidate_memory = false;
					_far_candidate_count = 0;
				} else {
					_locked = false;
					_soft_lost = true;
					_in_hold = true;
					_lock_missed++;
					_hold_missed++;
				}
			} else {
				found = select_main_ring(candidates_near, target);

				if (found) {
					_has_far_candidate_memory = false;
					_far_candidate_count = 0;
				} else {
					RingCandidate far_target;
					bool far_found = select_far_ring(candidates_far, far_target);

					if (far_found) {
						if (confirm_far_candidate(far_target)) {
							target = far_target;
							found = true;
						} else {
							found = false;
						}
					} else {
						_has_far_candidate_memory = false;
						_far_candidate_center = cv::Point2f(0.0f, 0.0f);
						_far_candidate_size = 0.0f;
						_far_candidate_count = 0;
						found = false;
					}
				}

				if (found) {
					_locked = true;
					_soft_lost = false;
					_in_hold = false;
					_lock_missed = 0;
					_hold_missed = 0;

					init_track_state(target, now_ts);

					_lock_center = target.center;
					_lock_visible_size = target.visible_size;
				}
			}

			if (found) {
				double x = 0.0;
				double y = 0.0;
				double z = 0.0;

				estimate_ring_pose_from_image(
					target,
					cv_ptr->image.cols,
					cv_ptr->image.rows,
					x,
					y,
					z);

				_last_good_target = target;
				_has_last_good_target = true;

				_last_x = x;
				_last_y = y;
				_last_z = z;

				_last_seen_x = x;
				_last_seen_y = y;
				_last_seen_z = z;
				_has_last_seen_pose = true;
				_last_seen_timer_started = false;
				_last_seen_lost_start_time = rclcpp::Time(0, 0, RCL_ROS_TIME);

				_last_detect_time = now_ts;
				_has_detect_time = true;

				geometry_msgs::msg::PoseStamped pose_msg;
				pose_msg.header.stamp = msg->header.stamp;
				pose_msg.header.frame_id = "camera_optical_frame";
				pose_msg.pose.position.x = x;
				pose_msg.pose.position.y = y;
				pose_msg.pose.position.z = z;
				pose_msg.pose.orientation.x = 0.0;
				pose_msg.pose.orientation.y = 0.0;
				pose_msg.pose.orientation.z = 0.0;
				pose_msg.pose.orientation.w = 1.0;

				_target_pose_pub->publish(pose_msg);

				valid_msg.data = true;
				reset_msg.data = "ACTIVE";

				annotate_image(cv_ptr, target, x, y, z);
			}
			else if (_has_track_memory) {
				const double lost_time_s =
					(_last_match_time.nanoseconds() > 0)
						? (now_ts - _last_match_time).seconds()
						: 0.0;

				if (!_last_seen_timer_started) {
					_last_seen_timer_started = true;
					_last_seen_lost_start_time = now_ts;
				}

				const double last_seen_hold_time_s =
					_last_seen_timer_started
						? (now_ts - _last_seen_lost_start_time).seconds()
						: 0.0;

				const bool pass_commit = should_commit_pass_through();

				const double active_hold_timeout =
					pass_commit
						? std::max(_param_soft_hold_timeout_s, _param_pass_commit_timeout_s)
						: _param_soft_hold_timeout_s;

				const int active_hold_max_missed =
					pass_commit
						? std::max(_param_hold_max_missed, _param_pass_commit_max_missed)
						: _param_hold_max_missed;

				const bool use_last_seen_hold =
					has_last_seen_pose() &&
					last_seen_hold_time_s <= _param_last_seen_hold_timeout_s;

				const bool confidence_hold_ok =
					_track_confidence >= _param_track_confidence_min_for_hold;

				if (((lost_time_s <= active_hold_timeout &&
					_hold_missed <= active_hold_max_missed &&
					confidence_hold_ok) ||
					use_last_seen_hold)) {

					double x = 0.0;
					double y = 0.0;
					double z = 0.0;

					if (use_last_seen_hold) {
						x = _last_seen_x;
						y = _last_seen_y;
						z = _last_seen_z;
					}
					else if (pass_commit) {
						x = _last_x;
						y = _last_y;
						z = std::max(0.05, _last_z - _param_pass_commit_forward_speed_mps * lost_time_s);
					}
					else {
						estimate_pose_from_ssm_state(x, y, z);
					}

					geometry_msgs::msg::PoseStamped pose_msg;
					pose_msg.header.stamp = msg->header.stamp;
					pose_msg.header.frame_id = "camera_optical_frame";
					pose_msg.pose.position.x = x;
					pose_msg.pose.position.y = y;
					pose_msg.pose.position.z = z;
					pose_msg.pose.orientation.x = 0.0;
					pose_msg.pose.orientation.y = 0.0;
					pose_msg.pose.orientation.z = 0.0;
					pose_msg.pose.orientation.w = 1.0;
					_target_pose_pub->publish(pose_msg);

					valid_msg.data = true;
					reset_msg.data = "ACTIVE";

					_last_x = x;
					_last_y = y;
					_last_z = z;

					RingCandidate pred_target = _has_last_good_target ? _last_good_target : RingCandidate{};
					pred_target.center = _track_center;
					pred_target.visible_size = _track_size;
					pred_target.radius = _track_size * 0.5f;

					annotate_image(cv_ptr, pred_target, x, y, z);

					std::ostringstream ss;

					if (use_last_seen_hold) {
						ss << "LAST_SEEN HOLD t=" << std::fixed << std::setprecision(2)
						   << last_seen_hold_time_s << "s";
					} else if (pass_commit) {
						ss << "PASS COMMIT lost_t=" << std::fixed << std::setprecision(2)
						   << lost_time_s << "s";
					} else {
						ss << "SSM HOLD lost_t=" << std::fixed << std::setprecision(2)
						   << lost_time_s << "s conf=" << _track_confidence;
					}

					cv::putText(
						cv_ptr->image,
						ss.str(),
						cv::Point(20, 245),
						cv::FONT_HERSHEY_SIMPLEX,
						0.7,
						cv::Scalar(0, 165, 255),
						2,
						cv::LINE_AA);
				} else {
					bool passed = false;

					if (_has_last_good_target) {
						passed = is_passed_ring();
					}

					if (passed) {
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
					} else {
						cv::putText(
							cv_ptr->image,
							"RING LOST HARD",
							cv::Point(20, 35),
							cv::FONT_HERSHEY_SIMPLEX,
							0.8,
							cv::Scalar(0, 0, 255),
							2,
							cv::LINE_AA);

						const double hard_lost_time =
							(_last_match_time.nanoseconds() > 0)
								? (now_ts - _last_match_time).seconds()
								: 1e9;

						const bool pass_commit_now = should_commit_pass_through();

						const double active_reset_timeout =
							pass_commit_now
								? std::max(_param_reset_timeout_s, _param_pass_commit_timeout_s)
								: _param_reset_timeout_s;

						const bool last_seen_hold_expired =
							(!_has_last_seen_pose) ||
							(last_seen_hold_time_s > _param_last_seen_hold_timeout_s);

						if (hard_lost_time > active_reset_timeout &&
							last_seen_hold_expired) {
							reset_msg.data = "RESET";
							valid_msg.data = false;

							cv::putText(
								cv_ptr->image,
								"RESET: HARD LOST TIMEOUT",
								cv::Point(20, 280),
								cv::FONT_HERSHEY_SIMPLEX,
								0.7,
								cv::Scalar(0, 0, 255),
								2,
								cv::LINE_AA);

							reset_lock_state();
						}
					}
				}
			}
			else {
				cv::putText(
					cv_ptr->image,
					"SEARCHING INITIAL RING",
					cv::Point(20, 35),
					cv::FONT_HERSHEY_SIMPLEX,
					0.8,
					cv::Scalar(0, 0, 255),
					2,
					cv::LINE_AA);

				if (_far_candidate_count > 0) {
					std::ostringstream far_ss;
					far_ss << "FAR_CONFIRM=" << _far_candidate_count << "/" << _param_far_confirm_frames;
					cv::putText(
						cv_ptr->image,
						far_ss.str(),
						cv::Point(20, 70),
						cv::FONT_HERSHEY_SIMPLEX,
						0.7,
						cv::Scalar(0, 165, 255),
						2,
						cv::LINE_AA);
				}
			}
		}

		_target_valid_pub->publish(valid_msg);
		_reset_pub->publish(reset_msg);

		cv_bridge::CvImage out_msg;
		out_msg.header = msg->header;
		out_msg.encoding = sensor_msgs::image_encodings::BGR8;
		out_msg.image = cv_ptr->image;

		_image_pub->publish(*out_msg.toImageMsg());
	}
	catch (const cv_bridge::Exception &) {
	}
	catch (const std::exception &) {
	}
}

int main(int argc, char **argv)
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<RingDetectorNode>());
	rclcpp::shutdown();
	return 0;
}