#include "CubeDetector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <vector>

#include <sensor_msgs/image_encodings.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>

// ═══════════════════════════════════════════════════════════════════
//  Constants
// ═══════════════════════════════════════════════════════════════════
namespace
{
constexpr int kStateSize       = 6;  // [x, y, z, vx, vy, vz]
constexpr int kMeasurementSize = 3;  // [x, y, z]
constexpr const char *kModeAlgorithm = "algorithm";
constexpr const char *kModeYolo8n = "yolo8n";
constexpr const char *kModeYolo26n = "yolo26n";
} // namespace

// ═══════════════════════════════════════════════════════════════════
//  Constructor
// ═══════════════════════════════════════════════════════════════════
CubeDetectorNode::CubeDetectorNode()
	: Node("cube_detector_node")
{
	loadParameters();
	initKalman();

	// Pre-compute morphology kernel once
	_morph_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	_clahe = cv::createCLAHE(_p.clahe_clip_limit, cv::Size(_p.clahe_grid_size, _p.clahe_grid_size));

	// ── Khoi tao FrameTransformer voi belly_fixed_camera ──────────
	// Toan bo output cua node nay su dung PX4 local NED frame.
	// Camera optical frame:
	//   optical X: sang phai anh
	//   optical Y: xuong duoi anh
	//   optical Z: huong nhin cua camera
	//
	// PX4 body frame FRD:
	//   body X: phia truoc UAV
	//   body Y: ben phai UAV
	//   body Z: huong xuong
	//
	// FrameTransformer se thuc hien:
	//   optical -> mount/body FRD -> local NED
	const auto cameraOffsetBody = Eigen::Vector3d(
    _p.cam_offset_x,
    _p.cam_offset_y,
    _p.cam_offset_z);

	const auto ft_config =
		frame_transform::FrameTransformer::makeBellyFixedCameraLeft90Config(cameraOffsetBody);

	_frame_transformer.setConfig(ft_config);

	auto qos = rclcpp::QoS(10).best_effort();

	_image_sub = create_subscription<sensor_msgs::msg::Image>(
		_p.image_topic, qos,
		std::bind(&CubeDetectorNode::image_callback, this, std::placeholders::_1));

	_camera_info_sub = create_subscription<sensor_msgs::msg::CameraInfo>(
		_p.camera_info_topic, qos,
		std::bind(&CubeDetectorNode::camera_info_callback, this, std::placeholders::_1));

	// Reset subscriber
	// Usage: ros2 topic pub --once /cube_detector/reset std_msgs/msg/Bool "{data: true}"
	_reset_sub = create_subscription<std_msgs::msg::Bool>(
		_p.reset_topic, 10,
		std::bind(&CubeDetectorNode::reset_callback, this, std::placeholders::_1));

	// Vehicle odometry subscriber - cap nhat VehicleState cho FrameTransformer moi frame.
	// Dung PX4 /fmu/out/vehicle_odometry de lay position + quaternion trong local NED,
	// khong dung MAVROS/ENU.
	_vehicle_odometry_sub = create_subscription<px4_msgs::msg::VehicleOdometry>(
		_p.vehicle_odometry_topic, qos,
		std::bind(&CubeDetectorNode::vehicleOdometryCallback, this, std::placeholders::_1));
	// Backward compatible topic: pose da transform + da Kalman.
	_target_pose_pub = create_publisher<geometry_msgs::msg::PoseStamped>(_p.target_pose_topic, qos);

	// Debug/tune topics
	_target_pose_camera_raw_pub =
		create_publisher<geometry_msgs::msg::PoseStamped>(_p.target_pose_camera_raw_topic, qos);

	_target_pose_world_raw_pub =
		create_publisher<geometry_msgs::msg::PoseStamped>(_p.target_pose_world_raw_topic, qos);

	_target_pose_world_filtered_pub =
		create_publisher<geometry_msgs::msg::PoseStamped>(_p.target_pose_world_filtered_topic, qos);

	_target_velocity_world_filtered_pub =
		create_publisher<geometry_msgs::msg::TwistStamped>(_p.target_velocity_world_filtered_topic, qos);

	_target_valid_pub = create_publisher<std_msgs::msg::Bool>(_p.target_valid_topic, qos);
	_perf_pub         = create_publisher<std_msgs::msg::String>(_p.performance_topic, 10);

	// Algorithm debug publisher
	// Topic: /cube_detector/algorithm_debug
	// detector_mode=algorithm: luoi 2x3 (LAB, mask mau, morph, pre-temporal, final, overlay)
	// YOLO: 2 panel (mask fallback + overlay)
	_hsv_debug_pub = create_publisher<sensor_msgs::msg::Image>(_p.hsv_debug_topic, qos);

	RCLCPP_INFO(get_logger(), "===== CubeDetectorNode started =====");
	RCLCPP_INFO(get_logger(), "image topic       : %s", _p.image_topic.c_str());
	RCLCPP_INFO(get_logger(), "camera info topic : %s", _p.camera_info_topic.c_str());
	RCLCPP_INFO(get_logger(), "vehicle odom topic: %s", _p.vehicle_odometry_topic.c_str());
	RCLCPP_INFO(get_logger(), "box_width_m       : %.3f", _p.box_width_m);
	RCLCPP_INFO(get_logger(), "min_box_area      : %d",   _p.min_box_area);
	RCLCPP_INFO(get_logger(), "max_lock_missed   : %d",   _p.max_lock_missed);
	RCLCPP_INFO(get_logger(), "kalman q_acc=(%.4f,%.4f,%.4f) r_pos=(%.4f,%.4f,%.4f)",
		_p.kalman_q_acc_x, _p.kalman_q_acc_y, _p.kalman_q_acc_z,
		_p.kalman_r_pos_x, _p.kalman_r_pos_y, _p.kalman_r_pos_z);
	RCLCPP_INFO(get_logger(), "FrameTransformer  : belly_fixed_camera offset=(%.3f,%.3f,%.3f)",
		_p.cam_offset_x, _p.cam_offset_y, _p.cam_offset_z);
	RCLCPP_INFO(get_logger(), "frame camera/ned  : %s / %s",
		_p.camera_frame_id.c_str(), _p.world_frame_id.c_str());
	RCLCPP_INFO(get_logger(), "Output filtered   : %s and %s",
		_p.target_pose_topic.c_str(), _p.target_pose_world_filtered_topic.c_str());
	RCLCPP_INFO(get_logger(), "detector_mode     : %s", _p.detector_mode.c_str());

	if (_p.detector_mode == kModeYolo8n) {
		_yolo8_loaded = loadYoloModel(_yolo8_net, _p.yolo8_model_path, "yolo8n");
	} else if (_p.detector_mode == kModeYolo26n) {
		_yolo26_loaded = loadYoloModel(_yolo26_net, _p.yolo26_model_path, "yolo26n");
	}
}

// ═══════════════════════════════════════════════════════════════════
//  Parameter loading
// ═══════════════════════════════════════════════════════════════════
void CubeDetectorNode::loadParameters()
{
	// Topics
	_p.image_topic = declare_parameter<std::string>("topics.image", _p.image_topic);
	_p.camera_info_topic = declare_parameter<std::string>("topics.camera_info", _p.camera_info_topic);
	_p.vehicle_odometry_topic = declare_parameter<std::string>("topics.vehicle_odometry", _p.vehicle_odometry_topic);
	_p.reset_topic = declare_parameter<std::string>("topics.reset", _p.reset_topic);

	_p.hsv_debug_topic = declare_parameter<std::string>("topics.hsv_debug", _p.hsv_debug_topic);
	_p.performance_topic = declare_parameter<std::string>("topics.performance", _p.performance_topic);

	_p.target_pose_topic =
		declare_parameter<std::string>("topics.target_pose", _p.target_pose_topic);
	_p.target_pose_camera_raw_topic =
		declare_parameter<std::string>("topics.target_pose_camera_raw", _p.target_pose_camera_raw_topic);
	_p.target_pose_world_raw_topic =
		declare_parameter<std::string>("topics.target_pose_world_raw", _p.target_pose_world_raw_topic);
	_p.target_pose_world_filtered_topic =
		declare_parameter<std::string>("topics.target_pose_world_filtered", _p.target_pose_world_filtered_topic);
	_p.target_velocity_world_filtered_topic =
		declare_parameter<std::string>("topics.target_velocity_world_filtered", _p.target_velocity_world_filtered_topic);
	_p.target_valid_topic =
		declare_parameter<std::string>("topics.target_valid", _p.target_valid_topic);

	// Frame ids
	_p.camera_frame_id = declare_parameter<std::string>("camera_frame_id", _p.camera_frame_id);
	_p.world_frame_id = declare_parameter<std::string>("world_frame_id", _p.world_frame_id);

	// Box HSV - Orange target
	// OpenCV H[0-180], S[0-255], V[0-255]
	_p.h_min = declare_parameter<int>("h_min", _p.h_min);
	_p.s_min = declare_parameter<int>("s_min", _p.s_min);
	_p.v_min = declare_parameter<int>("v_min", _p.v_min);
	_p.h_max = declare_parameter<int>("h_max", _p.h_max);
	_p.s_max = declare_parameter<int>("s_max", _p.s_max);
	_p.v_max = declare_parameter<int>("v_max", _p.v_max);
	_p.detector_mode = declare_parameter<std::string>("detector_mode", _p.detector_mode);

	_p.lab_l_min = declare_parameter<int>("lab_l_min", _p.lab_l_min);
	_p.lab_a_min = declare_parameter<int>("lab_a_min", _p.lab_a_min);
	_p.lab_b_min = declare_parameter<int>("lab_b_min", _p.lab_b_min);
	_p.lab_l_max = declare_parameter<int>("lab_l_max", _p.lab_l_max);
	_p.lab_a_max = declare_parameter<int>("lab_a_max", _p.lab_a_max);
	_p.lab_b_max = declare_parameter<int>("lab_b_max", _p.lab_b_max);
	_p.clahe_clip_limit = declare_parameter<double>("clahe_clip_limit", _p.clahe_clip_limit);
	_p.clahe_grid_size = declare_parameter<int>("clahe_grid_size", _p.clahe_grid_size);
	_p.min_fill_ratio = declare_parameter<double>("min_fill_ratio", _p.min_fill_ratio);
	_p.min_aspect_ratio = declare_parameter<double>("min_aspect_ratio", _p.min_aspect_ratio);
	_p.max_aspect_ratio = declare_parameter<double>("max_aspect_ratio", _p.max_aspect_ratio);
	_p.max_circularity = declare_parameter<double>("max_circularity", _p.max_circularity);

	_p.lab_enable_overexposed_mask =
		declare_parameter<bool>("lab_enable_overexposed_mask", _p.lab_enable_overexposed_mask);
	_p.lab_ov_l_min = declare_parameter<int>("lab_ov_l_min", _p.lab_ov_l_min);
	_p.lab_ov_a_min = declare_parameter<int>("lab_ov_a_min", _p.lab_ov_a_min);
	_p.lab_ov_b_min = declare_parameter<int>("lab_ov_b_min", _p.lab_ov_b_min);
	_p.lab_ov_l_max = declare_parameter<int>("lab_ov_l_max", _p.lab_ov_l_max);
	_p.lab_ov_a_max = declare_parameter<int>("lab_ov_a_max", _p.lab_ov_a_max);
	_p.lab_ov_b_max = declare_parameter<int>("lab_ov_b_max", _p.lab_ov_b_max);

	_p.lab_enable_shadow_mask =
		declare_parameter<bool>("lab_enable_shadow_mask", _p.lab_enable_shadow_mask);
	_p.lab_sh_l_min = declare_parameter<int>("lab_sh_l_min", _p.lab_sh_l_min);
	_p.lab_sh_a_min = declare_parameter<int>("lab_sh_a_min", _p.lab_sh_a_min);
	_p.lab_sh_b_min = declare_parameter<int>("lab_sh_b_min", _p.lab_sh_b_min);
	_p.lab_sh_l_max = declare_parameter<int>("lab_sh_l_max", _p.lab_sh_l_max);
	_p.lab_sh_a_max = declare_parameter<int>("lab_sh_a_max", _p.lab_sh_a_max);
	_p.lab_sh_b_max = declare_parameter<int>("lab_sh_b_max", _p.lab_sh_b_max);

	_p.lab_ratio_ba_gate = declare_parameter<bool>("lab_ratio_ba_gate", _p.lab_ratio_ba_gate);
	_p.lab_ratio_ba_min  = declare_parameter<int>("lab_ratio_ba_min", _p.lab_ratio_ba_min);
	_p.bgr_ratio_or_branch = declare_parameter<bool>("bgr_ratio_or_branch", _p.bgr_ratio_or_branch);
	_p.bgr_r_over_g        = declare_parameter<double>("bgr_r_over_g", _p.bgr_r_over_g);
	_p.bgr_r_over_b        = declare_parameter<double>("bgr_r_over_b", _p.bgr_r_over_b);

	_p.yolo8_model_path = declare_parameter<std::string>("yolo8_model_path", _p.yolo8_model_path);
	_p.yolo26_model_path = declare_parameter<std::string>("yolo26_model_path", _p.yolo26_model_path);
	_p.yolo_input_size = declare_parameter<int>("yolo_input_size", _p.yolo_input_size);
	_p.yolo_conf_threshold = declare_parameter<double>("yolo_conf_threshold", _p.yolo_conf_threshold);
	_p.yolo_nms_threshold = declare_parameter<double>("yolo_nms_threshold", _p.yolo_nms_threshold);
	_p.yolo_target_class_id = declare_parameter<int>("yolo_target_class_id", _p.yolo_target_class_id);
	_p.yolo_use_cuda = declare_parameter<bool>("yolo_use_cuda", _p.yolo_use_cuda);
	_p.lost_grace_frames = declare_parameter<int>("lost_grace_frames", _p.lost_grace_frames);

	// Box area
	_p.min_box_area = declare_parameter<int>("min_box_area", _p.min_box_area);
	_p.max_box_area = declare_parameter<int>("max_box_area", _p.max_box_area);

	// Circle HSV
	_p.circle_h_min = declare_parameter<int>("circle_h_min", _p.circle_h_min);
	_p.circle_s_min = declare_parameter<int>("circle_s_min", _p.circle_s_min);
	_p.circle_v_min = declare_parameter<int>("circle_v_min", _p.circle_v_min);
	_p.circle_h_max = declare_parameter<int>("circle_h_max", _p.circle_h_max);
	_p.circle_s_max = declare_parameter<int>("circle_s_max", _p.circle_s_max);
	_p.circle_v_max = declare_parameter<int>("circle_v_max", _p.circle_v_max);

	// Circle geometry
	_p.circle_min_radius         = declare_parameter<double>("circle_min_radius",         _p.circle_min_radius);
	_p.circle_max_radius         = declare_parameter<double>("circle_max_radius",         _p.circle_max_radius);
	_p.circle_position_tolerance = declare_parameter<double>("circle_position_tolerance", _p.circle_position_tolerance);

	// Physical / lock
	_p.box_width_m              = declare_parameter<double>("box_width_m",              _p.box_width_m);
	_p.max_lock_missed          = declare_parameter<int>   ("max_lock_missed",          _p.max_lock_missed);
	_p.lock_max_dist_px         = declare_parameter<double>("lock_max_dist_px",         _p.lock_max_dist_px);
	_p.lock_min_size_ratio      = declare_parameter<double>("lock_min_size_ratio",      _p.lock_min_size_ratio);
	_p.lock_search_roi_half_extent_px =
		declare_parameter<int>("lock_search_roi_half_extent_px", _p.lock_search_roi_half_extent_px);
	_p.lock_roi_fallback_full_frame =
		declare_parameter<bool>("lock_roi_fallback_full_frame", _p.lock_roi_fallback_full_frame);
	_p.mask_temporal_alpha = declare_parameter<double>("mask_temporal_alpha", _p.mask_temporal_alpha);

	_p.pose_use_min_area_rect = declare_parameter<bool>("pose_use_min_area_rect", _p.pose_use_min_area_rect);
	_p.pose_corner_subpix     = declare_parameter<bool>("pose_corner_subpix", _p.pose_corner_subpix);
	_p.pose_subpix_win        = declare_parameter<int>("pose_subpix_win", _p.pose_subpix_win);

	_p.texture_reject_enable    = declare_parameter<bool>("texture_reject_enable", _p.texture_reject_enable);
	_p.texture_min_lab_a_stddev  =
		declare_parameter<double>("texture_min_lab_a_stddev", _p.texture_min_lab_a_stddev);

	_p.last_seen_hold_timeout_s = declare_parameter<double>("last_seen_hold_timeout_s", _p.last_seen_hold_timeout_s);

	// Kalman 6-state process noise
	_p.kalman_q_acc_x = declare_parameter<double>("kalman_q_acc_x", _p.kalman_q_acc_x);
	_p.kalman_q_acc_y = declare_parameter<double>("kalman_q_acc_y", _p.kalman_q_acc_y);
	_p.kalman_q_acc_z = declare_parameter<double>("kalman_q_acc_z", _p.kalman_q_acc_z);
	_p.kalman_r_pos_x = declare_parameter<double>("kalman_r_pos_x", _p.kalman_r_pos_x);
	_p.kalman_r_pos_y = declare_parameter<double>("kalman_r_pos_y", _p.kalman_r_pos_y);
	_p.kalman_r_pos_z = declare_parameter<double>("kalman_r_pos_z", _p.kalman_r_pos_z);

	// Scoring weights
	_p.score_dist_weight        = declare_parameter<double>("score_dist_weight",        _p.score_dist_weight);
	_p.score_size_ratio_weight  = declare_parameter<double>("score_size_ratio_weight",  _p.score_size_ratio_weight);
	_p.score_area_weight        = declare_parameter<double>("score_area_weight",        _p.score_area_weight);
	_p.score_fill_weight        = declare_parameter<double>("score_fill_weight",        _p.score_fill_weight);
	_p.score_circularity_weight = declare_parameter<double>("score_circularity_weight", _p.score_circularity_weight);
	_p.score_center_weight      = declare_parameter<double>("score_center_weight",      _p.score_center_weight);

	// Camera offset trong body frame (belly_fixed_camera)
	_p.cam_offset_x = declare_parameter<double>("cam_offset_x", _p.cam_offset_x);
	_p.cam_offset_y = declare_parameter<double>("cam_offset_y", _p.cam_offset_y);
	_p.cam_offset_z = declare_parameter<double>("cam_offset_z", _p.cam_offset_z);

	// Khi mat target: van publish prediction cua Kalman de danh gia/tune,
	// nhung /cube_detector/target_valid se la false.
	_p.publish_prediction_when_lost =
		declare_parameter<bool>("publish_prediction_when_lost", _p.publish_prediction_when_lost);

	if (_p.detector_mode != kModeAlgorithm &&
		_p.detector_mode != kModeYolo8n &&
		_p.detector_mode != kModeYolo26n) {
		RCLCPP_WARN(
			get_logger(),
			"Unknown detector_mode='%s', fallback to '%s'",
			_p.detector_mode.c_str(),
			kModeAlgorithm);
		_p.detector_mode = kModeAlgorithm;
	}
	_p.clahe_grid_size = std::max(2, _p.clahe_grid_size);
	_p.yolo_input_size = std::max(64, _p.yolo_input_size);
	_p.lost_grace_frames = std::max(0, _p.lost_grace_frames);
	_p.lock_search_roi_half_extent_px = std::max(32, _p.lock_search_roi_half_extent_px);
	_p.pose_subpix_win                 = std::max(3, _p.pose_subpix_win | 1);
	_p.mask_temporal_alpha             = std::min(1.0, std::max(0.0, _p.mask_temporal_alpha));
}

// ═══════════════════════════════════════════════════════════════════
//  Kalman 6-state init
// ═══════════════════════════════════════════════════════════════════
void CubeDetectorNode::initKalman()
{
	_kalman.kf = cv::KalmanFilter(kStateSize, kMeasurementSize, 0, CV_64F);

	_kalman.kf.transitionMatrix = cv::Mat::eye(kStateSize, kStateSize, CV_64F);

	_kalman.kf.measurementMatrix = cv::Mat::zeros(kMeasurementSize, kStateSize, CV_64F);
	_kalman.kf.measurementMatrix.at<double>(0, 0) = 1.0;
	_kalman.kf.measurementMatrix.at<double>(1, 1) = 1.0;
	_kalman.kf.measurementMatrix.at<double>(2, 2) = 1.0;

	_kalman.kf.processNoiseCov = cv::Mat::zeros(kStateSize, kStateSize, CV_64F);

	_kalman.kf.measurementNoiseCov = cv::Mat::eye(kMeasurementSize, kMeasurementSize, CV_64F);
	_kalman.kf.measurementNoiseCov.at<double>(0, 0) = _p.kalman_r_pos_x;
	_kalman.kf.measurementNoiseCov.at<double>(1, 1) = _p.kalman_r_pos_y;
	_kalman.kf.measurementNoiseCov.at<double>(2, 2) = _p.kalman_r_pos_z;

	_kalman.kf.errorCovPost = cv::Mat::eye(kStateSize, kStateSize, CV_64F);
	_kalman.kf.errorCovPost.at<double>(3, 3) = 10.0;
	_kalman.kf.errorCovPost.at<double>(4, 4) = 10.0;
	_kalman.kf.errorCovPost.at<double>(5, 5) = 10.0;

	_kalman.kf.statePost = cv::Mat::zeros(kStateSize, 1, CV_64F);
	_kalman.kf.statePre  = cv::Mat::zeros(kStateSize, 1, CV_64F);

	_kalman.q_acc_x = _p.kalman_q_acc_x;
	_kalman.q_acc_y = _p.kalman_q_acc_y;
	_kalman.q_acc_z = _p.kalman_q_acc_z;
	_kalman.r_pos_x = _p.kalman_r_pos_x;
	_kalman.r_pos_y = _p.kalman_r_pos_y;
	_kalman.r_pos_z = _p.kalman_r_pos_z;

	_kalman.initialized     = false;
	_kalman.lastPredictTime = rclcpp::Time(0, 0, get_clock()->get_clock_type());
}

// ═══════════════════════════════════════════════════════════════════
//  Camera info callback
// ═══════════════════════════════════════════════════════════════════
void CubeDetectorNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
	if (_has_camera_info) return;

	_camera_matrix = cv::Mat(3, 3, CV_64F, const_cast<double *>(msg->k.data())).clone();
	_dist_coeffs   = msg->d.empty()
		? cv::Mat::zeros(5, 1, CV_64F)
		: cv::Mat(static_cast<int>(msg->d.size()), 1, CV_64F,
		          const_cast<double *>(msg->d.data())).clone();

	_has_camera_info = true;
	RCLCPP_INFO(get_logger(), "Camera intrinsics received: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
		msg->k[0], msg->k[4], msg->k[2], msg->k[5]);
}

// ═══════════════════════════════════════════════════════════════════
//  Vehicle odometry callback
// ═══════════════════════════════════════════════════════════════════
void CubeDetectorNode::vehicleOdometryCallback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
{
	if (!msg)
	{
		RCLCPP_WARN(get_logger(), "vehicleOdometryCallback received null message");
		return;
	}

	if (msg->pose_frame != px4_msgs::msg::VehicleOdometry::POSE_FRAME_NED)
	{
		RCLCPP_WARN_THROTTLE(
			get_logger(),
			*get_clock(),
			2000,
			"VehicleOdometry pose_frame is not NED. pose_frame=%u",
			static_cast<unsigned>(msg->pose_frame));
		return;
	}

	const bool positionValid =
		std::isfinite(msg->position[0]) &&
		std::isfinite(msg->position[1]) &&
		std::isfinite(msg->position[2]);

	const bool quaternionValid =
		std::isfinite(msg->q[0]) &&
		std::isfinite(msg->q[1]) &&
		std::isfinite(msg->q[2]) &&
		std::isfinite(msg->q[3]);

	if (!positionValid || !quaternionValid)
	{
		RCLCPP_WARN_THROTTLE(
			get_logger(),
			*get_clock(),
			2000,
			"VehicleOdometry has non-finite position or quaternion");
		return;
	}

	frame_transform::VehicleStateData state;

	// PX4 local NED: x = North, y = East, z = Down.
	state.positionWorld = Eigen::Vector3d(
		static_cast<double>(msg->position[0]),
		static_cast<double>(msg->position[1]),
		static_cast<double>(msg->position[2]));

	// PX4 VehicleOdometry quaternion q[] theo thu tu Hamilton: w, x, y, z.
	// Trong pipeline nay xem la quaternion quay tu body FRD sang local NED.
	state.worldFromBody = Eigen::Quaterniond(
		static_cast<double>(msg->q[0]),
		static_cast<double>(msg->q[1]),
		static_cast<double>(msg->q[2]),
		static_cast<double>(msg->q[3]));

	_frame_transformer.setVehicleState(state);
	_has_vehicle_pose = true;
}

// ═══════════════════════════════════════════════════════════════════
//  Reset callback
// ═══════════════════════════════════════════════════════════════════
void CubeDetectorNode::reset_callback(const std_msgs::msg::Bool::SharedPtr /*msg*/)
{
	resetLockState();
	RCLCPP_WARN(get_logger(), "[RESET] Tracking state reset via %s", _p.reset_topic.c_str());
}

// ═══════════════════════════════════════════════════════════════════
//  Image callback  (main loop)
// ═══════════════════════════════════════════════════════════════════
void CubeDetectorNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
	if (!_has_camera_info) {
		RCLCPP_WARN_ONCE(get_logger(), "Waiting for camera info...");
		publishTargetValid(false);
		return;
	}

	cv_bridge::CvImagePtr cv_ptr;
	try {
		cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	} catch (const cv_bridge::Exception &e) {
		RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
		publishTargetValid(false);
		return;
	}

	const auto perf_start = std::chrono::steady_clock::now();
	const rclcpp::Time now_ts =
		(msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0)
			? now()
			: rclcpp::Time(msg->header.stamp);

	double pose_ms   = 0.0;
	double kalman_ms = 0.0;

	const auto start_detect = std::chrono::steady_clock::now();
	VictimModel victim = detectVictimModel(cv_ptr->image);
	const auto end_detect = std::chrono::steady_clock::now();
	const double detect_ms =
		std::chrono::duration_cast<std::chrono::microseconds>(end_detect - start_detect).count() * 1e-3;

	if (victim.valid) {
		_lock.center       = victim.boxCenter;
		_lock.visible_size = static_cast<float>(std::max(victim.boxBbox.width, victim.boxBbox.height));
		_lock.missed       = 0;
		_lock.locked       = true;

		double raw_x = 0.0, raw_y = 0.0, raw_z = 0.0;
		const auto start_pose = std::chrono::steady_clock::now();
		estimatePose(victim, cv_ptr->image.cols, cv_ptr->image.rows, raw_x, raw_y, raw_z);
		const auto end_pose = std::chrono::steady_clock::now();
		pose_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_pose - start_pose).count() * 1e-3;

		// Publish pose raw trong camera optical frame de debug/tune pinhole model.
		publishPose(_target_pose_camera_raw_pub, msg->header, _p.camera_frame_id, raw_x, raw_y, raw_z);

		if (!_has_vehicle_pose) {
			RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
				"[BOX VALID] detected but waiting for PX4 vehicle odometry before optical->NED transform.");
			publishTargetValid(false);
		} else {
			// Transform optical frame -> PX4 local NED frame.
			const Eigen::Vector3d pos_world = _frame_transformer.opticalPositionToWorld(
				Eigen::Vector3d(raw_x, raw_y, raw_z));

			// Publish raw NED pose truoc Kalman de danh gia transform.
			publishPose(_target_pose_world_raw_pub, msg->header, _p.world_frame_id,
				pos_world.x(), pos_world.y(), pos_world.z());

			const auto start_kalman = std::chrono::steady_clock::now();
			kalmanUpdate(pos_world.x(), pos_world.y(), pos_world.z(), now_ts);
			const auto end_kalman = std::chrono::steady_clock::now();
			kalman_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_kalman - start_kalman).count() * 1e-3;

			publishFilteredOutput(msg->header);
			publishTargetValid(true);

			_last_x = _kalman.x();
			_last_y = _kalman.y();
			_last_z = _kalman.z();
			_has_last_pose = true;

			_last_seen.has_pose      = true;
			_last_seen.x             = _kalman.x();
			_last_seen.y             = _kalman.y();
			_last_seen.z             = _kalman.z();
			_last_seen.timer_started = false;
			_last_seen.lost_start    = rclcpp::Time(0, 0, RCL_ROS_TIME);

			RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
				"[BOX VALID] box=(%d,%d,%d,%d) cam=(%.3f,%.3f,%.3f) ned_raw=(%.3f,%.3f,%.3f) filt=(%.3f,%.3f,%.3f) vel=(%.3f,%.3f,%.3f) m/s",
				victim.boxBbox.x, victim.boxBbox.y, victim.boxBbox.width, victim.boxBbox.height,
				raw_x, raw_y, raw_z,
				pos_world.x(), pos_world.y(), pos_world.z(),
				_kalman.x(), _kalman.y(), _kalman.z(),
				_kalman.vx(), _kalman.vy(), _kalman.vz());
		}
	}
	else {
		const auto start_kalman = std::chrono::steady_clock::now();
		kalmanPredict(now_ts);
		const auto end_kalman = std::chrono::steady_clock::now();
		kalman_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_kalman - start_kalman).count() * 1e-3;

		_lock.missed++;

		if (!_last_seen.timer_started) {
			_last_seen.timer_started = true;
			_last_seen.lost_start    = now_ts;
		}

		const double lost_s = _last_seen.timer_started ? (now_ts - _last_seen.lost_start).seconds() : 0.0;

		// Hysteresis: cho phep mat detection trong vai frame de tranh rung do choi sang.
		const bool still_valid_in_grace = _lock.locked && (_lock.missed <= _p.lost_grace_frames);
		publishTargetValid(still_valid_in_grace);

		// Van publish Kalman prediction neu bat param nay, de danh gia/tune filter.
		if (_p.publish_prediction_when_lost && _kalman.initialized) {
			publishFilteredOutput(msg->header);
			_last_x = _kalman.x();
			_last_y = _kalman.y();
			_last_z = _kalman.z();
			_has_last_pose = true;

			RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
				"[PREDICT] lost=%.2fs ned_pred=(%.3f,%.3f,%.3f) vel=(%.3f,%.3f,%.3f) m/s",
				lost_s,
				_kalman.x(), _kalman.y(), _kalman.z(),
				_kalman.vx(), _kalman.vy(), _kalman.vz());
		}

		if (_lock.missed > _p.max_lock_missed) {
			resetLockState();
		}

		RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
			"[SEARCHING] area=%d box=(%d,%d,%d,%d) missed=%d locked=%d",
			victim.boxBbox.area(),
			victim.boxBbox.x, victim.boxBbox.y, victim.boxBbox.width, victim.boxBbox.height,
			_lock.missed, static_cast<int>(_lock.locked));
	}

	const auto start_publish = std::chrono::steady_clock::now();
	const auto end_publish = std::chrono::steady_clock::now();
	const double publish_ms =
		std::chrono::duration_cast<std::chrono::microseconds>(end_publish - start_publish).count() * 1e-3;

	publishHsvDebug(cv_ptr->image, victim, msg->header);
	const double annotate_ms = 0.0;

	const auto perf_end = std::chrono::steady_clock::now();
	const double frame_ms =
		std::chrono::duration_cast<std::chrono::microseconds>(perf_end - perf_start).count() * 1e-3;
	_perf_total_ms    += frame_ms;
	_perf_frame_count += 1;

	_perf_detect_ms   += detect_ms;
	_perf_pose_ms     += (victim.valid ? pose_ms : 0.0);
	_perf_kalman_ms   += kalman_ms;
	_perf_publish_ms  += publish_ms;
	_perf_annotate_ms += annotate_ms;
	_perf_detail_count += 1;

	if (perf_end - _perf_last_pub >= std::chrono::seconds(1)) {
		const double wall_s =
			std::chrono::duration<double>(perf_end - _perf_last_pub).count();
		const double out_fps = (wall_s > 1e-6 && _perf_frame_count > 0)
			? static_cast<double>(_perf_frame_count) / wall_s : 0.0;
		const double avg_ms  = _perf_frame_count > 0
			? _perf_total_ms / static_cast<double>(_perf_frame_count) : 0.0;
		const double cap_fps = avg_ms > 1e-6 ? 1000.0 / avg_ms : 0.0;

		std::ostringstream fps_text;
		fps_text << "FPS=" << std::fixed << std::setprecision(1) << out_fps
		         << " cap=" << std::setprecision(0) << cap_fps;

		std_msgs::msg::String perf_msg;
		perf_msg.data = fps_text.str();
		_perf_pub->publish(perf_msg);
		_perf_overlay_text = perf_msg.data;

		_perf_last_pub     = perf_end;
		_perf_frame_count  = 0;
		_perf_total_ms     = 0.0;
		_perf_detect_ms    = 0.0;
		_perf_pose_ms      = 0.0;
		_perf_kalman_ms    = 0.0;
		_perf_publish_ms   = 0.0;
		_perf_annotate_ms  = 0.0;
		_perf_detail_count = 0;
	}
}

// ═══════════════════════════════════════════════════════════════════
//  Detection pipeline
// ═══════════════════════════════════════════════════════════════════
CubeDetectorNode::VictimModel CubeDetectorNode::detectVictimModel(const cv::Mat &frame)
{
	VictimModel result;

	if (!detectBoxRegion(frame, result)) {
		return result;
	}

	result.valid      = true;
	result.confidence = 0.85f;
	return result;
}

bool CubeDetectorNode::detectBoxRegion(const cv::Mat &frame, VictimModel &vm)
{
	vm.boxBbox          = cv::Rect();
	vm.boxCenter        = {0.0f, 0.0f};
	vm.pose_pixel_width = 0.0;
	vm.pose_center      = {0.0f, 0.0f};

	if (_p.detector_mode == kModeYolo8n) {
		if (!_yolo8_loaded) return false;
		return detectBoxRegionYolo(frame, _yolo8_net, kModeYolo8n, vm);
	}

	if (_p.detector_mode == kModeYolo26n) {
		if (!_yolo26_loaded) return false;
		return detectBoxRegionYolo(frame, _yolo26_net, kModeYolo26n, vm);
	}

	return detectBoxRegionAlgorithm(frame, vm, true);
}

void CubeDetectorNode::prepareAlgorithmLab(const cv::Mat &bgr, cv::Mat &lab)
{
	cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
	std::vector<cv::Mat> ch;
	cv::split(lab, ch);
	if (ch.size() == 3) {
		cv::Mat l_enhanced;
		_clahe->apply(ch[0], l_enhanced);
		ch[0] = l_enhanced;
		cv::merge(ch, lab);
	}
}

void CubeDetectorNode::buildAlgorithmColorMask(const cv::Mat &lab, const cv::Mat &bgr, cv::Mat &mask)
{
	cv::Mat m_primary;
	cv::inRange(
		lab,
		cv::Scalar(_p.lab_l_min, _p.lab_a_min, _p.lab_b_min),
		cv::Scalar(_p.lab_l_max, _p.lab_a_max, _p.lab_b_max),
		m_primary);

	mask = m_primary;

	if (_p.lab_enable_overexposed_mask) {
		cv::Mat mo;
		cv::inRange(
			lab,
			cv::Scalar(_p.lab_ov_l_min, _p.lab_ov_a_min, _p.lab_ov_b_min),
			cv::Scalar(_p.lab_ov_l_max, _p.lab_ov_a_max, _p.lab_ov_b_max),
			mo);
		cv::bitwise_or(mask, mo, mask);
	}

	if (_p.lab_enable_shadow_mask) {
		cv::Mat ms;
		cv::inRange(
			lab,
			cv::Scalar(_p.lab_sh_l_min, _p.lab_sh_a_min, _p.lab_sh_b_min),
			cv::Scalar(_p.lab_sh_l_max, _p.lab_sh_a_max, _p.lab_sh_b_max),
			ms);
		cv::bitwise_or(mask, ms, mask);
	}

	if (_p.lab_ratio_ba_gate) {
		std::vector<cv::Mat> ch;
		cv::split(lab, ch);
		if (ch.size() == 3) {
			cv::Mat ba_diff;
			cv::subtract(ch[2], ch[1], ba_diff, cv::noArray(), CV_16S);
			cv::Mat ba_mask;
			cv::inRange(
				ba_diff,
				cv::Scalar(static_cast<double>(_p.lab_ratio_ba_min)),
				cv::Scalar(32767.0),
				ba_mask);
			cv::bitwise_and(mask, ba_mask, mask);
		}
	}

	if (_p.bgr_ratio_or_branch && !bgr.empty() && bgr.type() == CV_8UC3) {
		std::vector<cv::Mat> bc;
		cv::split(bgr, bc);
		if (bc.size() == 3) {
			cv::Mat Rf, Gf, Bf;
			bc[2].convertTo(Rf, CV_32F);
			bc[1].convertTo(Gf, CV_32F);
			bc[0].convertTo(Bf, CV_32F);
			cv::Mat mg, mb, mor;
			cv::compare(Rf, _p.bgr_r_over_g * Gf, mg, cv::CMP_GT);
			cv::compare(Rf, _p.bgr_r_over_b * Bf, mb, cv::CMP_GT);
			cv::bitwise_and(mg, mb, mor);
			cv::bitwise_or(mask, mor, mask);
		}
	}
}

void CubeDetectorNode::applyAlgorithmMorphology(cv::Mat &mask)
{
	cv::morphologyEx(mask, mask, cv::MORPH_OPEN, _morph_kernel);
	cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, _morph_kernel);
}

void CubeDetectorNode::applyMaskTemporalFullFrame(cv::Mat &mask_full)
{
	if (mask_full.empty()) return;
	if (_p.mask_temporal_alpha <= 1e-9) {
		return;
	}

	const float a = static_cast<float>(_p.mask_temporal_alpha);
	cv::Mat cur_f;
	mask_full.convertTo(cur_f, CV_32F, 1.0 / 255.0);

	if (_mask_temporal_prev.size() != mask_full.size() ||
	    _mask_temporal_prev.type() != CV_32F) {
		_mask_temporal_prev = cur_f.clone();
		return;
	}

	cv::Mat blended = a * cur_f + (1.0f - a) * _mask_temporal_prev;
	_mask_temporal_prev = blended.clone();

	cv::Mat out8;
	cv::threshold(blended, out8, 0.5, 255.0, cv::THRESH_BINARY);
	out8.convertTo(mask_full, CV_8U);
}

bool CubeDetectorNode::makeAlgorithmMaskOnBgr(const cv::Mat &bgr, cv::Mat &mask)
{
	if (bgr.empty()) return false;
	cv::Mat lab;
	prepareAlgorithmLab(bgr, lab);
	buildAlgorithmColorMask(lab, bgr, mask);
	applyAlgorithmMorphology(mask);
	return true;
}

void CubeDetectorNode::makeAlgorithmMaskStagesOnBgr(
	const cv::Mat &bgr,
	cv::Mat &lab_bgr_vis,
	cv::Mat &mask_after_color,
	cv::Mat &mask_after_morph)
{
	if (bgr.empty()) return;
	cv::Mat lab;
	prepareAlgorithmLab(bgr, lab);
	cv::cvtColor(lab, lab_bgr_vis, cv::COLOR_Lab2BGR);
	buildAlgorithmColorMask(lab, bgr, mask_after_color);
	mask_after_morph = mask_after_color.clone();
	applyAlgorithmMorphology(mask_after_morph);
}

bool CubeDetectorNode::makeAlgorithmMask(const cv::Mat &frame, cv::Mat &mask)
{
	return makeAlgorithmMaskOnBgr(frame, mask);
}

cv::Rect CubeDetectorNode::computeLockedSearchRoi(const cv::Size &frame_size) const
{
	const int cx = static_cast<int>(std::lround(_lock.center.x));
	const int cy = static_cast<int>(std::lround(_lock.center.y));
	const int he = _p.lock_search_roi_half_extent_px;
	const int l  = cx - he;
	const int t  = cy - he;
	const int w  = he * 2;
	const int h  = he * 2;
	cv::Rect r(l, t, w, h);
	return r & cv::Rect(0, 0, frame_size.width, frame_size.height);
}

bool CubeDetectorNode::contourTextureAcceptable(
	const cv::Mat &lab, const std::vector<cv::Point> &contour_work) const
{
	if (!_p.texture_reject_enable || lab.empty() || contour_work.empty()) {
		return true;
	}

	cv::Rect b = cv::boundingRect(contour_work);
	b &= cv::Rect(0, 0, lab.cols, lab.rows);
	if (b.width <= 0 || b.height <= 0) return false;

	cv::Mat roi_lab = lab(b);
	std::vector<cv::Mat> pch;
	cv::split(roi_lab, pch);
	if (pch.size() < 2) return true;

	cv::Mat mask_roi = cv::Mat::zeros(b.size(), CV_8U);
	std::vector<cv::Point> shifted;
	shifted.reserve(contour_work.size());
	for (const auto &p : contour_work) {
		shifted.emplace_back(p.x - b.x, p.y - b.y);
	}
	std::vector<std::vector<cv::Point>> one = {shifted};
	cv::drawContours(mask_roi, one, 0, cv::Scalar(255), cv::FILLED, cv::LINE_8);

	cv::Scalar mean, stddev;
	cv::meanStdDev(pch[1], mean, stddev, mask_roi);

	return stddev[0] >= _p.texture_min_lab_a_stddev;
}

bool CubeDetectorNode::detectBoxRegionAlgorithm(const cv::Mat &frame, VictimModel &vm, bool allow_locked_roi)
{
	const cv::Rect full(0, 0, frame.cols, frame.rows);
	const bool      use_roi = allow_locked_roi && _lock.locked;

	cv::Rect roi_rect = full;
	if (use_roi) {
		roi_rect = computeLockedSearchRoi(frame.size());
		if (roi_rect.width < 24 || roi_rect.height < 24) {
			roi_rect = full;
		}
	}

	const bool roi_is_crop =
		use_roi && (roi_rect.x != full.x || roi_rect.y != full.y || roi_rect.width != full.width ||
		            roi_rect.height != full.height);

	cv::Mat saved_temporal_prev = _mask_temporal_prev.clone();

	const auto runContourPass = [&](const cv::Mat &mask_full, const cv::Rect &r, const cv::Mat &work_bgr,
	                                const cv::Mat &lab_tex) -> bool {
		cv::Mat                 mask_work = mask_full(r);
		std::vector<std::vector<cv::Point>> contours_roi;
		cv::findContours(mask_work, contours_roi, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		const cv::Point2f img_center(frame.cols * 0.5f, frame.rows * 0.5f);
		const cv::Point   off(r.x, r.y);

		bool   found     = false;
		double bestScore = -1e18;

		for (const auto &c_roi : contours_roi) {
			std::vector<cv::Point> contour;
			contour.reserve(c_roi.size());
			for (const auto &pt : c_roi) {
				contour.emplace_back(pt.x + off.x, pt.y + off.y);
			}

			const double area = cv::contourArea(contour);
			if (area < _p.min_box_area || area > _p.max_box_area) continue;

			const double perimeter = cv::arcLength(contour, true);
			if (perimeter < 1e-6) continue;

			if (_p.texture_reject_enable && !lab_tex.empty() &&
			    !contourTextureAcceptable(lab_tex, c_roi)) {
				continue;
			}

			const cv::Rect bbox = cv::boundingRect(contour);
			if (bbox.width <= 0 || bbox.height <= 0) continue;

			const cv::Point2f center(
				bbox.x + bbox.width * 0.5f,
				bbox.y + bbox.height * 0.5f);

			const double aspect      = static_cast<double>(bbox.width) / bbox.height;
			const double bbox_area   = static_cast<double>(bbox.width * bbox.height);
			const double fill_ratio  = area / std::max(1.0, bbox_area);
			const double circularity = 4.0 * M_PI * area / (perimeter * perimeter);

			if (circularity > _p.max_circularity) continue;
			if (aspect < _p.min_aspect_ratio || aspect > _p.max_aspect_ratio) continue;
			if (fill_ratio < _p.min_fill_ratio) continue;

			double       pose_pw = static_cast<double>(bbox.width);
			cv::Point2f pose_pc  = center;

			if (_p.pose_use_min_area_rect) {
				if (_p.pose_corner_subpix && c_roi.size() >= 4) {
					cv::Mat gray;
					cv::cvtColor(work_bgr, gray, cv::COLOR_BGR2GRAY);
					std::vector<cv::Point2f> pts;
					pts.reserve(c_roi.size());
					for (const auto &p : c_roi) {
						pts.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
					}
					const int w = std::max(3, _p.pose_subpix_win | 1);
					cv::cornerSubPix(
						gray,
						pts,
						cv::Size(w, w),
						cv::Size(-1, -1),
						cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.05));
					const cv::RotatedRect rr = cv::minAreaRect(pts);
					pose_pw = static_cast<double>(std::max(rr.size.width, rr.size.height));
					pose_pc = rr.center;
					pose_pc.x += static_cast<float>(off.x);
					pose_pc.y += static_cast<float>(off.y);
				} else {
					const cv::RotatedRect rr = cv::minAreaRect(contour);
					pose_pw = static_cast<double>(std::max(rr.size.width, rr.size.height));
					pose_pc = rr.center;
				}
			}

			double score = 0.0;
			if (_lock.locked) {
				if (!isLockedCandidateValid(bbox, center)) continue;
				score = scoreLockedCandidate(center, bbox, area);
			} else {
				score = scoreUnlockedCandidate(area, fill_ratio, circularity, center, img_center);
			}

			if (!found || score > bestScore) {
				bestScore           = score;
				vm.boxBbox          = bbox;
				vm.boxCenter        = center;
				vm.pose_pixel_width = pose_pw;
				vm.pose_center      = pose_pc;
				found               = true;
			}
		}
		return found;
	};

	const auto buildRawMaskInRect = [&](const cv::Rect &r, cv::Mat &mask_full) -> bool {
		cv::Mat work = frame(r);
		cv::Mat mask_roi;
		if (!makeAlgorithmMaskOnBgr(work, mask_roi)) return false;
		mask_full = cv::Mat::zeros(frame.rows, frame.cols, CV_8U);
		mask_roi.copyTo(mask_full(r));
		return true;
	};

	// ── Pass 1: ROI (when locked) ───────────────────────────────
	if (roi_is_crop) {
		cv::Mat raw_full;
		if (!buildRawMaskInRect(roi_rect, raw_full)) {
			_mask_temporal_prev = saved_temporal_prev;
			return false;
		}
		_dbg_algorithm_mask_pre_temporal = raw_full.clone();
		applyMaskTemporalFullFrame(raw_full);
		_dbg_algorithm_mask_full = raw_full.clone();
		_last_debug_roi          = roi_rect;

		cv::Mat lab_tex;
		if (_p.texture_reject_enable) {
			prepareAlgorithmLab(frame(roi_rect), lab_tex);
		}

		if (runContourPass(raw_full, roi_rect, frame(roi_rect), lab_tex)) {
			return true;
		}

		_mask_temporal_prev = saved_temporal_prev;

		if (!_p.lock_roi_fallback_full_frame) {
			return false;
		}
	}

	// ── Pass 2: full frame (search hoac fallback) ───────────────
	cv::Mat raw_full2;
	if (!buildRawMaskInRect(full, raw_full2)) {
		return false;
	}
	_dbg_algorithm_mask_pre_temporal = raw_full2.clone();
	applyMaskTemporalFullFrame(raw_full2);
	_dbg_algorithm_mask_full = raw_full2.clone();
	_last_debug_roi          = full;

	cv::Mat lab_tex2;
	if (_p.texture_reject_enable) {
		prepareAlgorithmLab(frame, lab_tex2);
	}

	return runContourPass(raw_full2, full, frame, lab_tex2);
}

bool CubeDetectorNode::loadYoloModel(cv::dnn::Net &net, const std::string &model_path, const std::string &name)
{
	if (model_path.empty()) {
		RCLCPP_WARN(get_logger(), "[%s] model_path is empty", name.c_str());
		return false;
	}
	if (!std::filesystem::exists(model_path)) {
		RCLCPP_ERROR(get_logger(), "[%s] model file not found: %s", name.c_str(), model_path.c_str());
		return false;
	}

	try {
		net = cv::dnn::readNet(model_path);
		if (_p.yolo_use_cuda) {
			net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
			net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
		} else {
			net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
			net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
		}
		RCLCPP_INFO(get_logger(), "[%s] model loaded: %s", name.c_str(), model_path.c_str());
		return true;
	} catch (const cv::Exception &e) {
		RCLCPP_ERROR(get_logger(), "[%s] failed to load model: %s", name.c_str(), e.what());
		return false;
	}
}

bool CubeDetectorNode::detectBoxRegionYolo(
	const cv::Mat &frame,
	cv::dnn::Net &net,
	const std::string &pipeline_name,
	VictimModel &vm)
{
	if (frame.empty()) return false;
	const int input_size = _p.yolo_input_size;
	cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(input_size, input_size), cv::Scalar(), true, false);

	net.setInput(blob);
	std::vector<cv::Mat> outputs;
	net.forward(outputs, net.getUnconnectedOutLayersNames());
	if (outputs.empty()) return false;

	cv::Mat out = outputs[0];
	if (out.dims == 3) {
		out = out.reshape(1, out.size[1]);
	}

	const float x_factor = static_cast<float>(frame.cols) / static_cast<float>(input_size);
	const float y_factor = static_cast<float>(frame.rows) / static_cast<float>(input_size);

	std::vector<int> class_ids;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;

	for (int i = 0; i < out.rows; ++i) {
		const float *row = out.ptr<float>(i);
		const float obj_conf = row[4];
		if (obj_conf < static_cast<float>(_p.yolo_conf_threshold)) continue;

		int class_id = 0;
		float class_score = 1.0f;
		if (out.cols > 5) {
			cv::Mat scores(1, out.cols - 5, CV_32FC1, const_cast<float *>(row + 5));
			cv::Point class_pt;
			double max_score = 0.0;
			cv::minMaxLoc(scores, nullptr, &max_score, nullptr, &class_pt);
			class_id = class_pt.x;
			class_score = static_cast<float>(max_score);
		}

		const float conf = obj_conf * class_score;
		if (conf < static_cast<float>(_p.yolo_conf_threshold)) continue;
		if (_p.yolo_target_class_id >= 0 && class_id != _p.yolo_target_class_id) continue;

		const float cx = row[0];
		const float cy = row[1];
		const float w = row[2];
		const float h = row[3];
		const int left = static_cast<int>((cx - 0.5f * w) * x_factor);
		const int top = static_cast<int>((cy - 0.5f * h) * y_factor);
		const int width = static_cast<int>(w * x_factor);
		const int height = static_cast<int>(h * y_factor);
		cv::Rect rect(left, top, width, height);
		rect &= cv::Rect(0, 0, frame.cols, frame.rows);
		if (rect.area() <= 0) continue;

		boxes.push_back(rect);
		confidences.push_back(conf);
		class_ids.push_back(class_id);
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(
		boxes,
		confidences,
		static_cast<float>(_p.yolo_conf_threshold),
		static_cast<float>(_p.yolo_nms_threshold),
		indices);

	if (indices.empty()) return false;

	int best_idx = indices[0];
	for (int idx : indices) {
		if (confidences[idx] > confidences[best_idx]) best_idx = idx;
	}
	vm.boxBbox = boxes[best_idx];
	vm.boxCenter = cv::Point2f(
		vm.boxBbox.x + 0.5f * static_cast<float>(vm.boxBbox.width),
		vm.boxBbox.y + 0.5f * static_cast<float>(vm.boxBbox.height));
	vm.pose_pixel_width = static_cast<double>(vm.boxBbox.width);
	vm.pose_center      = vm.boxCenter;

	RCLCPP_DEBUG_THROTTLE(
		get_logger(),
		*get_clock(),
		1000,
		"[%s] det conf=%.3f box=(%d,%d,%d,%d)",
		pipeline_name.c_str(),
		confidences[best_idx],
		vm.boxBbox.x, vm.boxBbox.y, vm.boxBbox.width, vm.boxBbox.height);
	return true;
}

bool CubeDetectorNode::isLockedCandidateValid(const cv::Rect &bbox, const cv::Point2f &center) const
{
	const double dx   = center.x - _lock.center.x;
	const double dy   = center.y - _lock.center.y;
	const double dist = std::sqrt(dx * dx + dy * dy);

	if (dist > _p.lock_max_dist_px) return false;

	const float vis = static_cast<float>(std::max(bbox.width, bbox.height));
	if (_lock.visible_size > 1e-6f && vis < _lock.visible_size * _p.lock_min_size_ratio) {
		return false;
	}

	return true;
}

double CubeDetectorNode::scoreLockedCandidate(
	const cv::Point2f &center, const cv::Rect &bbox, double area) const
{
	const double dx         = center.x - _lock.center.x;
	const double dy         = center.y - _lock.center.y;
	const double dist       = std::sqrt(dx * dx + dy * dy);
	const double vis        = static_cast<double>(std::max(bbox.width, bbox.height));
	const double size_ratio = vis / std::max(1.0f, _lock.visible_size);

	return -_p.score_dist_weight       * dist
	       -_p.score_size_ratio_weight * std::abs(1.0 - size_ratio)
	       +_p.score_area_weight       * area;
}

double CubeDetectorNode::scoreUnlockedCandidate(
	double area, double fill_ratio, double circularity,
	const cv::Point2f &center, const cv::Point2f &img_center) const
{
	const double dx   = center.x - img_center.x;
	const double dy   = center.y - img_center.y;
	const double dist = std::sqrt(dx * dx + dy * dy);

	return  _p.score_area_weight        * area
	       +_p.score_fill_weight        * fill_ratio
	       -_p.score_circularity_weight * circularity
	       -_p.score_center_weight      * dist;
}

bool CubeDetectorNode::detectCircularHandle(
	const cv::Mat &frame, const cv::Rect &boxRegion,
	cv::Point2f &circleCenter, float &radius)
{
	cv::Rect searchRegion = {
		boxRegion.x,
		std::max(0, boxRegion.y - boxRegion.height),
		boxRegion.width,
		boxRegion.height * 2
	};
	if (searchRegion.area() <= 0) return false;

	cv::Mat hsv;
	cv::cvtColor(frame(searchRegion), hsv, cv::COLOR_BGR2HSV);

	cv::Mat mask;
	cv::inRange(hsv,
		cv::Scalar(_p.circle_h_min, _p.circle_s_min, _p.circle_v_min),
		cv::Scalar(_p.circle_h_max, _p.circle_s_max, _p.circle_v_max),
		mask);

	std::vector<cv::Vec3f> circles;
	cv::HoughCircles(mask, circles, cv::HOUGH_GRADIENT,
		1, mask.rows / 2, 100, 20,
		static_cast<int>(_p.circle_min_radius),
		static_cast<int>(_p.circle_max_radius));

	if (circles.empty()) return false;

	circleCenter = { circles[0][0] + searchRegion.x, circles[0][1] + searchRegion.y };
	radius       = circles[0][2];
	return true;
}

bool CubeDetectorNode::validateVictimGeometry(const VictimModel &model)
{
	if (!model.hasCircle) return false;

	const float boxCenterX = model.boxBbox.x + model.boxBbox.width * 0.5f;
	return (model.circleCenter.y <= model.boxBbox.y) &&
	       (std::abs(model.circleCenter.x - boxCenterX) <= model.boxBbox.width * 0.4f);
}

// ═══════════════════════════════════════════════════════════════════
//  Pose estimation (camera optical frame)
// ═══════════════════════════════════════════════════════════════════
void CubeDetectorNode::estimatePose(
	const VictimModel &target, int /*image_width*/, int /*image_height*/,
	double &x, double &y, double &z) const
{
	const double fx = _camera_matrix.at<double>(0, 0);
	const double fy = _camera_matrix.at<double>(1, 1);
	const double cx = _camera_matrix.at<double>(0, 2);
	const double cy = _camera_matrix.at<double>(1, 2);

	const bool        use_pose_geom = target.pose_pixel_width > 1e-6;
	const double      pixel_width =
		use_pose_geom ? target.pose_pixel_width : static_cast<double>(target.boxBbox.width);
	const cv::Point2f cc = use_pose_geom ? target.pose_center : target.boxCenter;

	if (pixel_width < 1.0) {
		x = 0.0;
		y = 0.0;
		z = 0.0;
		return;
	}

	z = (fx * _p.box_width_m) / pixel_width;
	x = (cc.x - cx) * z / fx;
	y = (cc.y - cy) * z / fy;
}

// ═══════════════════════════════════════════════════════════════════
//  Kalman 6-state implementation (hoat dong trong he PX4 LOCAL NED)
// ═══════════════════════════════════════════════════════════════════
static void applyConstantVelocityModel(cv::KalmanFilter &kf, double dt,
                                       double q_acc_x, double q_acc_y, double q_acc_z)
{
	kf.transitionMatrix = cv::Mat::eye(kStateSize, kStateSize, CV_64F);
	kf.transitionMatrix.at<double>(0, 3) = dt;
	kf.transitionMatrix.at<double>(1, 4) = dt;
	kf.transitionMatrix.at<double>(2, 5) = dt;

	const double dt2 = dt * dt;
	const double dt3 = dt2 * dt;
	const double dt4 = dt3 * dt;

	kf.processNoiseCov = cv::Mat::zeros(kStateSize, kStateSize, CV_64F);

	kf.processNoiseCov.at<double>(0, 0) = 0.25 * dt4 * q_acc_x;
	kf.processNoiseCov.at<double>(0, 3) = 0.50 * dt3 * q_acc_x;
	kf.processNoiseCov.at<double>(3, 0) = 0.50 * dt3 * q_acc_x;
	kf.processNoiseCov.at<double>(3, 3) =        dt2 * q_acc_x;

	kf.processNoiseCov.at<double>(1, 1) = 0.25 * dt4 * q_acc_y;
	kf.processNoiseCov.at<double>(1, 4) = 0.50 * dt3 * q_acc_y;
	kf.processNoiseCov.at<double>(4, 1) = 0.50 * dt3 * q_acc_y;
	kf.processNoiseCov.at<double>(4, 4) =        dt2 * q_acc_y;

	kf.processNoiseCov.at<double>(2, 2) = 0.25 * dt4 * q_acc_z;
	kf.processNoiseCov.at<double>(2, 5) = 0.50 * dt3 * q_acc_z;
	kf.processNoiseCov.at<double>(5, 2) = 0.50 * dt3 * q_acc_z;
	kf.processNoiseCov.at<double>(5, 5) =        dt2 * q_acc_z;
}

void CubeDetectorNode::kalmanPredict(const rclcpp::Time &now_ts)
{
	if (!_kalman.initialized) return;

	double dt = 0.0;
	if (_kalman.lastPredictTime.nanoseconds() > 0) {
		dt = (now_ts - _kalman.lastPredictTime).seconds();
	}

	if (dt < 0.0) {
		RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
			"[Kalman] out-of-order predict stamp, skip | dt=%.4f", dt);
		return;
	}

	if (dt > 1e-9) {
		applyConstantVelocityModel(
			_kalman.kf, dt,
			_kalman.q_acc_x, _kalman.q_acc_y, _kalman.q_acc_z);
		_kalman.kf.predict();
	}

	_kalman.lastPredictTime = now_ts;
}

void CubeDetectorNode::kalmanUpdate(double meas_x, double meas_y, double meas_z,
                                    const rclcpp::Time &now_ts)
{
	if (!_kalman.initialized) {
		_kalman.kf.statePost.at<double>(0, 0) = meas_x;
		_kalman.kf.statePost.at<double>(1, 0) = meas_y;
		_kalman.kf.statePost.at<double>(2, 0) = meas_z;
		_kalman.kf.statePost.at<double>(3, 0) = 0.0;
		_kalman.kf.statePost.at<double>(4, 0) = 0.0;
		_kalman.kf.statePost.at<double>(5, 0) = 0.0;

		_kalman.kf.statePre = _kalman.kf.statePost.clone();

		_kalman.initialized     = true;
		_kalman.lastPredictTime = now_ts;

		RCLCPP_INFO(get_logger(),
			"[Kalman] initialized from first NED pose | ned=(%.3f, %.3f, %.3f)",
			meas_x, meas_y, meas_z);
		return;
	}

	double dt = 0.0;
	if (_kalman.lastPredictTime.nanoseconds() > 0) {
		dt = (now_ts - _kalman.lastPredictTime).seconds();
	}

	if (dt < 0.0) {
		RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
			"[Kalman] out-of-order measurement stamp, skip | dt=%.4f", dt);
		return;
	}

	if (dt > 1e-9) {
		applyConstantVelocityModel(
			_kalman.kf, dt,
			_kalman.q_acc_x, _kalman.q_acc_y, _kalman.q_acc_z);
		_kalman.kf.predict();
	}

	cv::Mat measurement(kMeasurementSize, 1, CV_64F);
	measurement.at<double>(0, 0) = meas_x;
	measurement.at<double>(1, 0) = meas_y;
	measurement.at<double>(2, 0) = meas_z;
	_kalman.kf.correct(measurement);

	_kalman.lastPredictTime = now_ts;
}

// ═══════════════════════════════════════════════════════════════════
//  Publishing helpers
// ═══════════════════════════════════════════════════════════════════
geometry_msgs::msg::PoseStamped CubeDetectorNode::buildPoseMsg(
	const std_msgs::msg::Header &header,
	const std::string &frame_id,
	double x,
	double y,
	double z) const
{
	geometry_msgs::msg::PoseStamped msg;
	msg.header          = header;
	msg.header.frame_id = frame_id;

	msg.pose.position.x = x;
	msg.pose.position.y = y;
	msg.pose.position.z = z;

	msg.pose.orientation.w = 1.0;
	msg.pose.orientation.x = 0.0;
	msg.pose.orientation.y = 0.0;
	msg.pose.orientation.z = 0.0;

	return msg;
}

geometry_msgs::msg::TwistStamped CubeDetectorNode::buildVelocityMsg(
	const std_msgs::msg::Header &header,
	const std::string &frame_id,
	double vx,
	double vy,
	double vz) const
{
	geometry_msgs::msg::TwistStamped msg;
	msg.header          = header;
	msg.header.frame_id = frame_id;

	msg.twist.linear.x = vx;
	msg.twist.linear.y = vy;
	msg.twist.linear.z = vz;

	msg.twist.angular.x = 0.0;
	msg.twist.angular.y = 0.0;
	msg.twist.angular.z = 0.0;

	return msg;
}

void CubeDetectorNode::publishPose(
	const rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr &publisher,
	const std_msgs::msg::Header &header,
	const std::string &frame_id,
	double x,
	double y,
	double z) const
{
	if (!publisher) return;
	publisher->publish(buildPoseMsg(header, frame_id, x, y, z));
}

void CubeDetectorNode::publishTargetValid(bool valid)
{
	std_msgs::msg::Bool valid_msg;
	valid_msg.data = valid;
	_target_valid_pub->publish(valid_msg);
}

void CubeDetectorNode::publishFilteredOutput(const std_msgs::msg::Header &header)
{
	if (!_kalman.initialized) {
		return;
	}

	const auto filteredPose = buildPoseMsg(
		header,
		_p.world_frame_id,
		_kalman.x(),
		_kalman.y(),
		_kalman.z());

	const auto filteredVelocity = buildVelocityMsg(
		header,
		_p.world_frame_id,
		_kalman.vx(),
		_kalman.vy(),
		_kalman.vz());

	// Topic cu, giu tuong thich voi controller dang subscribe /cube_detector/target_pose
	_target_pose_pub->publish(filteredPose);

	// Topic moi, ro nghia hon de danh gia/tune Kalman
	_target_pose_world_filtered_pub->publish(filteredPose);
	_target_velocity_world_filtered_pub->publish(filteredVelocity);
}

// ═══════════════════════════════════════════════════════════════════
//  HSV debug image publisher
// ═══════════════════════════════════════════════════════════════════
void CubeDetectorNode::publishHsvDebug(
	const cv::Mat &frame,
	const VictimModel &victim,
	const std_msgs::msg::Header &header)
{
	if (_hsv_debug_pub->get_subscription_count() == 0) return;

	const auto maskToGreenBgr = [](const cv::Mat &m) -> cv::Mat {
		cv::Mat out(m.size(), CV_8UC3, cv::Scalar(0, 0, 0));
		if (!m.empty()) {
			out.setTo(cv::Scalar(0, 220, 0), m);
		}
		return out;
	};

	const auto annotateRoiAndBox = [&](cv::Mat &panel) {
		if (_last_debug_roi.width > 0 && _last_debug_roi.height > 0) {
			cv::rectangle(panel, _last_debug_roi, cv::Scalar(0, 180, 255), 2, cv::LINE_AA);
		}
		if (victim.valid && victim.boxBbox.area() > 0) {
			cv::rectangle(panel, victim.boxBbox, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
			cv::circle(panel,
				cv::Point(static_cast<int>(victim.boxCenter.x),
				          static_cast<int>(victim.boxCenter.y)),
				5, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
		}
	};

	const auto annotateOverlayPanel = [&](cv::Mat &right_panel, const cv::Mat &mask) {
		cv::Mat blurred_bg;
		cv::GaussianBlur(right_panel, blurred_bg, cv::Size(0, 0), 3.0);
		cv::Mat mask_inv;
		cv::bitwise_not(mask, mask_inv);
		blurred_bg.copyTo(right_panel, mask_inv);
		right_panel.setTo(cv::Scalar(0, 140, 255), mask);
		if (victim.valid && victim.boxBbox.area() > 0) {
			if (_last_debug_roi.width > 0 && _last_debug_roi.height > 0) {
				cv::rectangle(right_panel, _last_debug_roi, cv::Scalar(0, 180, 255), 2, cv::LINE_AA);
			}
			cv::rectangle(right_panel, victim.boxBbox, cv::Scalar(0, 80, 255), 2, cv::LINE_AA);
			cv::circle(right_panel,
				cv::Point(static_cast<int>(victim.boxCenter.x),
				          static_cast<int>(victim.boxCenter.y)),
				5, cv::Scalar(0, 80, 255), -1, cv::LINE_AA);
		}
		cv::putText(right_panel,
			victim.valid ? "[DETECTED]" : "[SEARCHING]",
			cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.85,
			victim.valid ? cv::Scalar(0, 200, 80) : cv::Scalar(80, 80, 255),
			2, cv::LINE_AA);
	};

	cv::Mat mask_final;
	if (!_dbg_algorithm_mask_full.empty() && _dbg_algorithm_mask_full.size() == frame.size()) {
		mask_final = _dbg_algorithm_mask_full;
	} else if (!makeAlgorithmMask(frame, mask_final)) {
		return;
	}

	// YOLO: giu layout 2 panel (khong co pipeline LAB tren detection).
	if (_p.detector_mode != kModeAlgorithm) {
		cv::Mat left_panel = maskToGreenBgr(mask_final);
		annotateRoiAndBox(left_panel);
		cv::putText(left_panel, "Mask (fallback LAB)",
			cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.85,
			cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
		cv::putText(left_panel, std::string("Mode=") + _p.detector_mode,
			cv::Point(10, 55), cv::FONT_HERSHEY_SIMPLEX, 0.55,
			cv::Scalar(200, 200, 200), 1, cv::LINE_AA);

		cv::Mat right_panel = frame.clone();
		annotateOverlayPanel(right_panel, mask_final);

		cv::Mat combined;
		cv::hconcat(left_panel, right_panel, combined);
		cv::line(combined,
			cv::Point(frame.cols, 0),
			cv::Point(frame.cols, frame.rows),
			cv::Scalar(100, 100, 100), 2);

		cv_bridge::CvImage cv_img;
		cv_img.header   = header;
		cv_img.encoding = sensor_msgs::image_encodings::BGR8;
		cv_img.image    = combined;
		_hsv_debug_pub->publish(*cv_img.toImageMsg());
		return;
	}

	// Algorithm: luoi 2x3 — tung buoc xu ly tren full frame (cung canvas voi detection cuoi).
	cv::Mat lab_bgr, mask_color, mask_morph;
	makeAlgorithmMaskStagesOnBgr(frame, lab_bgr, mask_color, mask_morph);

	cv::Mat p_lab = lab_bgr.empty() ? cv::Mat(frame.size(), CV_8UC3, cv::Scalar(0, 0, 0)) : lab_bgr.clone();
	cv::putText(p_lab, "1: LAB + CLAHE (BGR)",
		cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.72,
		cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

	cv::Mat p_color = maskToGreenBgr(mask_color);
	annotateRoiAndBox(p_color);
	cv::putText(p_color, "2: Mask mau (pre-morph)",
		cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.72,
		cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

	cv::Mat p_morph = maskToGreenBgr(mask_morph);
	annotateRoiAndBox(p_morph);
	cv::putText(p_morph, "3: Sau morphology (full frame)",
		cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.58,
		cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

	cv::Mat mask_pre_temp = mask_morph;
	if (!_dbg_algorithm_mask_pre_temporal.empty() &&
	    _dbg_algorithm_mask_pre_temporal.size() == frame.size()) {
		mask_pre_temp = _dbg_algorithm_mask_pre_temporal;
	}
	cv::Mat p_pre_t = maskToGreenBgr(mask_pre_temp);
	annotateRoiAndBox(p_pre_t);
	cv::putText(p_pre_t, "4: Truoc temporal (mask detection)",
		cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.55,
		cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

	cv::Mat p_final = maskToGreenBgr(mask_final);
	annotateRoiAndBox(p_final);
	cv::putText(p_final, "5: Sau temporal (vao contour)",
		cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.65,
		cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

	cv::Mat p_overlay = frame.clone();
	annotateOverlayPanel(p_overlay, mask_final);
	cv::putText(p_overlay, "6: Overlay",
		cv::Point(10, 55), cv::FONT_HERSHEY_SIMPLEX, 0.72,
		cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

	{
		std::ostringstream ss;
		ss << "Mode=" << _p.detector_mode
		   << " LAB L" << _p.lab_l_min << "-" << _p.lab_l_max
		   << " a" << _p.lab_a_min << "-" << _p.lab_a_max
		   << " b" << _p.lab_b_min << "-" << _p.lab_b_max;
		cv::putText(p_overlay, ss.str(),
			cv::Point(10, frame.rows - 14), cv::FONT_HERSHEY_SIMPLEX, 0.48,
			cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
	}

	cv::Mat row1, row2, combined;
	cv::hconcat(p_lab, p_color, row1);
	cv::hconcat(row1, p_morph, row1);
	cv::hconcat(p_pre_t, p_final, row2);
	cv::hconcat(row2, p_overlay, row2);
	cv::vconcat(row1, row2, combined);

	cv::line(combined,
		cv::Point(0, frame.rows),
		cv::Point(combined.cols, frame.rows),
		cv::Scalar(80, 80, 80), 2);
	cv::line(combined,
		cv::Point(frame.cols, 0),
		cv::Point(frame.cols, combined.rows),
		cv::Scalar(80, 80, 80), 2);
	cv::line(combined,
		cv::Point(frame.cols * 2, 0),
		cv::Point(frame.cols * 2, combined.rows),
		cv::Scalar(80, 80, 80), 2);

	cv_bridge::CvImage cv_img;
	cv_img.header   = header;
	cv_img.encoding = sensor_msgs::image_encodings::BGR8;
	cv_img.image    = combined;
	_hsv_debug_pub->publish(*cv_img.toImageMsg());
}

// ═══════════════════════════════════════════════════════════════════
//  State management
// ═══════════════════════════════════════════════════════════════════
void CubeDetectorNode::resetLockState()
{
	_lock.reset();
	_last_seen.reset();
	_kalman.reset();

	_has_last_pose = false;
	_last_x = _last_y = _last_z = 0.0;

	_mask_temporal_prev.release();
	_dbg_algorithm_mask_full.release();
	_dbg_algorithm_mask_pre_temporal.release();
	_last_debug_roi = cv::Rect();
}

// ═══════════════════════════════════════════════════════════════════
//  Debug annotation
// ═══════════════════════════════════════════════════════════════════
void CubeDetectorNode::annotateImage(
	cv_bridge::CvImagePtr image, const VictimModel &target,
	double x, double y, double z,
	const std::string &perf_text) const
{
	cv::Mat &img = image->image;

	if (target.boxBbox.area() > 0) {
		cv::rectangle(img, target.boxBbox, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
		cv::circle(img, cv::Point(target.boxCenter), 5, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
	}

	cv::rectangle(img, cv::Point(8, 8), cv::Point(340, 260), cv::Scalar(40, 40, 40), -1);

	const auto putRow = [&](const std::string &text, int row, cv::Scalar color) {
		cv::putText(img, text, cv::Point(15, 30 + row * 35),
			cv::FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv::LINE_AA);
	};

	const std::string status = std::string("Box: ") + (target.valid ? "LOCKED" : "SEARCHING");
	cv::putText(img, status, cv::Point(15, 30),
		cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

	const cv::Scalar pose_color = _has_last_pose ? cv::Scalar(255, 255, 0) : cv::Scalar(180, 180, 180);

	auto fmtPos = [&](const std::string &label, double val) -> std::string {
		if (!_has_last_pose) return label + ": N/A";
		std::ostringstream ss;
		ss << label << ": " << std::fixed << std::setprecision(3) << val << " m";
		return ss.str();
	};
	auto fmtVel = [&](const std::string &label, double val) -> std::string {
		if (!_kalman.initialized) return label + ": N/A";
		std::ostringstream ss;
		ss << label << ": " << std::fixed << std::setprecision(3) << val << " m/s";
		return ss.str();
	};

	putRow(fmtPos("[NED]X", x),  1, pose_color);
	putRow(fmtPos("[NED]Y", y),  2, pose_color);
	putRow(fmtPos("[NED]Z", z),  3, pose_color);

	const cv::Scalar vel_color = cv::Scalar(100, 220, 255);
	putRow(fmtVel("VX", _kalman.vx()), 4, vel_color);
	putRow(fmtVel("VY", _kalman.vy()), 5, vel_color);
	putRow(fmtVel("VZ", _kalman.vz()), 6, vel_color);

	if (!perf_text.empty()) {
		cv::putText(img, perf_text, cv::Point(15, 30 + 7 * 35),
			cv::FONT_HERSHEY_SIMPLEX, 0.70, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
	}
}

// ═══════════════════════════════════════════════════════════════════
//  Entry point
// ═══════════════════════════════════════════════════════════════════
int main(int argc, char *argv[])
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<CubeDetectorNode>());
	rclcpp::shutdown();
	return 0;
}