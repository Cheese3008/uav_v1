#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/header.hpp>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>

class RingDetectorNode : public rclcpp::Node
{
public:
	RingDetectorNode();

private:
	struct RingCandidate
	{
		cv::Point2f center;
		float radius{0.0f};
		float area{0.0f};
		float circularity{0.0f};
		float bbox_w{0.0f};
		float bbox_h{0.0f};
		float bbox_area{0.0f};
		float visible_size{0.0f};
		std::vector<cv::Point> contour;
	};

	// ===== Body Kalman dimensions =====
	static constexpr int kBodyKalmanStateDim = 6;
	static constexpr int kBodyKalmanMeasDim = 3;

	// ===== ROS callbacks =====
	void loadParameters();
	void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
	void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
	void ring_detect_reset_callback(const std_msgs::msg::String::SharedPtr msg);

	// ===== Body Kalman =====
	void initBodyKalmanMatrices();
	void initBodyKalman(const Eigen::Vector3d &measurement);
	void predictBodyKalman(double dt_s);
	void updateBodyKalman(const Eigen::Vector3d &measurement);
	void resetBodyKalman();

	Eigen::Vector3d cameraOpticalToFrontBody(const Eigen::Vector3d &camera_position) const;

	void publishCameraRawTarget(
		const std_msgs::msg::Header &header,
		double camera_x,
		double camera_y,
		double camera_z);

	void publishBodyRawTarget(
		const std_msgs::msg::Header &header,
		const Eigen::Vector3d &body_raw);

	void publishBodyFilteredTarget(const std_msgs::msg::Header &header);

	void publishBodyTargetFromMeasurement(
		const std_msgs::msg::Header &header,
		double camera_x,
		double camera_y,
		double camera_z);

	bool publishBodyTargetPredictionOnly(const std_msgs::msg::Header &header);

	// ===== Detection =====
	cv::Mat build_white_mask(const cv::Mat &bgr) const;

	cv::Mat make_color_detect_debug_image(
		const cv::Mat &bgr_frame) const;

	cv::Mat make_side_by_side_debug_image(
		const cv::Mat &left_image,
		const cv::Mat &right_image) const;

	std::vector<RingCandidate> detect_ring_candidates(const cv::Mat &frame, cv::Mat &edges);
	std::vector<RingCandidate> detect_ring_candidates_far(const cv::Mat &frame, cv::Mat &edges);

	bool select_main_ring(const std::vector<RingCandidate> &candidates, RingCandidate &best) const;
	bool select_far_ring(const std::vector<RingCandidate> &candidates, RingCandidate &best) const;
	bool find_locked_ring(const std::vector<RingCandidate> &candidates, RingCandidate &best) const;
	bool confirm_far_candidate(const RingCandidate &candidate);

	// ===== Pose estimation =====
	void estimate_ring_pose_from_image(
		const RingCandidate &target,
		int image_width,
		int image_height,
		double &x,
		double &y,
		double &z) const;

	// ===== Tracking / hold state =====
	bool is_passed_ring() const;
	bool can_hold_with_body_kalman(const rclcpp::Time &now_ts) const;
	void reset_lock_state();

	// ===== Debug image =====
	void annotate_image(
		cv_bridge::CvImagePtr image,
		const RingCandidate &target,
		double camera_x,
		double camera_y,
		double camera_z,
		const std::string &state_text) const;

private:
	// ===== Topics =====
	std::string _image_topic;
	std::string _camera_info_topic;

	std::string _processed_image_topic;
	std::string _target_valid_topic;
	std::string _reset_status_topic;
	std::string _ring_detect_reset_topic;

	std::string _target_pose_camera_raw_topic;
	std::string _target_error_body_raw_topic;
	std::string _target_error_body_filtered_topic;
	std::string _target_velocity_body_filtered_topic;

	std::string _target_frame_id;
	std::string _body_frame_id{"base_link_frd"};

	// ===== Detection params (near mode) =====
	double _param_ring_diameter_m{1.8};
	double _param_min_area{700.0};
	double _param_min_radius_px{12.0};
	double _param_min_visible_size{60.0};
	double _param_circularity_min{0.05};
	double _param_aspect_min{0.30};
	double _param_aspect_max{3.50};

	// ===== White color mask / debug params =====
	bool _param_use_white_mask{true};

	// Dùng để bật/tắt khung bên phải trong ảnh /ring_detect/image_proc.
	// true  : ảnh image_proc = detection debug + color detect debug
	// false : ảnh image_proc chỉ có detection debug
	bool _param_publish_color_detect_debug{true};

	int _param_white_s_max{45};
	int _param_white_v_min{170};
	int _param_white_morph_kernel{3};

	// ===== Far detection params =====
	double _param_far_min_area{120.0};
	double _param_far_min_radius_px{5.0};
	double _param_far_min_visible_size{20.0};
	double _param_far_circularity_min{0.02};
	double _param_far_aspect_min{0.35};
	double _param_far_aspect_max{3.00};
	double _param_far_canny_low{35.0};
	double _param_far_canny_high{110.0};
	double _param_far_roi_margin_ratio{0.01};
	int _param_far_confirm_frames{3};
	double _param_far_confirm_pos_gate_px{45.0};
	double _param_far_confirm_size_ratio{0.45};

	// ===== Lock params =====
	double _param_lock_max_dist{180.0};
	double _param_lock_min_size_ratio{0.45};
	int _param_max_lock_missed{30};

	// ===== Hold / pass params =====
	int _param_hold_max_missed{15};
	double _param_pass_forward_thresh{0.35};
	double _param_pass_min_visible_size{220.0};
	double _param_body_kf_hold_timeout_s{0.30};
	double _param_reset_timeout_s{0.50};

	// ===== Front fixed camera transform params =====
	double _param_camera_offset_x{0.09};
	double _param_camera_offset_y{0.0};
	double _param_camera_offset_z{0.0};

	// ===== Body Kalman params =====
	double _param_body_kf_q_pos{0.02};
	double _param_body_kf_q_vel{0.10};
	double _param_body_kf_r_x{0.04};
	double _param_body_kf_r_y{0.02};
	double _param_body_kf_r_z{0.02};

	// ===== Lock state =====
	bool _locked{false};
	cv::Point2f _lock_center{0.0f, 0.0f};
	float _lock_visible_size{0.0f};
	int _lock_missed{0};
	int _hold_missed{0};

	RingCandidate _last_good_target;
	bool _has_last_good_target{false};

	double _last_camera_x{0.0};
	double _last_camera_y{0.0};
	double _last_camera_z{0.0};
	Eigen::Vector3d _last_body_raw{Eigen::Vector3d::Zero()};
	bool _has_last_body_raw{false};

	// ===== Body Kalman state =====
	Eigen::Matrix<double, kBodyKalmanStateDim, 1> _body_kf_state;
	Eigen::Matrix<double, kBodyKalmanStateDim, kBodyKalmanStateDim> _body_kf_P;
	Eigen::Matrix<double, kBodyKalmanStateDim, kBodyKalmanStateDim> _body_kf_Q;
	Eigen::Matrix<double, kBodyKalmanMeasDim, kBodyKalmanMeasDim> _body_kf_R;
	Eigen::Matrix<double, kBodyKalmanMeasDim, kBodyKalmanStateDim> _body_kf_H;
	bool _body_kf_initialized{false};
	rclcpp::Time _body_kf_last_time{0, 0, RCL_ROS_TIME};
	rclcpp::Time _last_measurement_time{0, 0, RCL_ROS_TIME};

	// ===== Far candidate confirmation =====
	bool _has_far_candidate_memory{false};
	cv::Point2f _far_candidate_center{0.0f, 0.0f};
	float _far_candidate_size{0.0f};
	int _far_candidate_count{0};

	// ===== ROS interfaces =====
	rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _image_sub;
	rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr _camera_info_sub;
	rclcpp::Subscription<std_msgs::msg::String>::SharedPtr _ring_detect_reset_sub;

	rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _image_pub;
	rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr _target_valid_pub;
	rclcpp::Publisher<std_msgs::msg::String>::SharedPtr _reset_status_pub;
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _target_pose_camera_raw_pub;
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _target_error_body_raw_pub;
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _target_error_body_filtered_pub;
	rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr _target_velocity_body_filtered_pub;

	// ===== Camera model =====
	cv::Mat _camera_matrix;
	cv::Mat _dist_coeffs;
	bool _has_camera_info{false};
};