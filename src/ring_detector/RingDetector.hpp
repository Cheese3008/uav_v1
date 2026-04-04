#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <std_msgs/msg/string.hpp>

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

	// ===== SSM dimensions =====
	static constexpr int kSsmStateDim = 8;
	static constexpr int kSsmMeasDim = 4;

	// ===== ROS callbacks =====
	void loadParameters();
	void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
	void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

	// ===== SSM init =====
	void initSsmMatrices();

	// ===== Detection / selection =====
	std::vector<RingCandidate> detect_ring_candidates(const cv::Mat &frame, cv::Mat &edges);
	std::vector<RingCandidate> detect_ring_candidates_far(const cv::Mat &frame, cv::Mat &edges);

	bool select_main_ring(const std::vector<RingCandidate> &candidates, RingCandidate &best) const;
	bool select_far_ring(const std::vector<RingCandidate> &candidates, RingCandidate &best) const;

	bool find_locked_ring(const std::vector<RingCandidate> &candidates, RingCandidate &best) const;
	bool find_predicted_ring(const std::vector<RingCandidate> &candidates, RingCandidate &best) const;
	bool confirm_far_candidate(const RingCandidate &candidate);
	void ring_detect_reset_callback(const std_msgs::msg::String::SharedPtr msg);

	rclcpp::Subscription<std_msgs::msg::String>::SharedPtr _ring_detect_reset_sub;

	// ===== Pose estimation =====
	void estimate_ring_pose_from_image(
		const RingCandidate &target,
		int image_width,
		int image_height,
		double &x,
		double &y,
		double &z) const;

	void estimate_pose_from_ssm_state(
		double &x,
		double &y,
		double &z) const;

	// ===== Tracking state =====
	bool is_passed_ring() const;
	bool should_commit_pass_through() const;
	bool has_last_seen_pose() const;

	void reset_lock_state();
	void init_track_state(const RingCandidate &target, const rclcpp::Time &now_ts);
	void predict_track_state(const rclcpp::Time &now_ts);
	void update_track_state_from_measurement(
		const RingCandidate &target,
		const rclcpp::Time &now_ts);

	void get_predicted_track_measurement(
		cv::Point2f &center,
		float &visible_size) const;

	// ===== Debug image =====
	void annotate_image(
		cv_bridge::CvImagePtr image,
		const RingCandidate &target,
		double x,
		double y,
		double z) const;

private:
	// ===== Topics =====
	std::string _image_topic;
	std::string _camera_info_topic;

	// ===== Detection params (near mode) =====
	double _param_ring_diameter_m{1.8};
	double _param_min_area{700.0};
	double _param_min_radius_px{12.0};
	double _param_min_visible_size{60.0};
	double _param_circularity_min{0.05};
	double _param_aspect_min{0.30};
	double _param_aspect_max{3.50};

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
	double _param_lock_alpha{0.75};

	// ===== Hold / pass params =====
	int _param_hold_max_missed{15};
	double _param_pass_z_thresh{1.2};
	double _param_pass_min_visible_size{220.0};

	// ===== Timeout params =====
	double _param_soft_hold_timeout_s{0.4};
	double _param_reset_timeout_s{0.3};
	double _param_last_seen_hold_timeout_s{1.5};

	// ===== Predictive reacquire params =====
	double _param_reacquire_gate_px{75.0};
	double _param_reacquire_size_ratio{0.20};
	double _param_reacquire_cost_pos{4.0};
	double _param_reacquire_cost_size{60.0};
	double _param_reacquire_cost_area{0.01};

	// ===== Pass commit params =====
	double _param_pass_commit_timeout_s{0.8};
	double _param_pass_commit_z_thresh{1.8};
	double _param_pass_commit_visible_size{180.0};
	double _param_pass_commit_center_px{90.0};
	int _param_pass_commit_max_missed{30};
	double _param_pass_commit_forward_speed_mps{1.2};

	// ===== SSM params =====
	double _param_ssm_q_pos{20.0};
	double _param_ssm_q_vel{80.0};
	double _param_ssm_q_size{25.0};
	double _param_ssm_q_size_vel{60.0};
	double _param_ssm_q_pose_xy{0.15};

	double _param_ssm_r_pos{18.0};
	double _param_ssm_r_size{12.0};
	double _param_ssm_r_pose_x{0.10};

	double _param_track_confidence_decay{0.90};
	double _param_track_confidence_gain{0.25};
	double _param_track_confidence_min_for_hold{0.15};

	// ===== Lock state =====
	bool _locked{false};
	bool _in_hold{false};

	cv::Point2f _lock_center{0.0f, 0.0f};
	float _lock_visible_size{0.0f};

	int _lock_missed{0};
	int _hold_missed{0};

	RingCandidate _last_good_target;
	bool _has_last_good_target{false};

	double _last_x{0.0};
	double _last_y{0.0};
	double _last_z{0.0};

	// ===== Persistent target memory =====
	bool _has_track_memory{false};
	bool _soft_lost{false};

	cv::Point2f _track_center{0.0f, 0.0f};
	cv::Point2f _track_velocity{0.0f, 0.0f};
	float _track_size{0.0f};
	float _track_size_velocity{0.0f};

	rclcpp::Time _last_track_time{0, 0, RCL_ROS_TIME};
	rclcpp::Time _last_match_time{0, 0, RCL_ROS_TIME};

	// ===== SSM state =====
	Eigen::Matrix<double, kSsmStateDim, 1> _ssm_state;
	Eigen::Matrix<double, kSsmStateDim, kSsmStateDim> _ssm_P;
	Eigen::Matrix<double, kSsmStateDim, kSsmStateDim> _ssm_Q;
	Eigen::Matrix<double, kSsmMeasDim, kSsmMeasDim> _ssm_R;
	Eigen::Matrix<double, kSsmMeasDim, kSsmStateDim> _ssm_H;

	bool _ssm_initialized{false};
	double _track_confidence{0.0};

	// ===== Last seen pose hold =====
	bool _last_seen_timer_started{false};
	rclcpp::Time _last_seen_lost_start_time{0, 0, RCL_ROS_TIME};

	double _last_seen_x{0.0};
	double _last_seen_y{0.0};
	double _last_seen_z{0.0};
	bool _has_last_seen_pose{false};

	// ===== Far candidate confirmation =====
	bool _has_far_candidate_memory{false};
	cv::Point2f _far_candidate_center{0.0f, 0.0f};
	float _far_candidate_size{0.0f};
	int _far_candidate_count{0};

	// ===== ROS interfaces =====
	rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _image_sub;
	rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr _camera_info_sub;

	rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _image_pub;
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _target_pose_pub;
	rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr _target_valid_pub;
	rclcpp::Publisher<std_msgs::msg::String>::SharedPtr _reset_pub;

	// ===== Camera model =====
	cv::Mat _camera_matrix;
	cv::Mat _dist_coeffs;
	bool _has_camera_info{false};

	// ===== Detection time =====
	rclcpp::Time _last_detect_time{0, 0, RCL_ROS_TIME};
	bool _has_detect_time{false};
};