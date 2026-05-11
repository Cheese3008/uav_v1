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
		float circle_area_ratio{0.0f};
		float radial_error{0.0f};
		int approx_vertices{0};
		std::vector<cv::Point> contour;
	};

	// ===== ROS callbacks =====
	void loadParameters();
	void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
	void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
	void ring_detect_reset_callback(const std_msgs::msg::String::SharedPtr msg);

	// ===== Single body-frame Kalman =====
	// Chỉ dùng một cv::KalmanFilter duy nhất:
	// state = [x, y, z, vx, vy, vz]^T
	// measurement = [x, y, z]^T
	// Measurement đầu vào là vị trí sau khi đổi từ camera optical frame sang body/drone frame.
	// Quy ước body XYZ: x = phía trước drone, y = bên phải drone, z = xuống dưới.
	void configureBodyKalman();
	void initBodyKalman(const Eigen::Vector3d &body_measurement);
	void predictBodyKalman(double dt_s);
	void updateBodyKalman(const Eigen::Vector3d &body_measurement);
	void resetBodyKalman();
	Eigen::Vector3d getBodyKalmanPosition() const;
	Eigen::Vector3d getBodyKalmanVelocity() const;

	Eigen::Vector3d cameraOpticalToBodyXyz(const Eigen::Vector3d &camera_position) const;
	Eigen::Vector3d bodyXyzToCameraOptical(const Eigen::Vector3d &body_position) const;
	bool projectBodyKalmanToImage(cv::Point2f &pixel, double &depth_z) const;

	void publishCameraRawTarget(
		const std_msgs::msg::Header &header,
		double camera_x,
		double camera_y,
		double camera_z);


	void publishBodyFilteredTarget(const std_msgs::msg::Header &header);

	void publishBodyTargetFromMeasurement(
		const std_msgs::msg::Header &header,
		double camera_x,
		double camera_y,
		double camera_z);

	bool publishBodyTargetPredictionOnly(const std_msgs::msg::Header &header);

	// ===== Detection =====
	cv::Mat build_neutral_gray_mask(const cv::Mat &bgr) const;

	cv::Mat make_color_detect_debug_image(
		const cv::Mat &bgr_frame) const;

	cv::Mat make_side_by_side_debug_image(
		const cv::Mat &left_image,
		const cv::Mat &right_image) const;

	cv::Mat make_labeled_debug_panel(
		const cv::Mat &image,
		const std::string &label) const;

	cv::Mat make_pipeline_debug_image(
		const cv::Mat &annotated_frame) const;

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

	// ===== Neutral gray color mask / debug params =====
	bool _param_use_white_mask{true};

	// Dùng để bật/tắt khung bên phải trong ảnh /ring_detect/image_proc.
	// true  : ảnh image_proc = detection debug + color detect debug
	// false : ảnh image_proc chỉ có detection debug
	bool _param_publish_color_detect_debug{false};

	// Bật/tắt publish ảnh /ring_detect/image_proc.
	// false: node vẫn detect và publish data topic, nhưng không xuất ảnh để nhẹ CPU/network.
	bool _param_publish_image{true};

	// Bật ảnh debug nhiều lớp trên cùng /ring_detect/image_proc:
	// annotated output, Y channel, neutral mask, detector input, Canny edge, contour overlay.
	bool _param_publish_pipeline_debug{false};
	bool _param_debug_draw_rejected_contours{true};
	int _param_debug_panel_width{520};

	// Với Canny edge, contour của vòng thường là viền mỏng nên area/circle_area có thể nhỏ.
	// Tắt filter này sẽ giúp nhận vòng to/dày tốt hơn, vẫn lọc tròn bằng vertices/ellipse/radial_error.
	bool _param_use_circle_area_ratio_filter{false};

	// Ngưỡng BGR để nhận màu trung tính từ trắng tới xám đen, không dùng HSV.
	// gray_min/gray_max: giới hạn độ sáng sau khi quy đổi gray từ BGR.
	// gray_color_diff_max: chênh lệch max(B,G,R)-min(B,G,R) không được quá lớn.
	int _param_gray_min{45};
	int _param_gray_max{255};
	int _param_gray_color_diff_max{55};
	int _param_gray_morph_kernel{3};

	// Detector kiểu YUV tham khảo: dùng Y cho sáng/tối, U/V gần 128 để giữ màu trung tính.
	bool _param_use_yuv_neutral_mask{true};
	int _param_yuv_u_center{128};
	int _param_yuv_v_center{128};
	int _param_yuv_uv_diff_max{35};
	int _param_yuv_median_kernel{5};

	// Giữ tên tham số cũ để YAML cũ không làm node lỗi khi load.
	int _param_white_min{150};
	int _param_white_color_diff_max{45};
	int _param_white_morph_kernel{3};

	// ===== Simple center-ring detection params =====
	// Chỉ xử lý vùng gần tâm ảnh để giảm tải và tránh bắt nhầm nền ở rìa ảnh.
	double _param_center_roi_ratio{0.70};
	double _param_max_center_distance_ratio{0.42};
	double _param_min_fill_ratio{0.12};
	double _param_max_fill_ratio{1.25};

	// Lọc hình tròn thật để loại hình vuông/trắng/xám gần tâm.
	double _param_min_circle_area_ratio{0.72};
	double _param_max_circle_area_ratio{1.20};
	double _param_max_radial_error{0.20};
	int _param_min_approx_vertices{8};
	double _param_approx_epsilon_ratio{0.018};

	// Detector cạnh kiểu tham khảo Python:
	// neutral gray mask -> bilateral/blur nhẹ -> Canny -> contour -> approxPolyDP.
	bool _param_use_canny_circle_detector{true};
	bool _param_use_gray_binary_threshold{false};
	int _param_gray_binary_threshold{45};
	bool _param_use_bilateral_filter{true};
	int _param_bilateral_d{5};
	double _param_bilateral_sigma_color{175.0};
	double _param_bilateral_sigma_space{175.0};
	double _param_canny_low{75.0};
	double _param_canny_high{200.0};
	int _param_edge_dilate_kernel{3};
	double _param_min_ellipse_axis_ratio{0.78};
	double _param_max_ellipse_center_shift_ratio{0.18};

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
	double _param_pass_x_thresh{0.35};
	double _param_pass_min_visible_size{220.0};
	double _param_body_kf_hold_timeout_s{0.30};
	double _param_reset_timeout_s{0.50};

	// ===== Body XYZ transform params =====
	double _param_camera_offset_x{0.09};
	double _param_camera_offset_y{0.0};
	double _param_camera_offset_z{0.0};

	// ===== Single body-frame Kalman params =====
	// q_acc là nhiễu gia tốc cho model vận tốc không đổi.
	// r_* là phương sai nhiễu đo vị trí theo từng trục body XYZ.
	// x = trước, y = phải, z = xuống.
	double _param_body_kf_q_acc{0.80};
	double _param_body_kf_r_x{0.02};
	double _param_body_kf_r_y{0.04};
	double _param_body_kf_r_z{0.02};

	bool _param_track_use_kf_gate{true};
	double _param_track_kf_pixel_gate_px{90.0};
	double _param_track_kf_depth_ratio_gate{0.45};
	bool _param_publish_reset_status{false};

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
	Eigen::Vector3d _last_body_filtered{Eigen::Vector3d::Zero()};
	bool _has_last_body_filtered{false};

	// ===== Single body-frame Kalman state =====
	cv::KalmanFilter _body_kf;
	bool _body_kf_initialized{false};
	Eigen::Vector3d _body_kf_position{Eigen::Vector3d::Zero()};
	Eigen::Vector3d _body_kf_velocity{Eigen::Vector3d::Zero()};
	rclcpp::Time _body_kf_last_time{0, 0, RCL_ROS_TIME};
	rclcpp::Time _last_measurement_time{0, 0, RCL_ROS_TIME};

	// ===== Far candidate confirmation =====
	bool _has_far_candidate_memory{false};
	cv::Point2f _far_candidate_center{0.0f, 0.0f};
	float _far_candidate_size{0.0f};
	int _far_candidate_count{0};

	// ===== Pipeline debug cache =====
	cv::Mat _debug_neutral_mask;
	cv::Mat _debug_y_channel;
	cv::Mat _debug_detector_binary;
	cv::Mat _debug_edge;
	cv::Mat _debug_candidate_overlay;
	int _debug_raw_contour_count{0};
	int _debug_accepted_candidate_count{0};

	// ===== ROS interfaces =====
	rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _image_sub;
	rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr _camera_info_sub;
	rclcpp::Subscription<std_msgs::msg::String>::SharedPtr _ring_detect_reset_sub;

	rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _image_pub;
	rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr _target_valid_pub;
	rclcpp::Publisher<std_msgs::msg::String>::SharedPtr _reset_status_pub;
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _target_pose_camera_raw_pub;
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _target_error_body_filtered_pub;
	rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr _target_velocity_body_filtered_pub;

	// ===== Camera model =====
	cv::Mat _camera_matrix;
	cv::Mat _dist_coeffs;
	bool _has_camera_info{false};
};