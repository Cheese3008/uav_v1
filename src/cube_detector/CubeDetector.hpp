#pragma once

#include <chrono>
#include <string>

#include <Eigen/Core>

#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>

#include "FrameTransformer.hpp"

class CubeDetectorNode : public rclcpp::Node
{
public:
	CubeDetectorNode();

private:
	// ───────────────────────── Inner types ──────────────────────────
	struct VictimModel
	{
		bool        valid        = false;
		bool        hasCircle    = false;
		float       confidence   = 0.0f;
		cv::Rect    boxBbox;
		cv::Point2f boxCenter;
		cv::Point2f circleCenter;
		float       circleRadius = 0.0f;
	};

	/**
	 * Mô tả:
	 *     Kalman 6-state cho object trong world frame.
	 *
	 * State:
	 *     [x, y, z, vx, vy, vz]^T
	 *
	 * Measurement:
	 *     [x, y, z]^T đã được transform từ camera optical frame sang PX4 local NED frame.
	 */
	struct KalmanState
	{
		cv::KalmanFilter kf;

		double q_acc_x = 0.01;
		double q_acc_y = 0.01;
		double q_acc_z = 0.01;

		double r_pos_x = 0.05;
		double r_pos_y = 0.05;
		double r_pos_z = 0.05;

		bool initialized = false;

		rclcpp::Time lastPredictTime = rclcpp::Time(0, 0, RCL_ROS_TIME);

		double x()  const { return kf.statePost.at<double>(0, 0); }
		double y()  const { return kf.statePost.at<double>(1, 0); }
		double z()  const { return kf.statePost.at<double>(2, 0); }
		double vx() const { return kf.statePost.at<double>(3, 0); }
		double vy() const { return kf.statePost.at<double>(4, 0); }
		double vz() const { return kf.statePost.at<double>(5, 0); }

		void reset()
		{
			initialized = false;
			kf.statePost = cv::Mat::zeros(6, 1, CV_64F);
			kf.statePre  = cv::Mat::zeros(6, 1, CV_64F);

			kf.errorCovPost = cv::Mat::eye(6, 6, CV_64F);
			kf.errorCovPost.at<double>(3, 3) = 10.0;
			kf.errorCovPost.at<double>(4, 4) = 10.0;
			kf.errorCovPost.at<double>(5, 5) = 10.0;

			lastPredictTime = rclcpp::Time(0, 0, RCL_ROS_TIME);
		}
	};

	struct LockState
	{
		bool        locked       = false;
		cv::Point2f center       = {0.0f, 0.0f};
		float       visible_size = 0.0f;
		int         missed       = 0;

		void reset()
		{
			locked       = false;
			center       = {0.0f, 0.0f};
			visible_size = 0.0f;
			missed       = 0;
		}
	};

	struct LastSeenState
	{
		bool         has_pose      = false;
		double       x = 0.0;
		double       y = 0.0;
		double       z = 0.0;
		bool         timer_started = false;
		rclcpp::Time lost_start    = rclcpp::Time(0, 0, RCL_ROS_TIME);

		void reset()
		{
			has_pose      = false;
			x             = 0.0;
			y             = 0.0;
			z             = 0.0;
			timer_started = false;
			lost_start    = rclcpp::Time(0, 0, RCL_ROS_TIME);
		}
	};

	// ─────────────────────────── Params ─────────────────────────────
	struct Params
	{
		// ROS topics
		std::string image_topic = "/camera_down/image_raw";
		std::string camera_info_topic = "/camera_down/camera_info";
		std::string vehicle_odometry_topic = "/fmu/out/vehicle_odometry";
		std::string reset_topic = "/cube_detector/reset";

		std::string hsv_debug_topic = "/cube_detector/algorithm_debug";
		std::string performance_topic = "/cube_detector/performance";

		// Backward compatible filtered pose topic
		std::string target_pose_topic = "/cube_detector/target_pose";

		// New explicit debug/tune topics
		std::string target_pose_camera_raw_topic = "/cube_detector/target_pose_camera_raw";
		std::string target_pose_world_raw_topic = "/cube_detector/target_pose_world_raw";
		std::string target_pose_world_filtered_topic = "/cube_detector/target_pose_world_filtered";
		std::string target_velocity_world_filtered_topic = "/cube_detector/target_velocity_world_filtered";
		std::string target_valid_topic = "/cube_detector/target_valid";

		// Frame ids
		std::string camera_frame_id = "camera_optical_frame";
		std::string world_frame_id = "map_ned";

		// Box HSV - Orange target
		// OpenCV scale: H[0-180], S[0-255], V[0-255]
		int h_min = 5;
		int s_min = 80;
		int v_min = 50;
		int h_max = 30;
		int s_max = 255;
		int v_max = 255;

		// Detector selection
		// - algorithm: LAB + CLAHE + contour-shape scoring
		// - yolo8n:    OpenCV DNN + YOLOv8n ONNX
		// - yolo26n:   OpenCV DNN + YOLO26n ONNX (same runtime, different model)
		std::string detector_mode = "algorithm";

		// Algorithm (LAB) params
		int lab_l_min = 0;
		int lab_a_min = 125;
		int lab_b_min = 135;
		int lab_l_max = 255;
		int lab_a_max = 190;
		int lab_b_max = 255;
		double clahe_clip_limit = 2.0;
		int clahe_grid_size = 8;
		double min_fill_ratio = 0.35;
		double min_aspect_ratio = 0.45;
		double max_aspect_ratio = 2.7;
		double max_circularity = 0.92;

		// YOLO params
		std::string yolo8_model_path = "";
		std::string yolo26_model_path = "";
		int yolo_input_size = 640;
		double yolo_conf_threshold = 0.35;
		double yolo_nms_threshold = 0.45;
		int yolo_target_class_id = -1;
		bool yolo_use_cuda = false;

		// Detection hysteresis
		int lost_grace_frames = 6;

		// Box area
		int min_box_area = 500;
		int max_box_area = 200000;

		// Circle HSV
		int circle_h_min = 0;
		int circle_s_min = 0;
		int circle_v_min = 0;
		int circle_h_max = 180;
		int circle_s_max = 50;
		int circle_v_max = 200;

		// Circle geometry
		double circle_min_radius = 15.0;
		double circle_max_radius = 80.0;
		double circle_position_tolerance = 0.15;

		// Physical
		double box_width_m = 0.45;

		// Lock
		int max_lock_missed = 30;
		double lock_max_dist_px = 120.0;
		double lock_min_size_ratio = 0.5;

		// Timing
		double last_seen_hold_timeout_s = 0.5;

		// Kalman 6-state process noise
		double kalman_q_acc_x = 0.01;
		double kalman_q_acc_y = 0.01;
		double kalman_q_acc_z = 0.01;

		// Kalman measurement noise
		double kalman_r_pos_x = 0.05;
		double kalman_r_pos_y = 0.05;
		double kalman_r_pos_z = 0.05;

		// Scoring weights
		double score_dist_weight = 3.0;
		double score_size_ratio_weight = 120.0;
		double score_area_weight = 0.02;
		double score_fill_weight = 30.0;
		double score_circularity_weight = 40.0;
		double score_center_weight = 0.5;

		// Camera offset trong body frame PX4 FRD/NED.
		// FRD: X forward, Y right, Z down. Camera nam duoi bung 9 cm => z duong 0.09.
		double cam_offset_x = 0.0;
		double cam_offset_y = 0.0;
		double cam_offset_z = 0.09;

		// Khi mat target:
		//   - target_valid publish false
		//   - neu true thi van publish Kalman prediction ra filtered topics de tune
		bool publish_prediction_when_lost = true;

		// Debug control
		// debug_enable: bat/tat /cube_detector/algorithm_debug.
		// Khi false thi khong tao anh debug, khong blur, khong hconcat.
		bool debug_enable = false;

		// Bat/tat publish cac topic raw debug:
		// /cube_detector/target_pose_camera_raw
		// /cube_detector/target_pose_world_raw
		bool debug_raw_pose_enable = false;

		// Bat/tat publish /cube_detector/performance.
		bool debug_performance_enable = true;
	} _p;

	// ───────────────────────── ROS I/O ──────────────────────────────
	rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _image_sub;
	rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr _camera_info_sub;
	rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr _reset_sub;
	rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr _vehicle_odometry_sub;

	// Backward compatible: filtered world pose
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _target_pose_pub;

	// Explicit debug/tune outputs
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _target_pose_camera_raw_pub;
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _target_pose_world_raw_pub;
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _target_pose_world_filtered_pub;
	rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr _target_velocity_world_filtered_pub;

	rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr _target_valid_pub;
	rclcpp::Publisher<std_msgs::msg::String>::SharedPtr _perf_pub;
	rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _hsv_debug_pub;

	// ─────────────────────── Camera state ───────────────────────────
	bool _has_camera_info = false;
	cv::Mat _camera_matrix;
	cv::Mat _dist_coeffs;

	// ──────────────── Frame transformer (optical -> NED) ──────────
	frame_transform::FrameTransformer _frame_transformer;
	bool _has_vehicle_pose = false;

	// ─────────────────── Detection / tracking state ─────────────────
	LockState _lock;
	LastSeenState _last_seen;
	KalmanState _kalman;

	bool _has_last_pose = false;
	double _last_x = 0.0;
	double _last_y = 0.0;
	double _last_z = 0.0;

	// ──────────────── Cached / pre-computed resources ───────────────
	cv::Mat _morph_kernel;
	cv::Ptr<cv::CLAHE> _clahe;
	cv::dnn::Net _yolo8_net;
	cv::dnn::Net _yolo26_net;
	bool _yolo8_loaded = false;
	bool _yolo26_loaded = false;

	using Clock = std::chrono::steady_clock;
	Clock::time_point _perf_last_pub = Clock::now();
	uint64_t _perf_frame_count = 0;
	double _perf_total_ms = 0.0;

	uint64_t _perf_detail_count = 0;
	double _perf_detect_ms = 0.0;
	double _perf_pose_ms = 0.0;
	double _perf_kalman_ms = 0.0;
	double _perf_publish_ms = 0.0;
	double _perf_annotate_ms = 0.0;
	std::string _perf_overlay_text;

	// ─────────────────────── Private methods ────────────────────────
	void loadParameters();

	// Callbacks
	void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
	void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
	void reset_callback(const std_msgs::msg::Bool::SharedPtr msg);
	void vehicleOdometryCallback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg);

	// Detection pipeline
	VictimModel detectVictimModel(const cv::Mat &frame);
	bool detectBoxRegion(const cv::Mat &frame, cv::Rect &boxBbox, cv::Point2f &boxCenter);
	bool detectBoxRegionAlgorithm(const cv::Mat &frame, cv::Rect &boxBbox, cv::Point2f &boxCenter);
	bool detectBoxRegionYolo(
		const cv::Mat &frame,
		cv::dnn::Net &net,
		const std::string &pipeline_name,
		cv::Rect &boxBbox,
		cv::Point2f &boxCenter);
	bool makeAlgorithmMask(const cv::Mat &frame, cv::Mat &mask);
	bool loadYoloModel(cv::dnn::Net &net, const std::string &model_path, const std::string &name);
	bool detectCircularHandle(
		const cv::Mat &frame,
		const cv::Rect &boxRegion,
		cv::Point2f &circleCenter,
		float &radius);
	bool validateVictimGeometry(const VictimModel &model);
	bool isLockedCandidateValid(const cv::Rect &bbox, const cv::Point2f &center) const;
	double scoreLockedCandidate(const cv::Point2f &center, const cv::Rect &bbox, double area) const;
	double scoreUnlockedCandidate(
		double area,
		double fill_ratio,
		double circularity,
		const cv::Point2f &center,
		const cv::Point2f &img_center) const;

	// Pose estimation: tra ve toa do trong camera OPTICAL frame
	void estimatePose(
		const VictimModel &target,
		int image_width,
		int image_height,
		double &x,
		double &y,
		double &z) const;

	// Kalman 6-state trong world frame
	void initKalman();
	void kalmanPredict(const rclcpp::Time &now_ts);
	void kalmanUpdate(double meas_x, double meas_y, double meas_z, const rclcpp::Time &now_ts);

	// Publishing helpers
	geometry_msgs::msg::PoseStamped buildPoseMsg(
		const std_msgs::msg::Header &header,
		const std::string &frame_id,
		double x,
		double y,
		double z) const;

	geometry_msgs::msg::TwistStamped buildVelocityMsg(
		const std_msgs::msg::Header &header,
		const std::string &frame_id,
		double vx,
		double vy,
		double vz) const;

	void publishPose(
		const rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr &publisher,
		const std_msgs::msg::Header &header,
		const std::string &frame_id,
		double x,
		double y,
		double z) const;

	void publishTargetValid(bool valid);
	void publishFilteredOutput(const std_msgs::msg::Header &header);

	// HSV debug image: mask panel + milky-white panel
	void publishHsvDebug(
		const cv::Mat &frame,
		const VictimModel &victim,
		const std_msgs::msg::Header &header);

	// State management
	void resetLockState();

	// Annotation
	void annotateImage(
		cv_bridge::CvImagePtr image,
		const VictimModel &target,
		double x,
		double y,
		double z,
		const std::string &perf_text) const;
};