#include "PencilDetectorNode.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>

// ═══════════════════════════════════════════════════════════════════
namespace
{

geometry_msgs::msg::TwistStamped zero_twist_world(
	const std_msgs::msg::Header &header,
	const std::string &world_frame_id)
{
	geometry_msgs::msg::TwistStamped m;
	m.header           = header;
	m.header.frame_id  = world_frame_id;
	m.twist.linear.x   = 0.0;
	m.twist.linear.y   = 0.0;
	m.twist.linear.z   = 0.0;
	m.twist.angular.x = 0.0;
	m.twist.angular.y = 0.0;
	m.twist.angular.z = 0.0;
	return m;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════
// Thuật toán phát hiện vàng (trước đây trong pencilDetector.cpp)
namespace yellow_pencil
{

bool detectYellowPencil(const cv::Mat &bgr, DetectResult &out, const DetectParams &p)
{
	out = DetectResult{};

	if (bgr.empty() || bgr.channels() != 3 || bgr.type() != CV_8UC3) {
		return false;
	}

	cv::Mat hsv;
	cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

	cv::Mat mask;
	cv::inRange(hsv, p.hsv_low, p.hsv_high, mask);

	const int k = std::max(3, p.morph_kernel_px | 1);
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
	cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
	cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	if (contours.empty()) {
		return false;
	}

	auto best_it = std::max_element(
		contours.begin(),
		contours.end(),
		[](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b) {
			return cv::contourArea(a) < cv::contourArea(b);
		});

	const double area = cv::contourArea(*best_it);
	if (area < p.min_area_px) {
		return false;
	}

	const cv::Rect bbox = cv::boundingRect(*best_it);
	cv::Moments mu = cv::moments(*best_it);
	cv::Point2f center{};
	if (std::abs(mu.m00) > 1e-6) {
		center = cv::Point2f(static_cast<float>(mu.m10 / mu.m00),
			static_cast<float>(mu.m01 / mu.m00));
	} else {
		center = cv::Point2f(
			bbox.x + bbox.width * 0.5f,
			bbox.y + bbox.height * 0.5f);
	}

	out.valid  = true;
	out.bbox   = bbox;
	out.center = center;
	out.area   = area;
	return true;
}

} // namespace yellow_pencil

// ═══════════════════════════════════════════════════════════════════
PencilDetectorNode::PencilDetectorNode()
	: Node("pencil_detector_node")
{
	loadParameters();

	const auto ft_config = frame_transform::FrameTransformer::makeBellyFixedCameraConfig(
		Eigen::Vector3d(_p.cam_offset_x, _p.cam_offset_y, _p.cam_offset_z));
	_frame_transformer.setConfig(ft_config);

	auto qos = rclcpp::QoS(1).best_effort();

	_image_sub = create_subscription<sensor_msgs::msg::Image>(
		_p.image_topic, qos,
		std::bind(&PencilDetectorNode::image_callback, this, std::placeholders::_1));

	_camera_info_sub = create_subscription<sensor_msgs::msg::CameraInfo>(
		_p.camera_info_topic, qos,
		std::bind(&PencilDetectorNode::camera_info_callback, this, std::placeholders::_1));

	_reset_sub = create_subscription<std_msgs::msg::Bool>(
		_p.reset_topic, rclcpp::QoS(10),
		std::bind(&PencilDetectorNode::reset_callback, this, std::placeholders::_1));

	_vehicle_odometry_sub = create_subscription<px4_msgs::msg::VehicleOdometry>(
		_p.vehicle_odometry_topic, qos,
		std::bind(&PencilDetectorNode::vehicle_odometry_callback, this, std::placeholders::_1));

	_target_pose_pub = create_publisher<geometry_msgs::msg::PoseStamped>(_p.target_pose_topic, qos);
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
	_debug_pub       = create_publisher<sensor_msgs::msg::Image>(_p.debug_topic, qos);

	RCLCPP_INFO(get_logger(), "===== PencilDetectorNode started =====");
	RCLCPP_INFO(get_logger(), "subscribe image   : %s", _p.image_topic.c_str());
	RCLCPP_INFO(get_logger(), "subscribe cam_info: %s", _p.camera_info_topic.c_str());
	RCLCPP_INFO(get_logger(), "subscribe odom    : %s", _p.vehicle_odometry_topic.c_str());
	RCLCPP_INFO(get_logger(), "subscribe reset   : %s", _p.reset_topic.c_str());
	RCLCPP_INFO(get_logger(), "publish debug     : %s", _p.debug_topic.c_str());
	RCLCPP_INFO(get_logger(), "publish pose world: %s", _p.target_pose_world_filtered_topic.c_str());
}

void PencilDetectorNode::loadParameters()
{
	_p.image_topic = declare_parameter<std::string>("topics.image", _p.image_topic);
	_p.camera_info_topic = declare_parameter<std::string>("topics.camera_info", _p.camera_info_topic);
	_p.vehicle_odometry_topic =
		declare_parameter<std::string>("topics.vehicle_odometry", _p.vehicle_odometry_topic);
	_p.reset_topic = declare_parameter<std::string>("topics.reset", _p.reset_topic);

	_p.debug_topic = declare_parameter<std::string>("topics.hsv_debug", _p.debug_topic);
	_p.performance_topic = declare_parameter<std::string>("topics.performance", _p.performance_topic);

	_p.target_pose_topic =
		declare_parameter<std::string>("topics.target_pose", _p.target_pose_topic);
	_p.target_pose_camera_raw_topic =
		declare_parameter<std::string>("topics.target_pose_camera_raw", _p.target_pose_camera_raw_topic);
	_p.target_pose_world_raw_topic =
		declare_parameter<std::string>("topics.target_pose_world_raw", _p.target_pose_world_raw_topic);
	_p.target_pose_world_filtered_topic =
		declare_parameter<std::string>(
			"topics.target_pose_world_filtered", _p.target_pose_world_filtered_topic);
	_p.target_velocity_world_filtered_topic =
		declare_parameter<std::string>(
			"topics.target_velocity_world_filtered", _p.target_velocity_world_filtered_topic);
	_p.target_valid_topic =
		declare_parameter<std::string>("topics.target_valid", _p.target_valid_topic);

	_p.camera_frame_id = declare_parameter<std::string>("camera_frame_id", _p.camera_frame_id);
	_p.world_frame_id  = declare_parameter<std::string>("world_frame_id", _p.world_frame_id);

	_p.cam_offset_x = declare_parameter<double>("cam_offset_x", _p.cam_offset_x);
	_p.cam_offset_y = declare_parameter<double>("cam_offset_y", _p.cam_offset_y);
	_p.cam_offset_z = declare_parameter<double>("cam_offset_z", _p.cam_offset_z);

	_p.object_extent_m = declare_parameter<double>("object_extent_m", _p.object_extent_m);

	_p.yellow_h_low  = declare_parameter<int>("yellow_h_low", _p.yellow_h_low);
	_p.yellow_s_low  = declare_parameter<int>("yellow_s_low", _p.yellow_s_low);
	_p.yellow_v_low  = declare_parameter<int>("yellow_v_low", _p.yellow_v_low);
	_p.yellow_h_high = declare_parameter<int>("yellow_h_high", _p.yellow_h_high);
	_p.yellow_s_high = declare_parameter<int>("yellow_s_high", _p.yellow_s_high);
	_p.yellow_v_high = declare_parameter<int>("yellow_v_high", _p.yellow_v_high);

	_p.morph_kernel_px = declare_parameter<int>("yellow_morph_kernel_px", _p.morph_kernel_px);
	_p.min_area_px     = declare_parameter<double>("yellow_min_area_px", _p.min_area_px);

	_p.publish_hz_min = declare_parameter<double>("publish_hz_min", _p.publish_hz_min);

	_p.morph_kernel_px = std::max(3, _p.morph_kernel_px | 1);
	if (_p.publish_hz_min < 0.0) {
		_p.publish_hz_min = 0.0;
	}
}

void PencilDetectorNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
	if (_has_camera_info) {
		return;
	}

	_camera_matrix = cv::Mat(3, 3, CV_64F, const_cast<double *>(msg->k.data())).clone();
	_dist_coeffs = msg->d.empty()
		? cv::Mat::zeros(5, 1, CV_64F)
		: cv::Mat(static_cast<int>(msg->d.size()), 1, CV_64F,
		          const_cast<double *>(msg->d.data())).clone();

	_has_camera_info = true;
	RCLCPP_INFO(get_logger(), "Camera intrinsics received: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
		msg->k[0], msg->k[4], msg->k[2], msg->k[5]);
}

void PencilDetectorNode::vehicle_odometry_callback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
{
	if (!msg) {
		RCLCPP_WARN(get_logger(), "vehicle_odometry_callback received null message");
		return;
	}

	if (msg->pose_frame != px4_msgs::msg::VehicleOdometry::POSE_FRAME_NED) {
		RCLCPP_WARN_THROTTLE(get_logger(),
			*get_clock(),
			2000,
			"VehicleOdometry pose_frame is not NED (pose_frame=%u)",
			static_cast<unsigned>(msg->pose_frame));
		return;
	}

	const bool position_valid =
		std::isfinite(msg->position[0]) &&
		std::isfinite(msg->position[1]) &&
		std::isfinite(msg->position[2]);

	const bool quaternion_valid =
		std::isfinite(msg->q[0]) &&
		std::isfinite(msg->q[1]) &&
		std::isfinite(msg->q[2]) &&
		std::isfinite(msg->q[3]);

	if (!position_valid || !quaternion_valid) {
		RCLCPP_WARN_THROTTLE(
			get_logger(), *get_clock(), 2000, "VehicleOdometry has non-finite position or quaternion");
		return;
	}

	frame_transform::VehicleStateData state;
	state.positionWorld = Eigen::Vector3d(
		static_cast<double>(msg->position[0]),
		static_cast<double>(msg->position[1]),
		static_cast<double>(msg->position[2]));
	state.worldFromBody = Eigen::Quaterniond(
		static_cast<double>(msg->q[0]),
		static_cast<double>(msg->q[1]),
		static_cast<double>(msg->q[2]),
		static_cast<double>(msg->q[3]));

	_frame_transformer.setVehicleState(state);
	_has_vehicle_pose = true;
}

void PencilDetectorNode::reset_callback(const std_msgs::msg::Bool::SharedPtr /*msg*/)
{
	RCLCPP_WARN(get_logger(), "[RESET] via %s", _p.reset_topic.c_str());
	_last_publish_time = rclcpp::Time(0, 0, get_clock()->get_clock_type());
}

yellow_pencil::DetectParams PencilDetectorNode::buildDetectParams() const
{
	yellow_pencil::DetectParams p;
	p.hsv_low         = cv::Scalar(_p.yellow_h_low, _p.yellow_s_low, _p.yellow_v_low);
	p.hsv_high        = cv::Scalar(_p.yellow_h_high, _p.yellow_s_high, _p.yellow_v_high);
	p.morph_kernel_px = _p.morph_kernel_px;
	p.min_area_px     = _p.min_area_px;
	return p;
}

void PencilDetectorNode::opticalPoseFromBBox(double pixel_extent,
	const cv::Point2f &center,
	double &x,
	double &y,
	double &z) const
{
	const double fx = _camera_matrix.at<double>(0, 0);
	const double fy = _camera_matrix.at<double>(1, 1);
	const double cx = _camera_matrix.at<double>(0, 2);
	const double cy = _camera_matrix.at<double>(1, 2);

	if (pixel_extent < 1.0 || !std::isfinite(pixel_extent)) {
		x = y = z = 0.0;
		return;
	}

	z = (fx * _p.object_extent_m) / pixel_extent;
	x = (static_cast<double>(center.x) - cx) * z / fx;
	y = (static_cast<double>(center.y) - cy) * z / fy;
}

geometry_msgs::msg::PoseStamped PencilDetectorNode::build_pose(
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

void PencilDetectorNode::publish_outputs(
	const std_msgs::msg::Header &header,
	double ox,
	double oy,
	double oz,
	bool valid)
{
	std_msgs::msg::Bool v;
	v.data = valid;
	_target_valid_pub->publish(v);

	if (!valid) {
		return;
	}

	const auto cam_pose = build_pose(header, _p.camera_frame_id, ox, oy, oz);
	_target_pose_camera_raw_pub->publish(cam_pose);

	if (!_has_vehicle_pose) {
		RCLCPP_WARN_THROTTLE(get_logger(),
			*get_clock(),
			1000,
			"[PENCIL] detected but waiting for PX4 vehicle odometry (optical→NED)");
		return;
	}

	const Eigen::Vector3d pos_world =
		_frame_transformer.opticalPositionToWorld(Eigen::Vector3d(ox, oy, oz));

	const auto world_raw =
		build_pose(header,
		           _p.world_frame_id,
		           pos_world.x(),
		           pos_world.y(),
		           pos_world.z());

	_target_pose_world_raw_pub->publish(world_raw);
	// Không có Kalman: filtered = raw để giữ API tương thích subscriber.
	_target_pose_world_filtered_pub->publish(world_raw);
	_target_velocity_world_filtered_pub->publish(zero_twist_world(header, _p.world_frame_id));
	_target_pose_pub->publish(world_raw);
}

void PencilDetectorNode::publish_debug_if_needed(
	const cv::Mat &frame,
	const yellow_pencil::DetectResult &det,
	const std_msgs::msg::Header &header)
{
	if (_debug_pub->get_subscription_count() == 0) {
		return;
	}

	cv::Mat viz = frame.clone();
	if (det.valid) {
		cv::rectangle(viz, det.bbox, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
		cv::circle(viz,
			cv::Point(static_cast<int>(det.center.x),
				static_cast<int>(det.center.y)),
			6,
			cv::Scalar(0, 165, 255),
			-1,
			cv::LINE_AA);
		cv::putText(viz,
			"Pencil yellow DETECTED",
			cv::Point(10, 30),
			cv::FONT_HERSHEY_SIMPLEX,
			0.8,
			cv::Scalar(0, 255, 100),
			2,
			cv::LINE_AA);
	} else {
		cv::putText(viz,
			"Pencil yellow SEARCHING",
			cv::Point(10, 30),
			cv::FONT_HERSHEY_SIMPLEX,
			0.8,
			cv::Scalar(80, 80, 255),
			2,
			cv::LINE_AA);
	}

	cv_bridge::CvImage cv_img;
	cv_img.header   = header;
	cv_img.encoding = sensor_msgs::image_encodings::BGR8;
	cv_img.image    = viz;
	_debug_pub->publish(*cv_img.toImageMsg());
}

void PencilDetectorNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
	if (!_has_camera_info) {
		RCLCPP_WARN_ONCE(get_logger(), "Waiting for camera info...");
		std_msgs::msg::Bool v;
		v.data = false;
		_target_valid_pub->publish(v);
		return;
	}

	const rclcpp::Time stamp = (msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0)
		? now()
		: rclcpp::Time(msg->header.stamp);

	if (_p.publish_hz_min > 1e-6) {
		const double dt = (_last_publish_time.nanoseconds() > 0)
			? (stamp - _last_publish_time).seconds()
			: 1e9;
		if (dt < 1.0 / _p.publish_hz_min) {
			return;
		}
		_last_publish_time = stamp;
	}

	cv_bridge::CvImagePtr cv_ptr;
	try {
		cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	} catch (const cv_bridge::Exception &e) {
		RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
		std_msgs::msg::Bool v;
		v.data = false;
		_target_valid_pub->publish(v);
		return;
	}

	const auto t0       = Clock::now();
	yellow_pencil::DetectResult det;
	const bool detected = yellow_pencil::detectYellowPencil(cv_ptr->image, det, buildDetectParams());

	double ox = 0.0, oy = 0.0, oz = 0.0;
	bool valid_output = false;
	if (detected && det.valid) {
		const double px =
			static_cast<double>(std::max(det.bbox.width, det.bbox.height));
		opticalPoseFromBBox(px, det.center, ox, oy, oz);
		valid_output =
			std::isfinite(ox) && std::isfinite(oy) && std::isfinite(oz) &&
			std::fabs(oz) > 1e-4;
	}

	publish_outputs(msg->header, ox, oy, oz, valid_output);
	publish_debug_if_needed(cv_ptr->image, det, msg->header);

	const auto t1 = Clock::now();
	const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
	_perf_total_ms += ms;
	++_perf_frame_count;

	const auto perf_end = Clock::now();
	if (perf_end - _perf_last_pub >= std::chrono::seconds(1) && _perf_frame_count > 0) {
		const double avg_ms = _perf_total_ms / static_cast<double>(_perf_frame_count);
		std::ostringstream oss;
		oss << "FPS~= " << std::fixed << std::setprecision(1) << (1000.0 / std::max(1e-6, avg_ms))
		    << " avg_ms=" << std::setprecision(2) << avg_ms << " pencil_detect";
		std_msgs::msg::String s;
		s.data = oss.str();
		_perf_pub->publish(s);
		_perf_last_pub      = perf_end;
		_perf_total_ms       = 0.0;
		_perf_frame_count    = 0;
	}
}

int main(int argc, char *argv[])
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<PencilDetectorNode>());
	rclcpp::shutdown();
	return 0;
}
