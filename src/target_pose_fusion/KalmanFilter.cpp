#include "KalmanFilter.hpp"

#include <algorithm>
#include <cmath>

namespace
{
constexpr char kInputTargetPoseTopic[] = "/target_pose";
constexpr char kResetTopic[] = "/reset";
constexpr char kVehicleAttitudeTopic[] = "/fmu/out/vehicle_attitude";

constexpr char kTargetErrorBodyTopic[] = "/target_error_body";
constexpr char kTargetErrorBodyFusionTopic[] = "/target_error_body_fusion";
constexpr char kKalmanResidualTopic[] = "/kalman_error";

constexpr char kFrameId[] = "body_level_frd";
} // namespace

TargetPoseFusionNode::TargetPoseFusionNode()
    : Node("target_pose_fusion_node")
{
    const auto sub_qos = rclcpp::QoS(1).best_effort();
    const auto pub_qos = rclcpp::QoS(1).best_effort();

    pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        kInputTargetPoseTopic,
        sub_qos,
        std::bind(&TargetPoseFusionNode::poseCallback, this, std::placeholders::_1));

    reset_sub_ = create_subscription<std_msgs::msg::String>(
        kResetTopic,
        sub_qos,
        std::bind(&TargetPoseFusionNode::resetCallback, this, std::placeholders::_1));

    vehicle_attitude_sub_ = create_subscription<px4_msgs::msg::VehicleAttitude>(
        kVehicleAttitudeTopic,
        sub_qos,
        std::bind(&TargetPoseFusionNode::vehicleAttitudeCallback, this, std::placeholders::_1));

    target_error_body_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        kTargetErrorBodyTopic,
        pub_qos);

    target_error_body_fusion_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        kTargetErrorBodyFusionTopic,
        pub_qos);

    kalman_residual_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        kKalmanResidualTopic,
        pub_qos);

    declareParameters();
    initKalman();

    timer_ = create_wall_timer(
        std::chrono::milliseconds(33),
        std::bind(&TargetPoseFusionNode::processAndPublish, this));

    last_predict_time_ = now();
    last_measurement_time_ = now();

    RCLCPP_INFO(get_logger(), "===== TargetPoseFusionNode started =====");
    RCLCPP_INFO(get_logger(), "sub  : %s", kInputTargetPoseTopic);
    RCLCPP_INFO(get_logger(), "sub  : %s", kResetTopic);
    RCLCPP_INFO(get_logger(), "sub  : %s", kVehicleAttitudeTopic);
    RCLCPP_INFO(get_logger(), "pub  : %s", kTargetErrorBodyTopic);
    RCLCPP_INFO(get_logger(), "pub  : %s", kTargetErrorBodyFusionTopic);
    RCLCPP_INFO(get_logger(), "pub  : %s", kKalmanResidualTopic);
}

void TargetPoseFusionNode::declareParameters()
{
    declare_parameter<double>("q_acc_x", 0.002);
    declare_parameter<double>("q_acc_y", 0.002);
    declare_parameter<double>("q_acc_z", 0.001);

    declare_parameter<double>("r_pos_x", 0.002);
    declare_parameter<double>("r_pos_y", 0.002);
    declare_parameter<double>("r_pos_z", 0.004);

    declare_parameter<double>("cam_offset_x", 0.12);
    declare_parameter<double>("cam_offset_y", 0.03);
    declare_parameter<double>("cam_offset_z", 0.242);
}

void TargetPoseFusionNode::initKalman()
{
    q_acc_x_ = get_parameter("q_acc_x").as_double();
    q_acc_y_ = get_parameter("q_acc_y").as_double();
    q_acc_z_ = get_parameter("q_acc_z").as_double();

    r_pos_x_ = get_parameter("r_pos_x").as_double();
    r_pos_y_ = get_parameter("r_pos_y").as_double();
    r_pos_z_ = get_parameter("r_pos_z").as_double();

    cam_offset_x_ = get_parameter("cam_offset_x").as_double();
    cam_offset_y_ = get_parameter("cam_offset_y").as_double();
    cam_offset_z_ = get_parameter("cam_offset_z").as_double();

    kf_ = cv::KalmanFilter(STATE_SIZE, MEASUREMENT_SIZE, 0, CV_64F);

    kf_.transitionMatrix = cv::Mat::eye(STATE_SIZE, STATE_SIZE, CV_64F);

    kf_.measurementMatrix = cv::Mat::zeros(MEASUREMENT_SIZE, STATE_SIZE, CV_64F);
    kf_.measurementMatrix.at<double>(0, 0) = 1.0;
    kf_.measurementMatrix.at<double>(1, 1) = 1.0;
    kf_.measurementMatrix.at<double>(2, 2) = 1.0;

    kf_.processNoiseCov = cv::Mat::zeros(STATE_SIZE, STATE_SIZE, CV_64F);

    kf_.measurementNoiseCov = cv::Mat::eye(MEASUREMENT_SIZE, MEASUREMENT_SIZE, CV_64F);
    kf_.measurementNoiseCov.at<double>(0, 0) = r_pos_x_;
    kf_.measurementNoiseCov.at<double>(1, 1) = r_pos_y_;
    kf_.measurementNoiseCov.at<double>(2, 2) = r_pos_z_;

    kf_.errorCovPost = cv::Mat::eye(STATE_SIZE, STATE_SIZE, CV_64F);
    kf_.errorCovPost.at<double>(3, 3) = 10.0;
    kf_.errorCovPost.at<double>(4, 4) = 10.0;
    kf_.errorCovPost.at<double>(5, 5) = 10.0;

    kf_.statePost = cv::Mat::zeros(STATE_SIZE, 1, CV_64F);
    kf_.statePre = cv::Mat::zeros(STATE_SIZE, 1, CV_64F);
}

void TargetPoseFusionNode::resetState()
{
    initialized_ = false;

    kf_.statePost = cv::Mat::zeros(STATE_SIZE, 1, CV_64F);
    kf_.statePre = cv::Mat::zeros(STATE_SIZE, 1, CV_64F);

    kf_.errorCovPost = cv::Mat::eye(STATE_SIZE, STATE_SIZE, CV_64F);
    kf_.errorCovPost.at<double>(3, 3) = 10.0;
    kf_.errorCovPost.at<double>(4, 4) = 10.0;
    kf_.errorCovPost.at<double>(5, 5) = 10.0;

    last_predict_time_ = now();
    last_measurement_time_ = now();

    RCLCPP_WARN(get_logger(), "[Kalman] reset state");
}

void TargetPoseFusionNode::resetCallback(const std_msgs::msg::String::SharedPtr msg)
{
    if (msg->data == "RESET") {
        resetState();
    }
}

void TargetPoseFusionNode::vehicleAttitudeCallback(
    const px4_msgs::msg::VehicleAttitude::SharedPtr msg)
{
    vehicle_attitude_msg_ = *msg;
    has_vehicle_attitude_ = true;
}

bool TargetPoseFusionNode::attitudeReady() const
{
    return has_vehicle_attitude_;
}

Eigen::Matrix3d TargetPoseFusionNode::opticalToBodyFrdRotation() const
{
    Eigen::Matrix3d rotation;
    rotation << 0.0, 0.0, 1.0,
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0;
    return rotation;
}

Eigen::Vector3d TargetPoseFusionNode::measurementOpticalToBody(const Eigen::Vector3d& p_opt) const
{
    const Eigen::Matrix3d optical_to_body = opticalToBodyFrdRotation();
    const Eigen::Vector3d p_cam_body = optical_to_body * p_opt;

    const Eigen::Vector3d cam_offset_body(
        cam_offset_x_,
        cam_offset_y_,
        cam_offset_z_);

    return cam_offset_body + p_cam_body;
}

Eigen::Vector3d TargetPoseFusionNode::bodyToLeveledBody(const Eigen::Vector3d& p_body) const
{
    const auto& q = vehicle_attitude_msg_.q;

    Eigen::Quaterniond q_nb(
        static_cast<double>(q[0]),
        static_cast<double>(q[1]),
        static_cast<double>(q[2]),
        static_cast<double>(q[3]));

    if (q_nb.norm() < 1e-9) {
        return p_body;
    }
    q_nb.normalize();

    const Eigen::Matrix3d R_nb = q_nb.toRotationMatrix();

    const double roll = std::atan2(R_nb(2, 1), R_nb(2, 2));
    const double pitch = std::asin(-std::clamp(R_nb(2, 0), -1.0, 1.0));
    const double yaw = std::atan2(R_nb(1, 0), R_nb(0, 0));

    const Eigen::AngleAxisd yaw_only_aa(yaw, Eigen::Vector3d::UnitZ());
    const Eigen::Matrix3d R_yaw_only = yaw_only_aa.toRotationMatrix();

    const Eigen::Vector3d p_ned = R_nb * p_body;
    const Eigen::Vector3d p_body_level = R_yaw_only.transpose() * p_ned;

    (void)roll;
    (void)pitch;
    return p_body_level;
}

void TargetPoseFusionNode::publishRawMeasurement(
    const Eigen::Vector3d& measurement_body_level,
    const rclcpp::Time& stamp) const
{
    geometry_msgs::msg::PoseStamped msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = kFrameId;
    msg.pose.position.x = measurement_body_level.x();  // forward
    msg.pose.position.y = measurement_body_level.y();  // right
    msg.pose.position.z = measurement_body_level.z();  // down
    msg.pose.orientation.w = 1.0;
    target_error_body_pub_->publish(msg);
}

void TargetPoseFusionNode::publishResidual(
    const cv::Mat& residual,
    const rclcpp::Time& stamp) const
{
    geometry_msgs::msg::PoseStamped msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = kFrameId;
    msg.pose.position.x = residual.at<double>(0, 0);
    msg.pose.position.y = residual.at<double>(1, 0);
    msg.pose.position.z = residual.at<double>(2, 0);
    msg.pose.orientation.w = 1.0;
    kalman_residual_pub_->publish(msg);
}

void TargetPoseFusionNode::publishEstimatedState(const rclcpp::Time& stamp) const
{
    geometry_msgs::msg::PoseStamped msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = kFrameId;
    msg.pose.position.x = kf_.statePre.at<double>(0, 0);
    msg.pose.position.y = kf_.statePre.at<double>(1, 0);
    msg.pose.position.z = kf_.statePre.at<double>(2, 0);
    msg.pose.orientation.w = 1.0;
    target_error_body_fusion_pub_->publish(msg);
}

void TargetPoseFusionNode::poseCallback(
    const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (!attitudeReady()) {
        return;
    }

    rclcpp::Time meas_ts = msg->header.stamp;
    if (meas_ts.nanoseconds() == 0) {
        meas_ts = now();
    }

    last_measurement_time_ = meas_ts;

    const Eigen::Vector3d measurement_optical(
        msg->pose.position.x,
        msg->pose.position.y,
        msg->pose.position.z);

    const Eigen::Vector3d measurement_body =
        measurementOpticalToBody(measurement_optical);

    const Eigen::Vector3d measurement_body_level =
        bodyToLeveledBody(measurement_body);

    publishRawMeasurement(measurement_body_level, meas_ts);

    if (!initialized_) {
        kf_.statePost.at<double>(0, 0) = measurement_body_level.x();
        kf_.statePost.at<double>(1, 0) = measurement_body_level.y();
        kf_.statePost.at<double>(2, 0) = measurement_body_level.z();
        kf_.statePost.at<double>(3, 0) = 0.0;
        kf_.statePost.at<double>(4, 0) = 0.0;
        kf_.statePost.at<double>(5, 0) = 0.0;

        kf_.statePre = kf_.statePost.clone();
        initialized_ = true;
        last_predict_time_ = meas_ts;

        publishEstimatedState(meas_ts);

        RCLCPP_INFO(
            get_logger(),
            "[Kalman] initialized leveled body error = (%.3f, %.3f, %.3f)",
            measurement_body_level.x(),
            measurement_body_level.y(),
            measurement_body_level.z());
        return;
    }

    cv::Mat measurement(MEASUREMENT_SIZE, 1, CV_64F);
    measurement.at<double>(0, 0) = measurement_body_level.x();
    measurement.at<double>(1, 0) = measurement_body_level.y();
    measurement.at<double>(2, 0) = measurement_body_level.z();

    const cv::Mat predicted_measurement = kf_.measurementMatrix * kf_.statePre;
    const cv::Mat residual = measurement - predicted_measurement;

    kf_.correct(measurement);
    publishResidual(residual, meas_ts);
}

void TargetPoseFusionNode::processAndPublish()
{
    if (!initialized_) {
        return;
    }

    const rclcpp::Time now_ts = now();

    double dt = (now_ts - last_predict_time_).seconds();
    if (dt <= 0.0) {
        dt = 1e-3;
    }

    dt = std::min(dt, 0.1);
    last_predict_time_ = now_ts;

    predict(dt);
    publishEstimatedState(now_ts);
}

void TargetPoseFusionNode::predict(double dt)
{
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt3 * dt;

    kf_.transitionMatrix = cv::Mat::eye(STATE_SIZE, STATE_SIZE, CV_64F);
    kf_.transitionMatrix.at<double>(0, 3) = dt;
    kf_.transitionMatrix.at<double>(1, 4) = dt;
    kf_.transitionMatrix.at<double>(2, 5) = dt;

    kf_.processNoiseCov = cv::Mat::zeros(STATE_SIZE, STATE_SIZE, CV_64F);

    kf_.processNoiseCov.at<double>(0, 0) = 0.25 * dt4 * q_acc_x_;
    kf_.processNoiseCov.at<double>(0, 3) = 0.5 * dt3 * q_acc_x_;
    kf_.processNoiseCov.at<double>(3, 0) = 0.5 * dt3 * q_acc_x_;
    kf_.processNoiseCov.at<double>(3, 3) = dt2 * q_acc_x_;

    kf_.processNoiseCov.at<double>(1, 1) = 0.25 * dt4 * q_acc_y_;
    kf_.processNoiseCov.at<double>(1, 4) = 0.5 * dt3 * q_acc_y_;
    kf_.processNoiseCov.at<double>(4, 1) = 0.5 * dt3 * q_acc_y_;
    kf_.processNoiseCov.at<double>(4, 4) = dt2 * q_acc_y_;

    kf_.processNoiseCov.at<double>(2, 2) = 0.25 * dt4 * q_acc_z_;
    kf_.processNoiseCov.at<double>(2, 5) = 0.5 * dt3 * q_acc_z_;
    kf_.processNoiseCov.at<double>(5, 2) = 0.5 * dt3 * q_acc_z_;
    kf_.processNoiseCov.at<double>(5, 5) = dt2 * q_acc_z_;

    kf_.predict();
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TargetPoseFusionNode>());
    rclcpp::shutdown();
    return 0;
}