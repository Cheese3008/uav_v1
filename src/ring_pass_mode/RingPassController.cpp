#include "RingPassController.hpp"

#include <algorithm>
#include <cmath>

namespace
{
    constexpr char kVehicleAttitudeTopic[] = "/fmu/out/vehicle_attitude";
    constexpr char kTargetErrorFusionTopic[] = "/ring_detect/target_error_body_filtered";
    constexpr char kTargetVelocityFusionTopic[] = "/ring_detect/target_velocity_body_filtered";
    constexpr char kRingDetectResetTopic[] = "/ring_detect/reset_cmd";
}

RingPassController::RingPassController(rclcpp::Node& node)
    : node_(node)
{
    vehicle_attitude_sub_ =
        node_.create_subscription<px4_msgs::msg::VehicleAttitude>(
            kVehicleAttitudeTopic,
            rclcpp::QoS(10).best_effort(),
            std::bind(&RingPassController::vehicleAttitudeCallback, this, std::placeholders::_1));

    target_error_sub_ =
        node_.create_subscription<geometry_msgs::msg::PoseStamped>(
            kTargetErrorFusionTopic,
            rclcpp::QoS(1).best_effort(),
            std::bind(&RingPassController::targetErrorCallback, this, std::placeholders::_1));

    target_velocity_sub_ =
        node_.create_subscription<geometry_msgs::msg::TwistStamped>(
            kTargetVelocityFusionTopic,
            rclcpp::QoS(1).best_effort(),
            std::bind(&RingPassController::targetVelocityCallback, this, std::placeholders::_1));

    ring_detect_reset_pub_ =
        node_.create_publisher<std_msgs::msg::String>(
            kRingDetectResetTopic,
            rclcpp::QoS(10).reliable());

    loadParameters();
    reset();
}

void RingPassController::loadParameters()
{
    node_.declare_parameter<float>("target_timeout", 0.3f);

    node_.declare_parameter<float>("ringpass_kp_lat", 0.8f);
    node_.declare_parameter<float>("ringpass_kd_lat", 0.0f);

    node_.declare_parameter<float>("ringpass_kp_z", 0.8f);
    node_.declare_parameter<float>("ringpass_kd_z", 0.0f);

    node_.declare_parameter<float>("ringpass_lat_deadband", 0.03f);
    node_.declare_parameter<float>("ringpass_z_deadband", 0.04f);

    node_.declare_parameter<float>("ringpass_v_forward_min", 1.5f);
    node_.declare_parameter<float>("ringpass_v_forward_max", 2.0f);
    node_.declare_parameter<float>("ringpass_r_approach", 0.35f);
    node_.declare_parameter<float>("ringpass_r_forward_full", 0.10f);
    node_.declare_parameter<float>("ringpass_klog_forward", 8.0f);

    node_.declare_parameter<float>("ringpass_v_lateral_max", 1.2f);
    node_.declare_parameter<float>("ringpass_v_vertical_max", 0.7f);

    node_.declare_parameter<float>("ringpass_slew_xy", 1.5f);
    node_.declare_parameter<float>("ringpass_slew_z", 1.2f);

    node_.declare_parameter<float>("ringpass_height_offset", 0.40f);
    node_.declare_parameter<float>("ringpass_finish_forward_error", 0.05f);

    node_.get_parameter("target_timeout", param_target_timeout_);

    node_.get_parameter("ringpass_kp_lat", param_kp_lat_);
    node_.get_parameter("ringpass_kd_lat", param_kd_lat_);

    node_.get_parameter("ringpass_kp_z", param_kp_z_);
    node_.get_parameter("ringpass_kd_z", param_kd_z_);

    node_.get_parameter("ringpass_lat_deadband", param_lat_deadband_);
    node_.get_parameter("ringpass_z_deadband", param_z_deadband_);

    node_.get_parameter("ringpass_v_forward_min", param_v_forward_min_);
    node_.get_parameter("ringpass_v_forward_max", param_v_forward_max_);
    node_.get_parameter("ringpass_r_approach", param_r_approach_);
    node_.get_parameter("ringpass_r_forward_full", param_r_forward_full_);
    node_.get_parameter("ringpass_klog_forward", param_klog_forward_);

    node_.get_parameter("ringpass_v_lateral_max", param_v_lateral_max_);
    node_.get_parameter("ringpass_v_vertical_max", param_v_vertical_max_);

    node_.get_parameter("ringpass_slew_xy", param_slew_xy_);
    node_.get_parameter("ringpass_slew_z", param_slew_z_);

    node_.get_parameter("ringpass_height_offset", param_height_offset_);
    node_.get_parameter("ringpass_finish_forward_error", param_finish_forward_error_);

    param_r_approach_ = std::max(param_r_approach_, param_r_forward_full_ + 1e-3f);
    param_v_forward_max_ = std::max(param_v_forward_max_, param_v_forward_min_);
    param_klog_forward_ = std::max(param_klog_forward_, 1e-3f);
}

void RingPassController::publishDetectorReset()
{
    if (!ring_detect_reset_pub_) {
        return;
    }

    std_msgs::msg::String msg;
    msg.data = "RESET";
    ring_detect_reset_pub_->publish(msg);

    RCLCPP_WARN(
        node_.get_logger(),
        "[RingPassController] publish /ring_detect/reset = RESET");
}

void RingPassController::reset()
{
    vx_last_ = 0.0f;
    vy_last_ = 0.0f;
    vz_last_ = 0.0f;

    finished_ = false;
    target_.valid = false;
    target_.velocity = Eigen::Vector3f::Zero();
    publishDetectorReset();
}

void RingPassController::vehicleAttitudeCallback(
    const px4_msgs::msg::VehicleAttitude::SharedPtr msg)
{
    vehicle_attitude_msg_ = *msg;
    has_vehicle_attitude_ = true;
}

void RingPassController::targetErrorCallback(
    const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    target_.value.x() = static_cast<float>(msg->pose.position.x);
    target_.value.y() = static_cast<float>(msg->pose.position.y);
    target_.value.z() = static_cast<float>(msg->pose.position.z);

    target_.stamp = (msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0)
        ? node_.now()
        : rclcpp::Time(msg->header.stamp);

    target_.valid = true;
}

void RingPassController::targetVelocityCallback(
    const geometry_msgs::msg::TwistStamped::SharedPtr msg)
{
    target_.velocity.x() = static_cast<float>(msg->twist.linear.x);
    target_.velocity.y() = static_cast<float>(msg->twist.linear.y);
    target_.velocity.z() = static_cast<float>(msg->twist.linear.z);
}

bool RingPassController::attitudeReady() const
{
    return has_vehicle_attitude_;
}

bool RingPassController::targetTimedOut() const
{
    if (!target_.valid) {
        return true;
    }

    const double age_s = (node_.now() - target_.stamp).seconds();
    return age_s > static_cast<double>(param_target_timeout_);
}

bool RingPassController::isReady() const
{
    return attitudeReady();
}

bool RingPassController::hasValidTarget() const
{
    return !targetTimedOut();
}

bool RingPassController::isFinished() const
{
    return finished_;
}

float RingPassController::getVehicleYaw() const
{
    const auto& q = vehicle_attitude_msg_.q;

    const float w = q[0];
    const float x = q[1];
    const float y = q[2];
    const float z = q[3];

    const float siny_cosp = 2.0f * (w * z + x * y);
    const float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);

    return std::atan2(siny_cosp, cosy_cosp);
}

float RingPassController::applySlew(float cmd, float prev, float accel_limit, float dt_s) const
{
    const float dt = std::max(dt_s, 1e-3f);
    const float max_delta = accel_limit * dt;
    const float delta = std::clamp(cmd - prev, -max_delta, max_delta);
    return prev + delta;
}

float RingPassController::computeForwardVelocity(float center_error) const
{
    if (center_error > param_r_approach_) {
        return 0.0f;
    }

    const float r = std::min(center_error, param_r_approach_);
    const float denom = std::log1p(param_klog_forward_ * param_r_approach_);

    float f = 1.0f - (
        std::log1p(param_klog_forward_ * r) /
        std::max(denom, 1e-6f)
    );
    f = std::clamp(f, 0.0f, 1.0f);

    return param_v_forward_min_ + (param_v_forward_max_ - param_v_forward_min_) * f;
}

Eigen::Vector2f RingPassController::bodyToWorldXY(const Eigen::Vector2f& body_xy, float yaw) const
{
    const float c = std::cos(yaw);
    const float s = std::sin(yaw);

    Eigen::Vector2f world_xy;
    world_xy.x() = c * body_xy.x() - s * body_xy.y();
    world_xy.y() = s * body_xy.x() + c * body_xy.y();
    return world_xy;
}

Eigen::Vector3f RingPassController::computeVelocityCommand(float dt_s)
{
    const float err_forward = target_.value.x();
    const float err_lateral = target_.value.y();
    const float err_down = target_.value.z() - param_height_offset_;

    const float vel_lateral_rel = target_.velocity.y();
    const float vel_down_rel = target_.velocity.z();

    const float center_error = std::sqrt(err_lateral * err_lateral + err_down * err_down);

    float vx_body_cmd = computeForwardVelocity(center_error);
    if (err_forward < 0.0f) {
        vx_body_cmd = 0.0f;
    }

    float vy_body_cmd = param_kp_lat_ * err_lateral - param_kd_lat_ * vel_lateral_rel;
    if (std::abs(err_lateral) < param_lat_deadband_) {
        vy_body_cmd = -param_kd_lat_ * vel_lateral_rel;
    }
    vy_body_cmd = std::clamp(vy_body_cmd, -param_v_lateral_max_, param_v_lateral_max_);

    float vz_cmd = param_kp_z_ * err_down - param_kd_z_ * vel_down_rel;
    if (std::abs(err_down) < param_z_deadband_) {
        vz_cmd = -param_kd_z_ * vel_down_rel;
    }
    vz_cmd = std::clamp(vz_cmd, -param_v_vertical_max_, param_v_vertical_max_);

    const float yaw_now = getVehicleYaw();
    const Eigen::Vector2f world_xy =
        bodyToWorldXY(Eigen::Vector2f(vx_body_cmd, vy_body_cmd), yaw_now);

    vx_last_ = applySlew(world_xy.x(), vx_last_, param_slew_xy_, dt_s);
    vy_last_ = applySlew(world_xy.y(), vy_last_, param_slew_xy_, dt_s);
    vz_last_ = applySlew(vz_cmd, vz_last_, param_slew_z_, dt_s);

    if (err_forward <= param_finish_forward_error_) {
        finished_ = true;
    }

    return Eigen::Vector3f(vx_last_, vy_last_, vz_last_);
}

Eigen::Vector3f RingPassController::computeHoldCommand(float dt_s)
{
    vx_last_ = applySlew(0.0f, vx_last_, param_slew_xy_, dt_s);
    vy_last_ = applySlew(0.0f, vy_last_, param_slew_xy_, dt_s);
    vz_last_ = applySlew(0.0f, vz_last_, param_slew_z_, dt_s);

    return Eigen::Vector3f(vx_last_, vy_last_, vz_last_);
}

Eigen::Vector3f RingPassController::update(float dt_s)
{
    if (!attitudeReady() || targetTimedOut() || finished_) {
        return computeHoldCommand(dt_s);
    }

    return computeVelocityCommand(dt_s);
}