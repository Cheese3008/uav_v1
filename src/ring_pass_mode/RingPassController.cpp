#include "RingPassController.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>

#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/components/node_with_mode.hpp>
#include <px4_ros2/control/setpoint_types/experimental/trajectory.hpp>

namespace
{
constexpr char kVehicleAttitudeTopic[] = "/fmu/out/vehicle_attitude";
constexpr char kVehicleLocalPositionTopic[] = "/fmu/out/vehicle_local_position";

// Output hiện tại của RingDetector:
// PoseStamped.position = [x, y, z] trong body/drone frame.
constexpr char kTargetBodyPositionTopic[] = "/ring_detect/target_error_body_filtered";
constexpr char kTargetBodyVelocityTopic[] = "/ring_detect/target_velocity_body_filtered";
constexpr char kTargetValidTopic[] = "/ring_detect/target_valid";
constexpr char kRingDetectResetTopic[] = "/ring_detect/reset";

constexpr char kModeName[] = "RingPassMode";
constexpr bool kEnableDebugOutput = true;
}

RingPassController::RingPassController(rclcpp::Node& node)
    : ModeBase(node, px4_ros2::ModeBase::Settings{std::string(kModeName)})
    , node_(node)
{
    trajectorySetpoint_ = std::make_shared<px4_ros2::TrajectorySetpointType>(*this);

    vehicleLocalPositionSub_ = node_.create_subscription<px4_msgs::msg::VehicleLocalPosition>(
        kVehicleLocalPositionTopic,
        rclcpp::QoS(10).best_effort(),
        std::bind(&RingPassController::vehicleLocalPositionCallback, this, std::placeholders::_1));

    vehicleAttitudeSub_ = node_.create_subscription<px4_msgs::msg::VehicleAttitude>(
        kVehicleAttitudeTopic,
        rclcpp::QoS(10).best_effort(),
        std::bind(&RingPassController::vehicleAttitudeCallback, this, std::placeholders::_1));

    targetPositionSub_ = node_.create_subscription<geometry_msgs::msg::PoseStamped>(
        kTargetBodyPositionTopic,
        rclcpp::QoS(1).best_effort(),
        std::bind(&RingPassController::targetPositionCallback, this, std::placeholders::_1));

    targetVelocitySub_ = node_.create_subscription<geometry_msgs::msg::TwistStamped>(
        kTargetBodyVelocityTopic,
        rclcpp::QoS(1).best_effort(),
        std::bind(&RingPassController::targetVelocityCallback, this, std::placeholders::_1));

    targetValidSub_ = node_.create_subscription<std_msgs::msg::Bool>(
        kTargetValidTopic,
        rclcpp::QoS(1).best_effort(),
        std::bind(&RingPassController::targetValidCallback, this, std::placeholders::_1));

    detectorResetPub_ = node_.create_publisher<std_msgs::msg::String>(
        kRingDetectResetTopic,
        rclcpp::QoS(10).reliable());

    loadParameters();

    modeRequirements().manual_control = false;
    stateEnterTime_ = node_.now();

    reset();

    RCLCPP_INFO(node_.get_logger(), "[RingPassMode] external mode created. Mode=%s", kModeName);
}

void RingPassController::loadParameters()
{
    node_.declare_parameter<double>("target_timeout", targetTimeoutSec_);
    node_.declare_parameter<double>("detector_reset_ignore_time", detectorResetIgnoreTimeSec_);
    node_.declare_parameter<double>("ringpass_min_time_before_finish", minTimeBeforeFinishSec_);
    node_.declare_parameter<double>("ringpass_min_forward_cmd_before_finish", minForwardCmdBeforeFinishMps_);
    node_.declare_parameter<double>("ring_mode_dt_min_sec", dtMinSec_);
    node_.declare_parameter<double>("ring_mode_dt_max_sec", dtMaxSec_);

    node_.declare_parameter<double>("ringpass_kp_y", kpY_);
    node_.declare_parameter<double>("ringpass_kd_y", kdY_);
    node_.declare_parameter<double>("ringpass_kp_z", kpZ_);
    node_.declare_parameter<double>("ringpass_kd_z", kdZ_);

    node_.declare_parameter<double>("ringpass_y_deadband", yDeadbandM_);
    node_.declare_parameter<double>("ringpass_z_deadband", zDeadbandM_);

    node_.declare_parameter<double>("ringpass_vy_max", vyMaxMps_);
    node_.declare_parameter<double>("ringpass_vz_max", vzMaxMps_);

    node_.declare_parameter<double>("ringpass_vx_min", vxMinMps_);
    node_.declare_parameter<double>("ringpass_vx_max", vxMaxMps_);
    node_.declare_parameter<double>("ringpass_yz_approach_radius", yzApproachRadiusM_);
    node_.declare_parameter<double>("ringpass_yz_full_speed_radius", yzFullSpeedRadiusM_);
    node_.declare_parameter<double>("ringpass_x_speed_curve_gain", xSpeedCurveGain_);

    node_.declare_parameter<double>("ringpass_target_z_offset", targetZOffsetM_);

    node_.declare_parameter<double>("ringpass_slew_xy", slewXyMps2_);
    node_.declare_parameter<double>("ringpass_slew_z", slewZMps2_);

    node_.get_parameter("target_timeout", targetTimeoutSec_);
    node_.get_parameter("detector_reset_ignore_time", detectorResetIgnoreTimeSec_);
    node_.get_parameter("ringpass_min_time_before_finish", minTimeBeforeFinishSec_);
    node_.get_parameter("ringpass_min_forward_cmd_before_finish", minForwardCmdBeforeFinishMps_);
    node_.get_parameter("ring_mode_dt_min_sec", dtMinSec_);
    node_.get_parameter("ring_mode_dt_max_sec", dtMaxSec_);

    node_.get_parameter("ringpass_kp_y", kpY_);
    node_.get_parameter("ringpass_kd_y", kdY_);
    node_.get_parameter("ringpass_kp_z", kpZ_);
    node_.get_parameter("ringpass_kd_z", kdZ_);

    node_.get_parameter("ringpass_y_deadband", yDeadbandM_);
    node_.get_parameter("ringpass_z_deadband", zDeadbandM_);

    node_.get_parameter("ringpass_vy_max", vyMaxMps_);
    node_.get_parameter("ringpass_vz_max", vzMaxMps_);

    node_.get_parameter("ringpass_vx_min", vxMinMps_);
    node_.get_parameter("ringpass_vx_max", vxMaxMps_);
    node_.get_parameter("ringpass_yz_approach_radius", yzApproachRadiusM_);
    node_.get_parameter("ringpass_yz_full_speed_radius", yzFullSpeedRadiusM_);
    node_.get_parameter("ringpass_x_speed_curve_gain", xSpeedCurveGain_);

    node_.get_parameter("ringpass_target_z_offset", targetZOffsetM_);

    node_.get_parameter("ringpass_slew_xy", slewXyMps2_);
    node_.get_parameter("ringpass_slew_z", slewZMps2_);

    targetTimeoutSec_ = std::max(targetTimeoutSec_, 0.01);
    detectorResetIgnoreTimeSec_ = std::max(detectorResetIgnoreTimeSec_, 0.0);
    minTimeBeforeFinishSec_ = std::max(minTimeBeforeFinishSec_, 0.0);
    minForwardCmdBeforeFinishMps_ = std::max(minForwardCmdBeforeFinishMps_, 0.0);
    dtMinSec_ = std::max(dtMinSec_, 1e-3);
    dtMaxSec_ = std::max(dtMaxSec_, dtMinSec_);

    yDeadbandM_ = std::max(yDeadbandM_, 0.0);
    zDeadbandM_ = std::max(zDeadbandM_, 0.0);

    vyMaxMps_ = std::max(vyMaxMps_, 0.0);
    vzMaxMps_ = std::max(vzMaxMps_, 0.0);

    vxMinMps_ = std::max(vxMinMps_, 0.0);
    vxMaxMps_ = std::max(vxMaxMps_, vxMinMps_);
    yzFullSpeedRadiusM_ = std::max(yzFullSpeedRadiusM_, 0.0);
    yzApproachRadiusM_ = std::max(yzApproachRadiusM_, yzFullSpeedRadiusM_ + 1e-3);
    xSpeedCurveGain_ = std::max(xSpeedCurveGain_, 1e-3);

    slewXyMps2_ = std::max(slewXyMps2_, 0.0);
    slewZMps2_ = std::max(slewZMps2_, 0.0);
    RCLCPP_INFO(
        node_.get_logger(),
        "[RingPassController] input body XYZ topic=%s velocity=%s valid=%s",
        kTargetBodyPositionTopic,
        kTargetBodyVelocityTopic,
        kTargetValidTopic);
}

void RingPassController::publishDetectorReset()
{
    if (!detectorResetPub_)
    {
        return;
    }

    std_msgs::msg::String msg;
    msg.data = "reset";
    detectorResetPub_->publish(msg);

    ignoreTargetUntil_ = node_.now() + rclcpp::Duration::from_seconds(detectorResetIgnoreTimeSec_);

    RCLCPP_WARN(
        node_.get_logger(),
        "[RingPassController] reset RingDetector lock/Kalman, ignore target for %.2f s",
        detectorResetIgnoreTimeSec_);
}

void RingPassController::reset()
{
    vxLastNed_ = 0.0;
    vyLastNed_ = 0.0;
    vzLastNed_ = 0.0;
    target_ = BodyTarget{};
    targetValidReceived_ = false;
    targetValidExternal_ = false;
    hasForwardCommandedInPass_ = false;
    completedReported_ = false;

    publishDetectorReset();
}

void RingPassController::vehicleAttitudeCallback(
    const px4_msgs::msg::VehicleAttitude::SharedPtr msg)
{
    vehicleAttitudeMsg_ = *msg;
    hasVehicleAttitude_ = true;
}

void RingPassController::targetPositionCallback(
    const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (node_.now() < ignoreTargetUntil_)
    {
        return;
    }

    if (targetValidReceived_ && !targetValidExternal_)
    {
        target_.valid = false;
        return;
    }

    target_.positionXyz.x() = msg->pose.position.x;
    target_.positionXyz.y() = msg->pose.position.y;
    target_.positionXyz.z() = msg->pose.position.z;

    target_.stamp = (msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0)
        ? node_.now()
        : rclcpp::Time(msg->header.stamp);

    target_.valid = true;
}

void RingPassController::targetVelocityCallback(
    const geometry_msgs::msg::TwistStamped::SharedPtr msg)
{
    if (node_.now() < ignoreTargetUntil_)
    {
        return;
    }

    target_.velocityXyz.x() = msg->twist.linear.x;
    target_.velocityXyz.y() = msg->twist.linear.y;
    target_.velocityXyz.z() = msg->twist.linear.z;
}

void RingPassController::targetValidCallback(
    const std_msgs::msg::Bool::SharedPtr msg)
{
    targetValidReceived_ = true;
    targetValidExternal_ = msg->data;

    if (!msg->data)
    {
        target_.valid = false;
    }
}

void RingPassController::vehicleLocalPositionCallback(
    const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg)
{
    currentPositionNed_.x() = msg->x;
    currentPositionNed_.y() = msg->y;
    currentPositionNed_.z() = msg->z;
    currentYawRad_ = msg->heading;
    hasVehicleLocalPosition_ = true;
}

bool RingPassController::attitudeReady() const
{
    return hasVehicleAttitude_;
}

bool RingPassController::targetTimedOut() const
{
    if (node_.now() < ignoreTargetUntil_)
    {
        return true;
    }

    if (targetValidReceived_ && !targetValidExternal_)
    {
        return true;
    }

    if (!target_.valid)
    {
        return true;
    }

    const double ageSec = (node_.now() - target_.stamp).seconds();
    return ageSec > targetTimeoutSec_;
}

bool RingPassController::isReady() const
{
    return attitudeReady();
}

bool RingPassController::hasValidTarget() const
{
    return !targetTimedOut();
}


double RingPassController::vehicleYawRad() const
{
    const auto& q = vehicleAttitudeMsg_.q;

    const double w = q[0];
    const double x = q[1];
    const double y = q[2];
    const double z = q[3];

    const double sinyCosp = 2.0 * (w * z + x * y);
    const double cosyCosp = 1.0 - 2.0 * (y * y + z * z);

    return std::atan2(sinyCosp, cosyCosp);
}

double RingPassController::applySlew(
    double command,
    double previous,
    double accelerationLimit,
    double dtSec) const
{
    const double safeDtSec = std::max(dtSec, 1e-3);
    const double maxDelta = accelerationLimit * safeDtSec;
    const double delta = std::clamp(command - previous, -maxDelta, maxDelta);
    return previous + delta;
}

double RingPassController::computeBodyVelocityX(double yzErrorRadius) const
{
    if (yzErrorRadius > yzApproachRadiusM_)
    {
        return 0.0;
    }

    const double radius = std::min(yzErrorRadius, yzApproachRadiusM_);
    const double denom = std::log1p(xSpeedCurveGain_ * yzApproachRadiusM_);

    double ratio = 1.0 -
        (std::log1p(xSpeedCurveGain_ * radius) / std::max(denom, 1e-6));

    ratio = std::clamp(ratio, 0.0, 1.0);
    return vxMinMps_ + (vxMaxMps_ - vxMinMps_) * ratio;
}

Eigen::Vector2d RingPassController::bodyXyToNedXy(
    const Eigen::Vector2d& velocityBodyXy,
    double yawRad) const
{
    const double cosYaw = std::cos(yawRad);
    const double sinYaw = std::sin(yawRad);

    Eigen::Vector2d velocityNedXy;
    velocityNedXy.x() = cosYaw * velocityBodyXy.x() - sinYaw * velocityBodyXy.y();
    velocityNedXy.y() = sinYaw * velocityBodyXy.x() + cosYaw * velocityBodyXy.y();

    return velocityNedXy;
}

Eigen::Vector3f RingPassController::computeVelocityCommand(double dtSec)
{
    // RingDetector hiện tại publish vị trí tương đối body XYZ:
    // x > 0: vòng ở phía trước drone.
    // y > 0: vòng lệch sang phải drone.
    // z > 0: vòng lệch xuống dưới drone.
    const double errorX = target_.positionXyz.x();
    const double errorY = target_.positionXyz.y();
    const double errorZ = target_.positionXyz.z() - targetZOffsetM_;

    const double velocityY = target_.velocityXyz.y();
    const double velocityZ = target_.velocityXyz.z();

    const double yzErrorRadius = std::hypot(errorY, errorZ);

    const double vxBodyCmd = computeBodyVelocityX(yzErrorRadius);
    if (state_ == RingModeState::RingPass && vxBodyCmd >= minForwardCmdBeforeFinishMps_)
    {
        hasForwardCommandedInPass_ = true;
    }

    double vyBodyCmd = kpY_ * errorY - kdY_ * velocityY;
    if (std::abs(errorY) < yDeadbandM_)
    {
        vyBodyCmd = -kdY_ * velocityY;
    }
    vyBodyCmd = std::clamp(vyBodyCmd, -vyMaxMps_, vyMaxMps_);

    double vzNedCmd = kpZ_ * errorZ - kdZ_ * velocityZ;
    if (std::abs(errorZ) < zDeadbandM_)
    {
        vzNedCmd = -kdZ_ * velocityZ;
    }
    vzNedCmd = std::clamp(vzNedCmd, -vzMaxMps_, vzMaxMps_);

    const Eigen::Vector2d velocityNedXy =
        bodyXyToNedXy(Eigen::Vector2d(vxBodyCmd, vyBodyCmd), vehicleYawRad());

    vxLastNed_ = applySlew(velocityNedXy.x(), vxLastNed_, slewXyMps2_, dtSec);
    vyLastNed_ = applySlew(velocityNedXy.y(), vyLastNed_, slewXyMps2_, dtSec);
    vzLastNed_ = applySlew(vzNedCmd, vzLastNed_, slewZMps2_, dtSec);

    RCLCPP_INFO_THROTTLE(
        node_.get_logger(),
        *(node_.get_clock()),
        500,
        "[RingPassController] err_body_xyz=(%.2f %.2f %.2f) vel_ned=(%.2f %.2f %.2f) r_yz=%.2f",
        errorX,
        errorY,
        errorZ,
        vxLastNed_,
        vyLastNed_,
        vzLastNed_,
        yzErrorRadius);

    return Eigen::Vector3f(
        static_cast<float>(vxLastNed_),
        static_cast<float>(vyLastNed_),
        static_cast<float>(vzLastNed_));
}

Eigen::Vector3f RingPassController::computeHoldCommand(double dtSec)
{
    vxLastNed_ = applySlew(0.0, vxLastNed_, slewXyMps2_, dtSec);
    vyLastNed_ = applySlew(0.0, vyLastNed_, slewXyMps2_, dtSec);
    vzLastNed_ = applySlew(0.0, vzLastNed_, slewZMps2_, dtSec);

    return Eigen::Vector3f(
        static_cast<float>(vxLastNed_),
        static_cast<float>(vyLastNed_),
        static_cast<float>(vzLastNed_));
}

Eigen::Vector3f RingPassController::update(double dtSec)
{
    if (!attitudeReady() || targetTimedOut())
    {
        return computeHoldCommand(dtSec);
    }

    return computeVelocityCommand(dtSec);
}


void RingPassController::onActivate()
{
    hasVehicleLocalPosition_ = false;
    state_ = RingModeState::WaitLocalPosition;
    stateEnterTime_ = node_.now();

    reset();

    RCLCPP_INFO(node_.get_logger(), "[RingPassMode] activated");
}

void RingPassController::onDeactivate()
{
    reset();
    publishVelocitySetpoint(Eigen::Vector3f::Zero());
    RCLCPP_INFO(node_.get_logger(), "[RingPassMode] deactivated");
}

void RingPassController::enterState(RingModeState newState)
{
    if (state_ == newState)
    {
        return;
    }

    state_ = newState;
    stateEnterTime_ = node_.now();

    if (newState == RingModeState::RingPass)
    {
        hasForwardCommandedInPass_ = false;
        ringPassStartTime_ = stateEnterTime_;
    }

    RCLCPP_INFO(node_.get_logger(), "[RingPassMode] enter state: %s", stateToString(state_));
}

void RingPassController::publishHoldPosition()
{
    if (!trajectorySetpoint_)
    {
        return;
    }

    trajectorySetpoint_->updatePosition(currentPositionNed_);
}

void RingPassController::publishVelocitySetpoint(const Eigen::Vector3f& velocityNed)
{
    if (!trajectorySetpoint_)
    {
        return;
    }

    trajectorySetpoint_->update(
        velocityNed,
        std::nullopt,
        std::optional<float>{currentYawRad_},
        std::nullopt);
}

double RingPassController::sanitizeDt(double dtSec) const
{
    if (!std::isfinite(dtSec))
    {
        return dtMinSec_;
    }

    return std::clamp(dtSec, dtMinSec_, dtMaxSec_);
}

const char* RingPassController::stateToString(RingModeState state) const
{
    switch (state)
    {
    case RingModeState::WaitLocalPosition:
        return "WAIT_LOCAL_POSITION";
    case RingModeState::WaitRingTarget:
        return "WAIT_RING_TARGET";
    case RingModeState::RingPass:
        return "RING_PASS";
    case RingModeState::Finished:
        return "FINISHED";
    default:
        return "UNKNOWN";
    }
}

void RingPassController::updateSetpoint(float dtSec)
{
    const double safeDtSec = sanitizeDt(static_cast<double>(dtSec));

    if (!hasVehicleLocalPosition_)
    {
        RCLCPP_WARN_THROTTLE(
            node_.get_logger(),
            *(node_.get_clock()),
            1000,
            "[RingPassMode] waiting for /fmu/out/vehicle_local_position");

        publishVelocitySetpoint(Eigen::Vector3f::Zero());
        return;
    }

    switch (state_)
    {
    case RingModeState::WaitLocalPosition:
    {
        publishHoldPosition();
        enterState(RingModeState::WaitRingTarget);
        break;
    }

    case RingModeState::WaitRingTarget:
    {
        (void)update(safeDtSec);
        publishHoldPosition();

        if (!isReady())
        {
            RCLCPP_WARN_THROTTLE(
                node_.get_logger(),
                *(node_.get_clock()),
                1000,
                "[RingPassMode] waiting for /fmu/out/vehicle_attitude");
            break;
        }

        if (!hasValidTarget())
        {
            if (node_.now() < ignoreTargetUntil_)
            {
                RCLCPP_WARN_THROTTLE(
                    node_.get_logger(),
                    *(node_.get_clock()),
                    1000,
                    "[RingPassMode] waiting detector reset settle before accepting ring target");
            }
            else
            {
                RCLCPP_WARN_THROTTLE(
                    node_.get_logger(),
                    *(node_.get_clock()),
                    1000,
                    "[RingPassMode] waiting for /ring_detect/target_error_body_filtered");
            }
            break;
        }

        enterState(RingModeState::RingPass);
        break;
    }

    case RingModeState::RingPass:
    {
        if (!hasValidTarget())
        {
            const double passDurationSec = (node_.now() - ringPassStartTime_).seconds();
            const bool canFinishByLostTarget =
                hasForwardCommandedInPass_ &&
                passDurationSec >= minTimeBeforeFinishSec_;

            if (canFinishByLostTarget)
            {
                RCLCPP_INFO(
                    node_.get_logger(),
                    "[RingPassMode] ring target lost after %.2f s with forward command, treat as passed through ring",
                    passDurationSec);

                publishVelocitySetpoint(Eigen::Vector3f::Zero());
                enterState(RingModeState::Finished);
            }
            else
            {
                RCLCPP_WARN(
                    node_.get_logger(),
                    "[RingPassMode] target lost too early, not finish. pass_time=%.2f/%.2f has_forward=%d",
                    passDurationSec,
                    minTimeBeforeFinishSec_,
                    static_cast<int>(hasForwardCommandedInPass_));

                target_ = BodyTarget{};
                vxLastNed_ = 0.0;
                vyLastNed_ = 0.0;
                vzLastNed_ = 0.0;
                publishVelocitySetpoint(Eigen::Vector3f::Zero());
                enterState(RingModeState::WaitRingTarget);
            }
            break;
        }

        const Eigen::Vector3f velocityCmdNed = update(safeDtSec);
        publishVelocitySetpoint(velocityCmdNed);
        break;
    }

    case RingModeState::Finished:
    default:
    {
        publishVelocitySetpoint(Eigen::Vector3f::Zero());
        if (!completedReported_)
        {
            completedReported_ = true;
            completed(px4_ros2::Result::Success);
        }
        break;
    }
    }
}

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<px4_ros2::NodeWithMode<RingPassController>>(kModeName, kEnableDebugOutput));
    rclcpp::shutdown();
    return 0;
}