#include "main.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

#include <px4_ros2/components/node_with_mode.hpp>

namespace
{
constexpr char kVehicleLocalPositionTopic[] = "/fmu/out/vehicle_local_position";

const std::string kModeName = "RingPassMode";
constexpr bool kEnableDebugOutput = true;
}

MissionNode::MissionNode(rclcpp::Node& node)
    : ModeBase(node, kModeName)
    , node_(node)
    , ringPassController_(node)
{
    loadParameters();
    setupInterfaces();

    trajectorySetpoint_ =
        std::make_shared<px4_ros2::TrajectorySetpointType>(*this);

    modeRequirements().manual_control = false;

    stateEnterTime_ = node_.now();

    RCLCPP_INFO(node_.get_logger(), "[RingPassMode] external mode created");
}

void MissionNode::loadParameters()
{
    node_.declare_parameter<float>("ring_mode_dt_min_sec", 0.005f);
    node_.declare_parameter<float>("ring_mode_dt_max_sec", 0.10f);

    node_.get_parameter("ring_mode_dt_min_sec", paramDtMinSec_);
    node_.get_parameter("ring_mode_dt_max_sec", paramDtMaxSec_);

    paramDtMinSec_ = std::max(paramDtMinSec_, 1e-3f);
    paramDtMaxSec_ = std::max(paramDtMaxSec_, paramDtMinSec_);

    RCLCPP_INFO(
        node_.get_logger(),
        "[RingPassMode] params: dt_min=%.4f dt_max=%.4f",
        paramDtMinSec_,
        paramDtMaxSec_);
}

void MissionNode::setupInterfaces()
{
    vehicleLocalPositionSub_ =
        node_.create_subscription<px4_msgs::msg::VehicleLocalPosition>(
            kVehicleLocalPositionTopic,
            rclcpp::QoS(10).best_effort(),
            std::bind(&MissionNode::vehicleLocalPositionCallback, this, std::placeholders::_1));
}

void MissionNode::vehicleLocalPositionCallback(
    const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg)
{
    currentPosition_.x() = msg->x;
    currentPosition_.y() = msg->y;
    currentPosition_.z() = msg->z;
    currentYaw_ = msg->heading;

    hasVehicleLocalPosition_ = true;
}

void MissionNode::onActivate()
{
    hasVehicleLocalPosition_ = false;
    stateFirstTick_ = true;
    stateEnterTime_ = node_.now();
    state_ = RingModeState::WAIT_LOCAL_POSITION;

    ringPassController_.reset();

    RCLCPP_INFO(node_.get_logger(), "[RingPassMode] activated, waiting ring target");
}

void MissionNode::onDeactivate()
{
    ringPassController_.reset();
    RCLCPP_INFO(node_.get_logger(), "[RingPassMode] deactivated");
}

void MissionNode::enterState(RingModeState newState)
{
    if (state_ == newState)
    {
        return;
    }

    state_ = newState;
    stateEnterTime_ = node_.now();
    stateFirstTick_ = true;

    RCLCPP_INFO(
        node_.get_logger(),
        "[RingPassMode] enter state: %s",
        ringModeStateToString(state_));
}

void MissionNode::publishHoldPosition()
{
    if (!trajectorySetpoint_)
    {
        return;
    }

    trajectorySetpoint_->updatePosition(currentPosition_);
}

void MissionNode::publishVelocitySetpoint(const Eigen::Vector3f& velocityNed)
{
    if (!trajectorySetpoint_)
    {
        return;
    }

    trajectorySetpoint_->update(
        velocityNed,
        std::nullopt,
        std::optional<float>{currentYaw_},
        std::nullopt);
}

float MissionNode::sanitizeDt(float dtSec) const
{
    if (!std::isfinite(dtSec))
    {
        return paramDtMinSec_;
    }

    return std::clamp(dtSec, paramDtMinSec_, paramDtMaxSec_);
}

const char* MissionNode::ringModeStateToString(RingModeState state) const
{
    switch (state)
    {
    case RingModeState::WAIT_LOCAL_POSITION:
        return "WAIT_LOCAL_POSITION";
    case RingModeState::WAIT_RING_TARGET:
        return "WAIT_RING_TARGET";
    case RingModeState::RING_PASS:
        return "RING_PASS";
    case RingModeState::FINISHED:
        return "FINISHED";
    default:
        return "UNKNOWN";
    }
}

void MissionNode::updateSetpoint(float dt_s)
{
    const float safeDtSec = sanitizeDt(dt_s);

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
    case RingModeState::WAIT_LOCAL_POSITION:
    {
        publishHoldPosition();
        enterState(RingModeState::WAIT_RING_TARGET);
        break;
    }

    case RingModeState::WAIT_RING_TARGET:
    {
        // Vẫn gọi update để vận tốc cũ trong RingPassController được slew về 0 nếu trước đó mất target.
        (void)ringPassController_.update(safeDtSec);
        publishHoldPosition();

        if (!ringPassController_.isReady())
        {
            RCLCPP_WARN_THROTTLE(
                node_.get_logger(),
                *(node_.get_clock()),
                1000,
                "[RingPassMode] waiting for vehicle attitude");
            break;
        }

        if (!ringPassController_.hasValidTarget())
        {
            RCLCPP_WARN_THROTTLE(
                node_.get_logger(),
                *(node_.get_clock()),
                1000,
                "[RingPassMode] waiting for ring target: /ring_pass/target_error_body_filtered");
            break;
        }

        enterState(RingModeState::RING_PASS);
        break;
    }

    case RingModeState::RING_PASS:
    {
        if (!ringPassController_.hasValidTarget())
        {
            RCLCPP_WARN_THROTTLE(
                node_.get_logger(),
                *(node_.get_clock()),
                1000,
                "[RingPassMode] ring target lost, hold current position");

            (void)ringPassController_.update(safeDtSec);
            publishHoldPosition();
            enterState(RingModeState::WAIT_RING_TARGET);
            break;
        }

        const Eigen::Vector3f velocityCmdNed = ringPassController_.update(safeDtSec);

        if (ringPassController_.isFinished())
        {
            publishVelocitySetpoint(Eigen::Vector3f::Zero());
            enterState(RingModeState::FINISHED);
            break;
        }

        publishVelocitySetpoint(velocityCmdNed);

        RCLCPP_INFO_THROTTLE(
            node_.get_logger(),
            *(node_.get_clock()),
            500,
            "[RingPassMode] vel_ned=(%.2f %.2f %.2f) pos=(%.2f %.2f %.2f)",
            velocityCmdNed.x(), velocityCmdNed.y(), velocityCmdNed.z(),
            currentPosition_.x(), currentPosition_.y(), currentPosition_.z());
        break;
    }

    case RingModeState::FINISHED:
    default:
    {
        publishVelocitySetpoint(Eigen::Vector3f::Zero());
        completed(px4_ros2::Result::Success);
        break;
    }
    }
}

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<px4_ros2::NodeWithMode<MissionNode>>(kModeName, kEnableDebugOutput));
    rclcpp::shutdown();
    return 0;
}
