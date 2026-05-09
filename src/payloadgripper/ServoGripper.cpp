#include "ServoGripper.hpp"

#include <algorithm>
#include <cstdint>

namespace
{
constexpr std::uint32_t kVehicleCmdDoSetActuator = 187;
}

ServoGripper::ServoGripper(rclcpp::Node &node)
    : node_(node)
{
    loadParameters();

    vehicleCommandPublisher_ = node_.create_publisher<px4_msgs::msg::VehicleCommand>(
        vehicleCommandTopic_,
        rclcpp::QoS(10).reliable());

    debugStatePublisher_ = node_.create_publisher<std_msgs::msg::Bool>(
        gripperDebugTopic_,
        rclcpp::QoS(1).reliable().transient_local());

    publishDebugState();

    RCLCPP_INFO(
        node_.get_logger(),
        "[ServoGripper] started | vehicle_command=%s debug=%s actuator_index=%d grab=%.2f release=%.2f repeat=%d",
        vehicleCommandTopic_.c_str(),
        gripperDebugTopic_.c_str(),
        actuatorIndex_,
        static_cast<double>(grabValue_),
        static_cast<double>(releaseValue_),
        repeatCommandCount_);
}

void ServoGripper::loadParameters()
{
    node_.declare_parameter<std::string>("topics.vehicle_command", "/fmu/in/vehicle_command");
    node_.declare_parameter<std::string>("topics.gripper_debug", "/payload_gripper/gripper_grabbed_debug");

    node_.declare_parameter<int>("gripper.actuator_index", 1);
    node_.declare_parameter<float>("gripper.grab_value", 1.0f);
    node_.declare_parameter<float>("gripper.release_value", -1.0f);
    node_.declare_parameter<int>("gripper.repeat_command_count", 3);
    node_.declare_parameter<bool>("gripper.initial_grabbed_state", false);

    node_.get_parameter("topics.vehicle_command", vehicleCommandTopic_);
    node_.get_parameter("topics.gripper_debug", gripperDebugTopic_);

    node_.get_parameter("gripper.actuator_index", actuatorIndex_);
    node_.get_parameter("gripper.grab_value", grabValue_);
    node_.get_parameter("gripper.release_value", releaseValue_);
    node_.get_parameter("gripper.repeat_command_count", repeatCommandCount_);
    node_.get_parameter("gripper.initial_grabbed_state", grabbed_);

    actuatorIndex_ = std::clamp(actuatorIndex_, 1, 6);
    grabValue_ = std::clamp(grabValue_, -1.0f, 1.0f);
    releaseValue_ = std::clamp(releaseValue_, -1.0f, 1.0f);
    repeatCommandCount_ = std::max(1, repeatCommandCount_);
}

void ServoGripper::grabAndHold(bool forceCommand)
{
    if (grabbed_ && !forceCommand)
    {
        publishDebugState();
        return;
    }

    publishActuatorValue(grabValue_);
    grabbed_ = true;
    publishDebugState();

    RCLCPP_WARN(
        node_.get_logger(),
        "[ServoGripper] GRAB/HOLD | actuator_index=%d value=%.2f",
        actuatorIndex_,
        static_cast<double>(grabValue_));
}

void ServoGripper::release(bool forceCommand)
{
    if (!grabbed_ && !forceCommand)
    {
        publishDebugState();
        return;
    }

    publishActuatorValue(releaseValue_);
    grabbed_ = false;
    publishDebugState();

    RCLCPP_WARN(
        node_.get_logger(),
        "[ServoGripper] RELEASE | actuator_index=%d value=%.2f",
        actuatorIndex_,
        static_cast<double>(releaseValue_));
}

void ServoGripper::resetState(bool grabbed)
{
    grabbed_ = grabbed;
    publishDebugState();
}

void ServoGripper::publishDebugState() const
{
    if (!debugStatePublisher_)
    {
        return;
    }

    std_msgs::msg::Bool debugMsg;
    debugMsg.data = grabbed_;
    debugStatePublisher_->publish(debugMsg);
}

bool ServoGripper::isGrabbed() const
{
    return grabbed_;
}

void ServoGripper::publishActuatorValue(float actuatorValue) const
{
    px4_msgs::msg::VehicleCommand command{};

    command.timestamp = static_cast<std::uint64_t>(node_.now().nanoseconds() / 1000ULL);

    command.param1 = 0.0f;
    command.param2 = 0.0f;
    command.param3 = 0.0f;
    command.param4 = 0.0f;
    command.param5 = 0.0;
    command.param6 = 0.0;
    command.param7 = 0.0f;

    switch (actuatorIndex_)
    {
    case 1:
        command.param1 = actuatorValue;
        break;
    case 2:
        command.param2 = actuatorValue;
        break;
    case 3:
        command.param3 = actuatorValue;
        break;
    case 4:
        command.param4 = actuatorValue;
        break;
    case 5:
        command.param5 = static_cast<double>(actuatorValue);
        break;
    case 6:
        command.param6 = static_cast<double>(actuatorValue);
        break;
    default:
        command.param1 = actuatorValue;
        break;
    }

    command.command = kVehicleCmdDoSetActuator;
    command.target_system = 1;
    command.target_component = 1;
    command.source_system = 1;
    command.source_component = 1;
    command.confirmation = 0;
    command.from_external = true;

    const int safeRepeatCount = std::max(1, repeatCommandCount_);
    for (int index = 0; index < safeRepeatCount; ++index)
    {
        vehicleCommandPublisher_->publish(command);
    }
}
