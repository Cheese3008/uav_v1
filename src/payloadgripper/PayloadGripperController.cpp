#include "PayloadGripperController.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>

#include <px4_ros2/components/node_with_mode.hpp>

namespace
{
const std::string kModeName = "GRIPPER_CTRL";
constexpr bool kEnableDebugOutput = true;
}

PayloadGripperController::PayloadGripperController(rclcpp::Node &node)
    : ModeBase(node, kModeName),
      node_(node)
{
    trajectorySetpoint_ = std::make_shared<px4_ros2::TrajectorySetpointType>(*this);
    vehicleLocalPosition_ = std::make_shared<px4_ros2::OdometryLocalPosition>(*this);

    loadParameters();

    servoGripper_ = std::make_unique<ServoGripper>(node_);

    objectPoseSub_ = node_.create_subscription<geometry_msgs::msg::PoseStamped>(
        objectPoseTopic_,
        rclcpp::QoS(1).best_effort(),
        std::bind(&PayloadGripperController::objectPoseCallback, this, std::placeholders::_1));

    if (!objectValidTopic_.empty())
    {
        objectValidSub_ = node_.create_subscription<std_msgs::msg::Bool>(
            objectValidTopic_,
            rclcpp::QoS(1).best_effort(),
            std::bind(&PayloadGripperController::objectValidCallback, this, std::placeholders::_1));
    }

    modeRequirements().manual_control = false;

    RCLCPP_INFO(node_.get_logger(), "[PG] PayloadGripperController started. Mode=%s", kModeName.c_str());
}

void PayloadGripperController::loadParameters()
{
    node_.declare_parameter<std::string>("topics.object_pose", "/cube_detector/target_pose_world_filtered");
    node_.declare_parameter<std::string>("topics.object_valid", "/cube_detector/target_valid");
    node_.declare_parameter<float>("object_timeout", 1.0f);

    node_.declare_parameter<float>("xy_kp", 0.8f);
    node_.declare_parameter<float>("xy_deadband", 0.03f);
    node_.declare_parameter<float>("xy_max_velocity", 1.5f);
    node_.declare_parameter<float>("slew_acc", 0.9f);

    node_.declare_parameter<float>("center_gate_radius", 0.38f);
    node_.declare_parameter<float>("approach_settle_time", 0.50f);

    node_.declare_parameter<float>("grab_target_altitude", 0.35f);
    node_.declare_parameter<float>("grab_altitude_tolerance", 0.03f);
    node_.declare_parameter<float>("descend_velocity", 0.25f);

    node_.declare_parameter<float>("descend_timeout", 8.0f);
    node_.declare_parameter<float>("grab_ready_settle_time", 0.50f);
    node_.declare_parameter<float>("climb_velocity", 0.25f);
    node_.declare_parameter<float>("climb_altitude_tolerance", 0.05f);
    node_.declare_parameter<float>("climb_timeout", 8.0f);

    node_.declare_parameter<bool>("gripper.release_on_deactivate", false);
    node_.declare_parameter<bool>("gripper.force_grab_command", true);

    node_.get_parameter("topics.object_pose", objectPoseTopic_);
    node_.get_parameter("topics.object_valid", objectValidTopic_);
    node_.get_parameter("object_timeout", paramObjectTimeout_);

    node_.get_parameter("xy_kp", paramXyKp_);
    node_.get_parameter("xy_deadband", paramXyDeadband_);
    node_.get_parameter("xy_max_velocity", paramXyMaxVelocity_);
    node_.get_parameter("slew_acc", paramSlewAcc_);

    node_.get_parameter("center_gate_radius", paramCenterGateRadius_);
    node_.get_parameter("approach_settle_time", paramApproachSettleTime_);

    node_.get_parameter("grab_target_altitude", paramGrabTargetAltitude_);
    node_.get_parameter("grab_altitude_tolerance", paramGrabAltitudeTolerance_);
    node_.get_parameter("descend_velocity", paramDescendVelocity_);

    node_.get_parameter("descend_timeout", paramDescendTimeout_);
    node_.get_parameter("grab_ready_settle_time", paramGrabReadySettleTime_);
    node_.get_parameter("climb_velocity", paramClimbVelocity_);
    node_.get_parameter("climb_altitude_tolerance", paramClimbAltitudeTolerance_);
    node_.get_parameter("climb_timeout", paramClimbTimeout_);

    node_.get_parameter("gripper.release_on_deactivate", paramReleaseOnDeactivate_);
    node_.get_parameter("gripper.force_grab_command", paramForceGrabCommand_);

    paramObjectTimeout_ = std::max(paramObjectTimeout_, 0.05f);

    paramXyKp_ = std::max(paramXyKp_, 0.0f);
    paramXyDeadband_ = std::max(paramXyDeadband_, 0.0f);
    paramXyMaxVelocity_ = std::max(paramXyMaxVelocity_, 0.0f);
    paramSlewAcc_ = std::max(paramSlewAcc_, 0.0f);

    paramCenterGateRadius_ = std::max(paramCenterGateRadius_, 1e-3f);
    paramApproachSettleTime_ = std::max(paramApproachSettleTime_, 0.0f);

    paramGrabTargetAltitude_ = std::max(paramGrabTargetAltitude_, 0.01f);
    paramGrabAltitudeTolerance_ = std::max(paramGrabAltitudeTolerance_, 0.005f);
    paramDescendVelocity_ = std::max(paramDescendVelocity_, 0.0f);

    paramDescendTimeout_ = std::max(paramDescendTimeout_, 0.1f);
    paramGrabReadySettleTime_ = std::max(paramGrabReadySettleTime_, 0.0f);
    paramClimbVelocity_ = std::max(paramClimbVelocity_, 0.0f);
    paramClimbAltitudeTolerance_ = std::max(paramClimbAltitudeTolerance_, 0.005f);
    paramClimbTimeout_ = std::max(paramClimbTimeout_, 0.1f);
}

void PayloadGripperController::onActivate()
{
    controllerActive_ = true;

    resetMissionRuntimeState();
    releaseGripperForNewMission();
    switchToState(State::SearchObject);
}

void PayloadGripperController::resetMissionRuntimeState()
{
    objectWorld_.validPose = false;
    objectLostPrev_ = true;

    approachSettledTime_ = 0.0f;
    descendTime_ = 0.0f;
    grabReadySettledTime_ = 0.0f;
    climbTime_ = 0.0f;

    vxFiltered_ = 0.0f;
    vyFiltered_ = 0.0f;

    grabCommandSent_ = false;
    completedReported_ = false;

    hasSavedReturnAltitude_ = false;
    savedReturnAltitude_ = 0.0f;
}

void PayloadGripperController::releaseGripperForNewMission()
{
    // Luôn mở bộ gắp ở đầu mode để mission mới bắt đầu từ trạng thái sẵn sàng.
    // forceCommand=true giúp vẫn gửi lệnh release xuống PX4 dù latch nội bộ đang báo opened.
    servoGripper_->release(true);

    RCLCPP_WARN(
        node_.get_logger(),
        "[PG] Mode activated. Gripper RELEASE command sent before SearchObject.");
}

void PayloadGripperController::completeMissionOnce()
{
    if (completedReported_)
    {
        return;
    }

    completedReported_ = true;
    ModeBase::completed(px4_ros2::Result::Success);
}

void PayloadGripperController::onDeactivate()
{
    controllerActive_ = false;
    holdPosition();

    if (paramReleaseOnDeactivate_)
    {
        servoGripper_->release(true);
    }
    else
    {
        servoGripper_->publishDebugState();
    }
}

void PayloadGripperController::objectPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (!controllerActive_)
    {
        return;
    }

    if (objectValidReceived_ && !objectValidExternal_)
    {
        objectWorld_.validPose = false;
        return;
    }

    rclcpp::Time msgTimestamp = msg->header.stamp;
    if (msgTimestamp.nanoseconds() == 0)
    {
        msgTimestamp = node_.now();
    }

    objectWorld_.positionWorld = Eigen::Vector3f(
        static_cast<float>(msg->pose.position.x),
        static_cast<float>(msg->pose.position.y),
        static_cast<float>(msg->pose.position.z));

    objectWorld_.timestamp = msgTimestamp;
    objectWorld_.validPose = true;
}

void PayloadGripperController::objectValidCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
    objectValidReceived_ = true;
    objectValidExternal_ = msg->data;

    if (!msg->data)
    {
        objectWorld_.validPose = false;
    }
}

void PayloadGripperController::updateSetpoint(float dt_s)
{
    const bool objectLost = checkObjectLost();

    if (state_ == State::SearchObject || state_ == State::ApproachObject)
    {
        if (objectLost && !objectLostPrev_)
        {
            RCLCPP_WARN(node_.get_logger(), "[PG] Object lost in state=%s", stateName(state_).c_str());
        }
        else if (!objectLost && objectLostPrev_)
        {
            RCLCPP_INFO(node_.get_logger(), "[PG] Object acquired");
        }
    }
    objectLostPrev_ = objectLost;

    switch (state_)
    {
    case State::SearchObject:
        handleSearchObjectState(objectLost);
        break;

    case State::ApproachObject:
        handleApproachObjectState(dt_s, objectLost);
        break;

    case State::DescendToGrabHeight:
        handleDescendToGrabHeightState(dt_s);
        break;

    case State::GrabReady:
        handleGrabReadyState(dt_s);
        break;

    case State::ClimbToSavedAltitude:
        handleClimbToSavedAltitudeState(dt_s);
        break;
    }
}

void PayloadGripperController::handleSearchObjectState(bool objectLost)
{
    RCLCPP_INFO_THROTTLE(
        node_.get_logger(),
        *(node_.get_clock()),
        1000,
        "[PG][SearchObject] objectLost=%d validPose=%d",
        static_cast<int>(objectLost),
        static_cast<int>(objectWorld_.validPose));

    if (!objectLost)
    {
        approachSettledTime_ = 0.0f;
        switchToState(State::ApproachObject);
        return;
    }

    holdPosition();
}

void PayloadGripperController::handleApproachObjectState(float dt_s, bool objectLost)
{
    if (objectLost)
    {
        switchToState(State::SearchObject);
        holdPosition();
        return;
    }

    const Eigen::Vector3f dronePosition = vehicleLocalPosition_->positionNed();
    const Eigen::Vector3f objectPosition = objectWorld_.positionWorld;
    const Eigen::Vector3f relativePosition = objectPosition - dronePosition;

    const Eigen::Vector2f errorXY(relativePosition.x(), relativePosition.y());
    const float lateralError = errorXY.norm();

    // Sai số theo bán kính R:
    // - Nếu UAV đã nằm trong vùng R quanh object thì controlErrorXY = 0
    // - Nếu nằm ngoài R thì chỉ điều khiển phần vượt ra ngoài R
    const Eigen::Vector2f controlErrorXY =
        computeRadialGateErrorXY(errorXY, paramCenterGateRadius_);

    const float radialError = controlErrorXY.norm();
    const Eigen::Vector2f velocityXY = computeVelocityXY(controlErrorXY, dt_s);

    trajectorySetpoint_->update(
        Eigen::Vector3f(velocityXY.x(), velocityXY.y(), 0.0f),
        std::nullopt,
        std::nullopt);

    // Điều kiện vào vùng tâm bây giờ dùng radialError.
    // radialError = 0 nghĩa là UAV đã nằm trong bán kính R.
    if (radialError <= paramXyDeadband_)
    {
        approachSettledTime_ += std::max(dt_s, 0.0f);
    }
    else
    {
        approachSettledTime_ = 0.0f;
    }

    RCLCPP_INFO_THROTTLE(
        node_.get_logger(),
        *(node_.get_clock()),
        500,
        "[PG][ApproachObject] errXY=(%.3f, %.3f) lateral=%.3f R=%.3f radialErr=%.3f settled=%.2f/%.2f cmdXY=(%.3f, %.3f)",
        errorXY.x(),
        errorXY.y(),
        lateralError,
        paramCenterGateRadius_,
        radialError,
        approachSettledTime_,
        paramApproachSettleTime_,
        velocityXY.x(),
        velocityXY.y());

    if (approachSettledTime_ >= paramApproachSettleTime_)
    {
        savedReturnAltitude_ = getVehicleAltitude();
        hasSavedReturnAltitude_ = true;

        RCLCPP_WARN(
            node_.get_logger(),
            "[PG] Object centered and stable. Saved return altitude=%.3f m, then descend to grab.",
            savedReturnAltitude_);

        switchToState(State::DescendToGrabHeight);
    }
}

Eigen::Vector2f PayloadGripperController::computeRadialGateErrorXY(
    const Eigen::Vector2f &errorXY,
    float radius) const
{
    const float lateralError = errorXY.norm();
    const float safeRadius = std::max(radius, 1e-3f);

    if (lateralError <= safeRadius)
    {
        return Eigen::Vector2f(0.0f, 0.0f);
    }

    const float outsideError = lateralError - safeRadius;
    const Eigen::Vector2f directionXY = errorXY / std::max(lateralError, 1e-3f);

    return directionXY * outsideError;
}

void PayloadGripperController::handleDescendToGrabHeightState(float dt_s)
{
    descendTime_ += std::max(dt_s, 0.0f);

    const bool objectVisible = !checkObjectLost();
    Eigen::Vector2f velocityXY(0.0f, 0.0f);
    float lateralError = -1.0f;

    // Khi dang ha xuong, neu camera/tracker van con thay object thi tiep tuc chinh XY.
    // Neu object bi mat do khoi qua lon che het camera, khong quay lai Search nua,
    // chi dua lenh XY ve 0 va tiep tuc ha den do cao gap co dinh.
    if (objectVisible)
    {
        const Eigen::Vector3f dronePosition = vehicleLocalPosition_->positionNed();
        const Eigen::Vector3f objectPosition = objectWorld_.positionWorld;
        const Eigen::Vector3f relativePosition = objectPosition - dronePosition;
        const Eigen::Vector2f errorXY(relativePosition.x(), relativePosition.y());

        lateralError = errorXY.norm();
        velocityXY = computeVelocityXY(errorXY, dt_s);
    }
    else
    {
        // Khong co data object trong luc dang ha:
        // - Khong quay lai SearchObject
        // - Khong tiep tuc sua XY theo pose cu
        // - Dat truc tiep vx = 0, vy = 0
        // - Van tiep tuc ha xuong do cao gap co dinh
        vxFiltered_ = 0.0f;
        vyFiltered_ = 0.0f;
        velocityXY = Eigen::Vector2f(0.0f, 0.0f);
    }

    const float altitude = getVehicleAltitude();
    const float velocityZ = computeFixedDescentVelocity();
    const bool altitudeReady = isAtGrabAltitude();
    const bool timeout = descendTime_ >= paramDescendTimeout_;

    trajectorySetpoint_->update(
        Eigen::Vector3f(velocityXY.x(), velocityXY.y(), velocityZ),
        std::nullopt,
        std::nullopt);

    RCLCPP_INFO_THROTTLE(
        node_.get_logger(),
        *(node_.get_clock()),
        500,
        "[PG][DescendToGrabHeight] visible=%d lateral=%.3f altitude=%.3f target=%.3f tol=%.3f cmd=(%.3f, %.3f, %.3f) time=%.2f/%.2f",
        static_cast<int>(objectVisible),
        lateralError,
        altitude,
        paramGrabTargetAltitude_,
        paramGrabAltitudeTolerance_,
        velocityXY.x(),
        velocityXY.y(),
        velocityZ,
        descendTime_,
        paramDescendTimeout_);

    if (altitudeReady || timeout)
    {
        if (altitudeReady)
        {
            RCLCPP_WARN(
                node_.get_logger(),
                "[PG] DA TOI DO CAO GAP | altitude=%.3f m target=%.3f m tolerance=%.3f m",
                altitude,
                paramGrabTargetAltitude_,
                paramGrabAltitudeTolerance_);
        }
        else
        {
            RCLCPP_WARN(
                node_.get_logger(),
                "[PG] Descend timeout, switching to GrabReady anyway | altitude=%.3f m target=%.3f m",
                altitude,
                paramGrabTargetAltitude_);
        }

        switchToState(State::GrabReady);
    }
}

void PayloadGripperController::handleGrabReadyState(float dt_s)
{
    grabReadySettledTime_ += std::max(dt_s, 0.0f);

    trajectorySetpoint_->update(
        Eigen::Vector3f(0.0f, 0.0f, 0.0f),
        std::nullopt,
        std::nullopt);

    RCLCPP_INFO_THROTTLE(
        node_.get_logger(),
        *(node_.get_clock()),
        500,
        "[PG][GrabReady] holding. settled=%.2f/%.2f",
        grabReadySettledTime_,
        paramGrabReadySettleTime_);

    if (grabReadySettledTime_ >= paramGrabReadySettleTime_)
    {
        if (!grabCommandSent_)
        {
            servoGripper_->grabAndHold(paramForceGrabCommand_);
            grabCommandSent_ = true;

            RCLCPP_WARN(
                node_.get_logger(),
                "[PG] Grab height reached. Internal servo grab command sent. grabbed=%d",
                static_cast<int>(servoGripper_->isGrabbed()));
        }

        switchToState(State::ClimbToSavedAltitude);
    }
}

void PayloadGripperController::handleClimbToSavedAltitudeState(float dt_s)
{
    climbTime_ += std::max(dt_s, 0.0f);

    const float altitude = getVehicleAltitude();
    const float velocityZ = computeClimbVelocityToSavedAltitude();
    const bool altitudeReady = isAtSavedReturnAltitude();
    const bool timeout = climbTime_ >= paramClimbTimeout_;

    trajectorySetpoint_->update(
        Eigen::Vector3f(0.0f, 0.0f, velocityZ),
        std::nullopt,
        std::nullopt);

    RCLCPP_INFO_THROTTLE(
        node_.get_logger(),
        *(node_.get_clock()),
        500,
        "[PG][ClimbToSavedAltitude] altitude=%.3f saved=%.3f tol=%.3f cmd_z=%.3f time=%.2f/%.2f",
        altitude,
        savedReturnAltitude_,
        paramClimbAltitudeTolerance_,
        velocityZ,
        climbTime_,
        paramClimbTimeout_);

    if (altitudeReady || timeout)
    {
        holdPosition();

        if (altitudeReady)
        {
            RCLCPP_WARN(
                node_.get_logger(),
                "[PG] Returned to saved altitude. altitude=%.3f m saved=%.3f m. Mission completed.",
                altitude,
                savedReturnAltitude_);
        }
        else
        {
            RCLCPP_WARN(
                node_.get_logger(),
                "[PG] Climb timeout. altitude=%.3f m saved=%.3f m. Completing anyway.",
                altitude,
                savedReturnAltitude_);
        }

        completeMissionOnce();
    }
}

void PayloadGripperController::holdPosition()
{
    trajectorySetpoint_->update(
        Eigen::Vector3f(0.0f, 0.0f, 0.0f),
        std::nullopt,
        std::nullopt);
}

bool PayloadGripperController::checkObjectLost() const
{
    if (!objectWorld_.validPose)
    {
        return true;
    }

    const double ageSec = (node_.now() - objectWorld_.timestamp).seconds();
    return ageSec > static_cast<double>(paramObjectTimeout_);
}

Eigen::Vector2f PayloadGripperController::computeVelocityXY(const Eigen::Vector2f &errorXY, float dt_s)
{
    Eigen::Vector2f commandXY(0.0f, 0.0f);

    if (std::abs(errorXY.x()) > paramXyDeadband_)
    {
        commandXY.x() = paramXyKp_ * errorXY.x();
    }

    if (std::abs(errorXY.y()) > paramXyDeadband_)
    {
        commandXY.y() = paramXyKp_ * errorXY.y();
    }

    commandXY.x() = std::clamp(commandXY.x(), -paramXyMaxVelocity_, paramXyMaxVelocity_);
    commandXY.y() = std::clamp(commandXY.y(), -paramXyMaxVelocity_, paramXyMaxVelocity_);

    vxFiltered_ = applySlew(commandXY.x(), vxFiltered_, paramSlewAcc_, dt_s);
    vyFiltered_ = applySlew(commandXY.y(), vyFiltered_, paramSlewAcc_, dt_s);

    return Eigen::Vector2f(vxFiltered_, vyFiltered_);
}

float PayloadGripperController::computeFixedDescentVelocity() const
{
    if (isAtGrabAltitude())
    {
        return 0.0f;
    }

    return std::abs(paramDescendVelocity_);
}

float PayloadGripperController::getVehicleAltitude() const
{
    return std::abs(vehicleLocalPosition_->positionNed().z());
}

bool PayloadGripperController::isAtGrabAltitude() const
{
    const float altitude = getVehicleAltitude();
    return altitude <= (paramGrabTargetAltitude_ + paramGrabAltitudeTolerance_);
}

float PayloadGripperController::computeClimbVelocityToSavedAltitude() const
{
    if (isAtSavedReturnAltitude())
    {
        return 0.0f;
    }

    // PX4/NED: vz < 0 la bay len.
    return -std::abs(paramClimbVelocity_);
}

bool PayloadGripperController::isAtSavedReturnAltitude() const
{
    if (!hasSavedReturnAltitude_)
    {
        return true;
    }

    const float altitude = getVehicleAltitude();
    return altitude >= (savedReturnAltitude_ - paramClimbAltitudeTolerance_);
}

float PayloadGripperController::applySlew(float commandVelocity, float previousVelocity, float accelLimit, float dt_s) const
{
    const float dt = std::max(dt_s, 1e-3f);
    const float maxDeltaVelocity = std::max(accelLimit, 0.0f) * dt;
    const float deltaVelocity = std::clamp(
        commandVelocity - previousVelocity,
        -maxDeltaVelocity,
        maxDeltaVelocity);

    return previousVelocity + deltaVelocity;
}

void PayloadGripperController::switchToState(State state)
{
    if (state_ == state)
    {
        return;
    }

    RCLCPP_INFO(
        node_.get_logger(),
        "[PG] State transition: %s -> %s",
        stateName(state_).c_str(),
        stateName(state).c_str());

    if (state == State::ApproachObject)
    {
        approachSettledTime_ = 0.0f;
    }
    else if (state == State::DescendToGrabHeight)
    {
        descendTime_ = 0.0f;
        vxFiltered_ = 0.0f;
        vyFiltered_ = 0.0f;
    }
    else if (state == State::GrabReady)
    {
        grabReadySettledTime_ = 0.0f;
        vxFiltered_ = 0.0f;
        vyFiltered_ = 0.0f;
        grabCommandSent_ = false;
    }
    else if (state == State::ClimbToSavedAltitude)
    {
        climbTime_ = 0.0f;
        vxFiltered_ = 0.0f;
        vyFiltered_ = 0.0f;
    }

    state_ = state;
}

std::string PayloadGripperController::stateName(State state) const
{
    switch (state)
    {
    case State::SearchObject:
        return "SearchObject";
    case State::ApproachObject:
        return "ApproachObject";
    case State::DescendToGrabHeight:
        return "DescendToGrabHeight";
    case State::GrabReady:
        return "GrabReady";
    case State::ClimbToSavedAltitude:
        return "ClimbToSavedAltitude";
    default:
        return "Unknown";
    }
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<px4_ros2::NodeWithMode<PayloadGripperController>>(kModeName, kEnableDebugOutput));
    rclcpp::shutdown();
    return 0;
}