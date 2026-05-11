#include "PointMissionMode.hpp"

#include <algorithm>
#include <chrono>
#include <array>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>

namespace
{
const std::string kModeName = "POINT_OFFBOARD";
constexpr bool kEnableDebugOutput = true;

point_mission_mode::HsvRange makeRange(
    const std::array<int, 3> &minValue,
    const std::array<int, 3> &maxValue)
{
    point_mission_mode::HsvRange range;
    range.min = minValue;
    range.max = maxValue;
    return range;
}

std::array<int, 3> vectorToArray3(const std::vector<int64_t> &value, const std::array<int, 3> &fallback)
{
    if (value.size() != 3U)
    {
        return fallback;
    }

    return {
        static_cast<int>(value[0]),
        static_cast<int>(value[1]),
        static_cast<int>(value[2])};
}

std::string hsvRangeToJson(const point_mission_mode::HsvRange &range)
{
    std::ostringstream ss;
    ss << "{\"min\":[" << range.min[0] << "," << range.min[1] << "," << range.min[2]
       << "],\"max\":[" << range.max[0] << "," << range.max[1] << "," << range.max[2] << "]}";
    return ss.str();
}
} // namespace

PointMissionMode::PointMissionMode()
    : rclcpp::Node("point_mission_mode"),
      _node(*this)
{
    loadParameters();
    setupFrameTransformer();
    initializePointsBodyFrame();

    _offboardControlModePub = _node.create_publisher<px4_msgs::msg::OffboardControlMode>(
        "/fmu/in/offboard_control_mode",
        rclcpp::QoS(10).best_effort());

    _trajectorySetpointPub = _node.create_publisher<px4_msgs::msg::TrajectorySetpoint>(
        "/fmu/in/trajectory_setpoint",
        rclcpp::QoS(10).best_effort());

    _pointsDebugPub = _node.create_publisher<std_msgs::msg::String>(
        "/point_mission/points_debug",
        rclcpp::QoS(1).transient_local());

    _stateDebugPub = _node.create_publisher<std_msgs::msg::String>(
        "/point_mission/state_debug",
        rclcpp::QoS(10).best_effort());

    _dropCommandPub = _node.create_publisher<std_msgs::msg::String>(
        _dropCommandTopic,
        rclcpp::QoS(10).reliable());

    _vehicleCommandPub = _node.create_publisher<px4_msgs::msg::VehicleCommand>(
        "/fmu/in/vehicle_command",
        rclcpp::QoS(10).best_effort());

    _imageDebugPub = _node.create_publisher<sensor_msgs::msg::Image>(
        _imageDebugTopic,
        rclcpp::QoS(1).best_effort());

    _vehicleLocalPositionSub = _node.create_subscription<px4_msgs::msg::VehicleLocalPosition>(
        "/fmu/out/vehicle_local_position",
        rclcpp::QoS(10).best_effort(),
        std::bind(&PointMissionMode::vehicleLocalPositionCallback, this, std::placeholders::_1));

    _vehicleStatusSub = _node.create_subscription<px4_msgs::msg::VehicleStatus>(
        "/fmu/out/vehicle_status",
        rclcpp::QoS(10).best_effort(),
        std::bind(&PointMissionMode::vehicleStatusCallback, this, std::placeholders::_1));

    _vehicleLandDetectedSub = _node.create_subscription<px4_msgs::msg::VehicleLandDetected>(
        "/fmu/out/vehicle_land_detected",
        rclcpp::QoS(10).best_effort(),
        std::bind(&PointMissionMode::vehicleLandDetectedCallback, this, std::placeholders::_1));

    _imageSub = _node.create_subscription<sensor_msgs::msg::Image>(
        _imageTopic,
        rclcpp::QoS(1).best_effort(),
        std::bind(&PointMissionMode::imageCallback, this, std::placeholders::_1));

    _cameraInfoSub = _node.create_subscription<sensor_msgs::msg::CameraInfo>(
        _cameraInfoTopic,
        rclcpp::QoS(1).best_effort(),
        std::bind(&PointMissionMode::cameraInfoCallback, this, std::placeholders::_1));

    _vehicleOdometrySub = _node.create_subscription<px4_msgs::msg::VehicleOdometry>(
        _vehicleOdometryTopic,
        rclcpp::QoS(1).best_effort(),
        std::bind(&PointMissionMode::vehicleOdometryCallback, this, std::placeholders::_1));

    const auto period = std::chrono::duration<double>(1.0 / static_cast<double>(_paramTimerRateHz));
    _timer = _node.create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        std::bind(&PointMissionMode::updateLoop, this));

    RCLCPP_INFO(
        _node.get_logger(),
        "[PointMission] auto OFFBOARD node started: rate=%.1fHz, auto_arm=%d, warmup=%d",
        static_cast<double>(_paramTimerRateHz),
        static_cast<int>(_paramAutoArm),
        _paramOffboardWarmupSetpointCount);
}

void PointMissionMode::loadParameters()
{
    _node.declare_parameter<float>("takeoff_altitude_m", 3.0f);
    _node.declare_parameter<float>("x_distance_m", 8.0f);
    _node.declare_parameter<float>("f_center_offset_x_m", 2.5f);
    _node.declare_parameter<float>("f_lateral_offset_y_m", 2.5f);
    _node.declare_parameter<float>("f5_offset_x_m", 2.5f);
    _node.declare_parameter<float>("lateral_sign", -1.0f);

    _node.declare_parameter<float>("max_horizontal_speed", 1.2f);
    _node.declare_parameter<float>("max_vertical_speed", 0.6f);
    _node.declare_parameter<float>("acceptance_xy", 0.20f);
    _node.declare_parameter<float>("acceptance_z", 0.15f);
    _node.declare_parameter<float>("land_command_interval_sec", 1.0f);
    _node.declare_parameter<float>("offboard.timer_rate_hz", _paramTimerRateHz);
    _node.declare_parameter<int>("offboard.warmup_setpoint_count", _paramOffboardWarmupSetpointCount);
    _node.declare_parameter<float>("offboard.command_interval_sec", _paramOffboardCommandIntervalSec);
    _node.declare_parameter<bool>("offboard.auto_arm", _paramAutoArm);
    _node.declare_parameter<bool>("offboard.require_offboard_and_armed", _paramRequireOffboardAndArmed);

    _node.declare_parameter<std::string>("topics.image", _imageTopic);
    _node.declare_parameter<std::string>("topics.camera_info", _cameraInfoTopic);
    _node.declare_parameter<std::string>("topics.vehicle_odometry", _vehicleOdometryTopic);
    _node.declare_parameter<std::string>("topics.image_debug", _imageDebugTopic);
    _node.declare_parameter<std::string>("topics.drop_command", _dropCommandTopic);

    _node.declare_parameter<float>("camera_mount.cam_offset_x", _cameraMountParams.camOffsetX);
    _node.declare_parameter<float>("camera_mount.cam_offset_y", _cameraMountParams.camOffsetY);
    _node.declare_parameter<float>("camera_mount.cam_offset_z", _cameraMountParams.camOffsetZ);

    _node.declare_parameter<std::vector<int64_t>>("hsv.red.min", std::vector<int64_t>{170, 80, 80});
    _node.declare_parameter<std::vector<int64_t>>("hsv.red.max", std::vector<int64_t>{10, 255, 255});
    _node.declare_parameter<std::vector<int64_t>>("hsv.yellow.min", std::vector<int64_t>{20, 80, 80});
    _node.declare_parameter<std::vector<int64_t>>("hsv.yellow.max", std::vector<int64_t>{40, 255, 255});
    _node.declare_parameter<std::vector<int64_t>>("hsv.blue.min", std::vector<int64_t>{95, 80, 60});
    _node.declare_parameter<std::vector<int64_t>>("hsv.blue.max", std::vector<int64_t>{130, 255, 255});

    _node.declare_parameter<int>("image_detector.min_area_px", 250);
    _node.declare_parameter<int>("image_detector.max_area_px", 250000);
    _node.declare_parameter<float>("image_detector.min_circularity", 0.55f);
    _node.declare_parameter<float>("image_detector.min_fill_ratio", 0.45f);
    _node.declare_parameter<int>("image_detector.morph_kernel_size", 5);

    _node.declare_parameter<float>("image_servo.kp_x", _imageErrorControllerParams.kpX);
    _node.declare_parameter<float>("image_servo.kp_y", _imageErrorControllerParams.kpY);
    _node.declare_parameter<float>("image_servo.max_xy_velocity", _imageErrorControllerParams.maxXYVelocity);
    _node.declare_parameter<float>("image_servo.center_tolerance_m", _imageErrorControllerParams.centerToleranceM);
    _node.declare_parameter<float>("image_servo.stable_hold_sec", _imageServoParams.stableHoldSec);
    _node.declare_parameter<float>("image_servo.drop_delay_sec", _imageServoParams.dropDelaySec);
    _node.declare_parameter<float>("image_servo.image_timeout_sec", 0.5f);
    _node.declare_parameter<bool>("image_servo.invert_x", _imageErrorControllerParams.invertX);
    _node.declare_parameter<bool>("image_servo.invert_y", _imageErrorControllerParams.invertY);
    _node.declare_parameter<bool>("image_servo.use_current_altitude_for_projection", _imageServoParams.useCurrentAltitudeForProjection);
    _node.declare_parameter<float>("image_servo.fixed_projection_range_m", _imageServoParams.fixedProjectionRangeM);
    _node.declare_parameter<float>("image_servo.min_projection_range_m", _imageServoParams.minProjectionRangeM);
    _node.declare_parameter<float>("image_servo.max_projection_range_m", _imageServoParams.maxProjectionRangeM);

    _node.declare_parameter<float>("image_kalman.x.q_acc", _imageErrorKalmanParams.x.qAcc);
    _node.declare_parameter<float>("image_kalman.x.r_pos", _imageErrorKalmanParams.x.rPos);
    _node.declare_parameter<float>("image_kalman.y.q_acc", _imageErrorKalmanParams.y.qAcc);
    _node.declare_parameter<float>("image_kalman.y.r_pos", _imageErrorKalmanParams.y.rPos);
    _node.declare_parameter<float>("image_kalman.initial_position_var", _imageErrorKalmanParams.x.initialPositionVariance);
    _node.declare_parameter<float>("image_kalman.initial_velocity_var", _imageErrorKalmanParams.x.initialVelocityVariance);
    _node.declare_parameter<float>("image_kalman.max_predict_dt", _imageErrorKalmanParams.x.maxPredictDt);

    _node.declare_parameter<bool>("debug.enable", true);

    _node.get_parameter("takeoff_altitude_m", _geometryParams.takeoffAltitudeM);
    _node.get_parameter("x_distance_m", _geometryParams.xDistanceM);
    _node.get_parameter("f_center_offset_x_m", _geometryParams.centerOffsetXM);
    _node.get_parameter("f_lateral_offset_y_m", _geometryParams.lateralOffsetYM);
    _node.get_parameter("f5_offset_x_m", _geometryParams.f5OffsetXM);
    _node.get_parameter("lateral_sign", _geometryParams.lateralSign);

    _node.get_parameter("max_horizontal_speed", _gotoParams.maxHorizontalSpeed);
    _node.get_parameter("max_vertical_speed", _gotoParams.maxVerticalSpeed);
    _node.get_parameter("acceptance_xy", _gotoParams.acceptanceXY);
    _node.get_parameter("acceptance_z", _gotoParams.acceptanceZ);
    _node.get_parameter("land_command_interval_sec", _paramLandCommandIntervalSec);
    _node.get_parameter("offboard.timer_rate_hz", _paramTimerRateHz);
    _node.get_parameter("offboard.warmup_setpoint_count", _paramOffboardWarmupSetpointCount);
    _node.get_parameter("offboard.command_interval_sec", _paramOffboardCommandIntervalSec);
    _node.get_parameter("offboard.auto_arm", _paramAutoArm);
    _node.get_parameter("offboard.require_offboard_and_armed", _paramRequireOffboardAndArmed);

    _node.get_parameter("topics.image", _imageTopic);
    _node.get_parameter("topics.camera_info", _cameraInfoTopic);
    _node.get_parameter("topics.vehicle_odometry", _vehicleOdometryTopic);
    _node.get_parameter("topics.image_debug", _imageDebugTopic);
    _node.get_parameter("topics.drop_command", _dropCommandTopic);

    _node.get_parameter("camera_mount.cam_offset_x", _cameraMountParams.camOffsetX);
    _node.get_parameter("camera_mount.cam_offset_y", _cameraMountParams.camOffsetY);
    _node.get_parameter("camera_mount.cam_offset_z", _cameraMountParams.camOffsetZ);

    std::vector<int64_t> hsvMin;
    std::vector<int64_t> hsvMax;

    _node.get_parameter("hsv.red.min", hsvMin);
    _node.get_parameter("hsv.red.max", hsvMax);
    _redHsvRange = makeRange(
        vectorToArray3(hsvMin, {170, 80, 80}),
        vectorToArray3(hsvMax, {10, 255, 255}));

    _node.get_parameter("hsv.yellow.min", hsvMin);
    _node.get_parameter("hsv.yellow.max", hsvMax);
    _yellowHsvRange = makeRange(
        vectorToArray3(hsvMin, {20, 80, 80}),
        vectorToArray3(hsvMax, {40, 255, 255}));

    _node.get_parameter("hsv.blue.min", hsvMin);
    _node.get_parameter("hsv.blue.max", hsvMax);
    _blueHsvRange = makeRange(
        vectorToArray3(hsvMin, {95, 80, 60}),
        vectorToArray3(hsvMax, {130, 255, 255}));

    _node.get_parameter("image_detector.min_area_px", _imageDetectorParams.minAreaPx);
    _node.get_parameter("image_detector.max_area_px", _imageDetectorParams.maxAreaPx);
    _node.get_parameter("image_detector.min_circularity", _imageDetectorParams.minCircularity);
    _node.get_parameter("image_detector.min_fill_ratio", _imageDetectorParams.minFillRatio);
    _node.get_parameter("image_detector.morph_kernel_size", _imageDetectorParams.morphKernelSize);

    _node.get_parameter("image_servo.kp_x", _imageErrorControllerParams.kpX);
    _node.get_parameter("image_servo.kp_y", _imageErrorControllerParams.kpY);
    _node.get_parameter("image_servo.max_xy_velocity", _imageErrorControllerParams.maxXYVelocity);
    _node.get_parameter("image_servo.center_tolerance_m", _imageErrorControllerParams.centerToleranceM);
    _node.get_parameter("image_servo.stable_hold_sec", _imageServoParams.stableHoldSec);
    _node.get_parameter("image_servo.drop_delay_sec", _imageServoParams.dropDelaySec);
    _node.get_parameter("image_servo.image_timeout_sec", _paramImageTimeoutSec);
    _node.get_parameter("image_servo.invert_x", _imageErrorControllerParams.invertX);
    _node.get_parameter("image_servo.invert_y", _imageErrorControllerParams.invertY);
    _node.get_parameter("image_servo.use_current_altitude_for_projection", _imageServoParams.useCurrentAltitudeForProjection);
    _node.get_parameter("image_servo.fixed_projection_range_m", _imageServoParams.fixedProjectionRangeM);
    _node.get_parameter("image_servo.min_projection_range_m", _imageServoParams.minProjectionRangeM);
    _node.get_parameter("image_servo.max_projection_range_m", _imageServoParams.maxProjectionRangeM);

    _node.get_parameter("image_kalman.x.q_acc", _imageErrorKalmanParams.x.qAcc);
    _node.get_parameter("image_kalman.x.r_pos", _imageErrorKalmanParams.x.rPos);
    _node.get_parameter("image_kalman.y.q_acc", _imageErrorKalmanParams.y.qAcc);
    _node.get_parameter("image_kalman.y.r_pos", _imageErrorKalmanParams.y.rPos);
    _node.get_parameter("image_kalman.initial_position_var", _imageErrorKalmanParams.x.initialPositionVariance);
    _node.get_parameter("image_kalman.initial_velocity_var", _imageErrorKalmanParams.x.initialVelocityVariance);
    _node.get_parameter("image_kalman.max_predict_dt", _imageErrorKalmanParams.x.maxPredictDt);
    _imageErrorKalmanParams.y.initialPositionVariance = _imageErrorKalmanParams.x.initialPositionVariance;
    _imageErrorKalmanParams.y.initialVelocityVariance = _imageErrorKalmanParams.x.initialVelocityVariance;
    _imageErrorKalmanParams.y.maxPredictDt = _imageErrorKalmanParams.x.maxPredictDt;

    _imageErrorControllerParams.stableHoldSec = _imageServoParams.stableHoldSec;
    _imageErrorControllerParams.dropDelaySec = _imageServoParams.dropDelaySec;

    _node.get_parameter("debug.enable", _paramDebugEnable);

    try
    {
        if (!std::isfinite(_geometryParams.takeoffAltitudeM) || _geometryParams.takeoffAltitudeM <= 0.0f)
        {
            throw std::runtime_error("takeoff_altitude_m phai lon hon 0");
        }

        if (!std::isfinite(_geometryParams.xDistanceM) || _geometryParams.xDistanceM <= 0.0f)
        {
            throw std::runtime_error("x_distance_m phai lon hon 0");
        }

        if (!std::isfinite(_geometryParams.centerOffsetXM) || _geometryParams.centerOffsetXM < 0.0f)
        {
            throw std::runtime_error("f_center_offset_x_m khong hop le");
        }

        if (!std::isfinite(_geometryParams.lateralOffsetYM) || _geometryParams.lateralOffsetYM < 0.0f)
        {
            throw std::runtime_error("f_lateral_offset_y_m khong hop le");
        }

        if (!std::isfinite(_geometryParams.f5OffsetXM) || _geometryParams.f5OffsetXM < 0.0f)
        {
            throw std::runtime_error("f5_offset_x_m khong hop le");
        }

        if (!std::isfinite(_geometryParams.lateralSign) || std::abs(_geometryParams.lateralSign) < 0.5f)
        {
            throw std::runtime_error("lateral_sign phai la -1 hoac +1");
        }

        _geometryParams.lateralSign = (_geometryParams.lateralSign < 0.0f) ? -1.0f : 1.0f;

        if (!std::isfinite(_paramLandCommandIntervalSec) || _paramLandCommandIntervalSec <= 0.0f)
        {
            throw std::runtime_error("land_command_interval_sec phai lon hon 0");
        }

        if (!std::isfinite(_paramTimerRateHz) || _paramTimerRateHz < 10.0f)
        {
            throw std::runtime_error("offboard.timer_rate_hz phai >= 10Hz");
        }

        if (_paramOffboardWarmupSetpointCount < 2)
        {
            throw std::runtime_error("offboard.warmup_setpoint_count phai >= 2");
        }

        if (!std::isfinite(_paramOffboardCommandIntervalSec) || _paramOffboardCommandIntervalSec <= 0.0f)
        {
            throw std::runtime_error("offboard.command_interval_sec phai lon hon 0");
        }

        if (!std::isfinite(_imageServoParams.stableHoldSec) || _imageServoParams.stableHoldSec < 0.0f)
        {
            throw std::runtime_error("image_servo.stable_hold_sec khong hop le");
        }

        if (!std::isfinite(_imageServoParams.dropDelaySec) || _imageServoParams.dropDelaySec < 0.0f)
        {
            throw std::runtime_error("image_servo.drop_delay_sec khong hop le");
        }

        if (!std::isfinite(_paramImageTimeoutSec) || _paramImageTimeoutSec <= 0.0f)
        {
            throw std::runtime_error("image_servo.image_timeout_sec phai lon hon 0");
        }

        if (!std::isfinite(_imageServoParams.fixedProjectionRangeM) || _imageServoParams.fixedProjectionRangeM <= 0.0f)
        {
            throw std::runtime_error("image_servo.fixed_projection_range_m phai lon hon 0");
        }

        if (!std::isfinite(_imageServoParams.minProjectionRangeM) || _imageServoParams.minProjectionRangeM <= 0.0f)
        {
            throw std::runtime_error("image_servo.min_projection_range_m phai lon hon 0");
        }

        if (!std::isfinite(_imageServoParams.maxProjectionRangeM) ||
            _imageServoParams.maxProjectionRangeM < _imageServoParams.minProjectionRangeM)
        {
            throw std::runtime_error("image_servo.max_projection_range_m khong hop le");
        }

        _imageTargetDetector.configure(_imageDetectorParams);
        _imageErrorKalman.configure(_imageErrorKalmanParams);
        _imageErrorVelocityController.configure(_imageErrorControllerParams);
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR(_node.get_logger(), "[PointMission] Loi loadParameters: %s", e.what());
        throw;
    }
    catch (...)
    {
        RCLCPP_ERROR(_node.get_logger(), "[PointMission] Loi loadParameters khong xac dinh");
        throw;
    }
}

void PointMissionMode::initializePointsBodyFrame()
{
    _points.clear();
    _missionSequence.clear();
    _currentSequenceIndex = 0U;

    const float missionZ = -_geometryParams.takeoffAltitudeM;
    const float side = _geometryParams.lateralSign;

    const Eigen::Vector3f f1BodyFrd(_geometryParams.xDistanceM, 0.0f, missionZ);
    const Eigen::Vector3f f2BodyFrd(_geometryParams.xDistanceM + _geometryParams.centerOffsetXM, 0.0f, missionZ);
    const Eigen::Vector3f f3BodyFrd(_geometryParams.xDistanceM + _geometryParams.centerOffsetXM, side * _geometryParams.lateralOffsetYM, missionZ);
    const Eigen::Vector3f f4BodyFrd(_geometryParams.xDistanceM + _geometryParams.centerOffsetXM, -side * _geometryParams.lateralOffsetYM, missionZ);
    const Eigen::Vector3f f5BodyFrd(_geometryParams.xDistanceM + _geometryParams.centerOffsetXM + _geometryParams.f5OffsetXM, 0.0f, missionZ);

    _points.push_back(point_mission_mode::point{"F1", f1BodyFrd, Eigen::Vector3f::Zero(), _redHsvRange, true, true, 1});
    _points.push_back(point_mission_mode::point{"F2", f2BodyFrd, Eigen::Vector3f::Zero(), _redHsvRange, true, false, 0});
    _points.push_back(point_mission_mode::point{"F3", f3BodyFrd, Eigen::Vector3f::Zero(), _yellowHsvRange, true, false, 0});
    _points.push_back(point_mission_mode::point{"F4", f4BodyFrd, Eigen::Vector3f::Zero(), _yellowHsvRange, true, false, 0});
    _points.push_back(point_mission_mode::point{"F5", f5BodyFrd, Eigen::Vector3f::Zero(), _blueHsvRange, true, false, 0});

    // Thu tu bay theo yeu cau: F1 -> F3 -> F2 -> F4 -> F5.
    _missionSequence = {0U, 2U, 1U, 3U, 4U};
}

void PointMissionMode::setupMissionAfterLocalPositionReady()
{
    _startPoseValid = false;
    _currentSequenceIndex = 0U;
    _landDetected = false;
    _landCommandSent = false;
    _latestDetectionValid = false;
    _latestFilteredImageError = point_mission_mode::FilteredImageError{};
    _stableStartTime.reset();
    _dropDelayStartTime.reset();
    _lastLandCommandTime = rclcpp::Time(0, 0, RCL_ROS_TIME);
    _lastOffboardCommandTime = rclcpp::Time(0, 0, RCL_ROS_TIME);
    _offboardWarmupCounter = 0;

    initializePointsBodyFrame();
    _imageErrorKalman.reset();
    _imageErrorVelocityController.reset();

    if (!tryCaptureStartPose())
    {
        switchToState(State::WaitLocalPosition);
        return;
    }

    computeAbsolutePointsNed();
    publishPointsDebug();
    switchToState(State::WarmupOffboard);

    RCLCPP_INFO(
        _node.get_logger(),
        "[PointMission] mission auto-start: takeoff=%.2fm, x=%.2fm, points=%zu, sequence=%zu, image_topic=%s, camera_info=%s, odom=%s",
        static_cast<double>(_geometryParams.takeoffAltitudeM),
        static_cast<double>(_geometryParams.xDistanceM),
        _points.size(),
        _missionSequence.size(),
        _imageTopic.c_str(),
        _cameraInfoTopic.c_str(),
        _vehicleOdometryTopic.c_str());
}

void PointMissionMode::updateLoop()
{
    if (!_active)
    {
        return;
    }

    try
    {
        switch (_state)
        {
        case State::WaitLocalPosition:
            handleWaitLocalPositionState();
            break;
        case State::WarmupOffboard:
            handleWarmupOffboardState();
            break;
        case State::Takeoff:
            handleTakeoffState();
            break;
        case State::NavigatePoints:
            handleNavigatePointsState();
            break;
        case State::ImageServoPoint:
            handleImageServoPointState();
            break;
        case State::DropDelay:
            handleDropDelayState();
            break;
        case State::ReturnHomeAltitude:
            handleReturnHomeAltitudeState();
            break;
        case State::Land:
            handleLandState();
            break;
        case State::Finished:
            handleFinishedState();
            break;
        }
    }
    catch (const std::exception &e)
    {
        publishVelocitySetpointNed(Eigen::Vector3f::Zero());
        RCLCPP_ERROR_THROTTLE(
            _node.get_logger(),
            *(_node.get_clock()),
            2000,
            "[PointMission] Loi updateLoop: %s",
            e.what());
    }
    catch (...)
    {
        publishVelocitySetpointNed(Eigen::Vector3f::Zero());
        RCLCPP_ERROR_THROTTLE(
            _node.get_logger(),
            *(_node.get_clock()),
            2000,
            "[PointMission] Loi updateLoop khong xac dinh");
    }
}

bool PointMissionMode::tryCaptureStartPose()
{
    if (!_localPositionValid)
    {
        _startPoseValid = false;
        return false;
    }

    const Eigen::Vector3f currentPosition = _localPositionNed;
    const float currentHeading = _localHeadingRad;

    if (!isFiniteVector(currentPosition) || !std::isfinite(currentHeading))
    {
        _startPoseValid = false;
        return false;
    }

    _startPositionNed = currentPosition;
    _startHeadingRad = currentHeading;
    _startPoseValid = true;

    RCLCPP_INFO(
        _node.get_logger(),
        "[PointMission] capture start NED=(%.2f, %.2f, %.2f), heading=%.3f rad",
        static_cast<double>(_startPositionNed.x()),
        static_cast<double>(_startPositionNed.y()),
        static_cast<double>(_startPositionNed.z()),
        static_cast<double>(_startHeadingRad));

    return true;
}

Eigen::Vector3f PointMissionMode::bodyFrdToNedDelta(const Eigen::Vector3f &bodyFrd) const
{
    if (!std::isfinite(_startHeadingRad))
    {
        throw std::runtime_error("PointMissionMode::bodyFrdToNedDelta heading khong hop le");
    }

    if (!isFiniteVector(bodyFrd))
    {
        throw std::runtime_error("PointMissionMode::bodyFrdToNedDelta bodyFrd khong hop le");
    }

    const float c = std::cos(_startHeadingRad);
    const float s = std::sin(_startHeadingRad);

    Eigen::Vector3f nedDelta;
    nedDelta.x() = c * bodyFrd.x() - s * bodyFrd.y();
    nedDelta.y() = s * bodyFrd.x() + c * bodyFrd.y();
    nedDelta.z() = bodyFrd.z();

    return nedDelta;
}

Eigen::Vector3f PointMissionMode::bodyVelocityToNed(const Eigen::Vector3f &velocityBodyFrd) const
{
    if (!_localPositionValid || !isFiniteVector(velocityBodyFrd))
    {
        return Eigen::Vector3f::Zero();
    }

    // Velocity setpoint gui cho PX4 la NED.
    // Chi dung yaw/heading de doi body-FRD horizontal -> NED, tranh roll/pitch tao lenh z ngoai y muon.
    const float currentHeading = _localHeadingRad;
    if (!std::isfinite(currentHeading))
    {
        return Eigen::Vector3f::Zero();
    }

    const float c = std::cos(currentHeading);
    const float s = std::sin(currentHeading);

    Eigen::Vector3f velocityNed;
    velocityNed.x() = c * velocityBodyFrd.x() - s * velocityBodyFrd.y();
    velocityNed.y() = s * velocityBodyFrd.x() + c * velocityBodyFrd.y();
    velocityNed.z() = velocityBodyFrd.z();

    return velocityNed;
}

Eigen::Vector3f PointMissionMode::worldDeltaToBodyFrd(const Eigen::Vector3f &worldDeltaNed) const
{
    if (!_vehicleOdomValid || !isFiniteVector(worldDeltaNed))
    {
        return Eigen::Vector3f::Zero();
    }

    return _vehicleQned.inverse() * worldDeltaNed;
}

void PointMissionMode::setupFrameTransformer()
{
    const Eigen::Vector3d cameraOffsetBody(
        static_cast<double>(_cameraMountParams.camOffsetX),
        static_cast<double>(_cameraMountParams.camOffsetY),
        static_cast<double>(_cameraMountParams.camOffsetZ));

    _frameTransformer.setConfig(
        frame_transform::FrameTransformer::makeBellyFixedCameraConfig(cameraOffsetBody));
    _frameTransformer.setBodyFromMountQuaternion(Eigen::Quaterniond::Identity());
}

void PointMissionMode::updateFrameTransformerVehicleState()
{
    frame_transform::VehicleState vehicleState;
    vehicleState.worldFromBody = Eigen::Quaterniond(
        static_cast<double>(_vehicleQned.w()),
        static_cast<double>(_vehicleQned.x()),
        static_cast<double>(_vehicleQned.y()),
        static_cast<double>(_vehicleQned.z()));
    vehicleState.positionWorld = Eigen::Vector3d(
        static_cast<double>(_vehiclePositionNed.x()),
        static_cast<double>(_vehiclePositionNed.y()),
        static_cast<double>(_vehiclePositionNed.z()));
    vehicleState.velocityWorld = Eigen::Vector3d(
        static_cast<double>(_vehicleVelocityNed.x()),
        static_cast<double>(_vehicleVelocityNed.y()),
        static_cast<double>(_vehicleVelocityNed.z()));
    vehicleState.valid = _vehicleOdomValid;

    _frameTransformer.setVehicleState(vehicleState);
}

void PointMissionMode::computeAbsolutePointsNed()
{
    if (!_startPoseValid)
    {
        throw std::runtime_error("PointMissionMode::computeAbsolutePointsNed startPose chua hop le");
    }

    for (auto &missionPoint : _points)
    {
        missionPoint.absoluteNed = _startPositionNed + bodyFrdToNedDelta(missionPoint.relativeBodyFrd);
    }
}

bool PointMissionMode::isFiniteVector(const Eigen::Vector3f &value) const
{
    return std::isfinite(value.x()) && std::isfinite(value.y()) && std::isfinite(value.z());
}

bool PointMissionMode::isReached(const Eigen::Vector3f &targetNed) const
{
    if (!_localPositionValid || !isFiniteVector(targetNed))
    {
        return false;
    }

    const Eigen::Vector3f error = targetNed - _localPositionNed;
    const float errorXY = error.head<2>().norm();
    const float errorZ = std::fabs(error.z());

    return errorXY <= _gotoParams.acceptanceXY && errorZ <= _gotoParams.acceptanceZ;
}

bool PointMissionMode::isVehicleReadyForMission() const
{
    if (!_paramRequireOffboardAndArmed)
    {
        return true;
    }

    if (!_vehicleStatusValid)
    {
        return false;
    }

    const bool offboardActive =
        _navState == px4_msgs::msg::VehicleStatus::NAVIGATION_STATE_OFFBOARD;
    const bool armed =
        _armingState == px4_msgs::msg::VehicleStatus::ARMING_STATE_ARMED;

    return offboardActive && armed;
}

const point_mission_mode::point &PointMissionMode::missionPointAt(std::size_t sequenceIndex) const
{
    if (sequenceIndex >= _missionSequence.size())
    {
        throw std::runtime_error("PointMissionMode::missionPointAt sequenceIndex vuot gioi han");
    }

    const std::size_t pointIndex = _missionSequence[sequenceIndex];
    if (pointIndex >= _points.size())
    {
        throw std::runtime_error("PointMissionMode::missionPointAt pointIndex vuot gioi han");
    }

    return _points[pointIndex];
}

Eigen::Vector3f PointMissionMode::takeoffAltitudeTargetNed() const
{
    if (!_startPoseValid)
    {
        throw std::runtime_error("PointMissionMode::takeoffAltitudeTargetNed startPose chua hop le");
    }

    return _startPositionNed + bodyFrdToNedDelta(
        Eigen::Vector3f(0.0f, 0.0f, -_geometryParams.takeoffAltitudeM));
}

void PointMissionMode::publishOffboardControlMode(bool usePosition, bool useVelocity)
{
    if (!_offboardControlModePub)
    {
        return;
    }

    px4_msgs::msg::OffboardControlMode msg{};
    msg.timestamp = _node.now().nanoseconds() / 1000;
    msg.position = usePosition;
    msg.velocity = useVelocity;
    msg.acceleration = false;
    msg.attitude = false;
    msg.body_rate = false;

    _offboardControlModePub->publish(msg);
}

void PointMissionMode::publishGotoSetpoint(const Eigen::Vector3f &targetNed)
{
    if (!_trajectorySetpointPub || !isFiniteVector(targetNed))
    {
        return;
    }

    publishOffboardControlMode(true, false);

    const float nan = std::numeric_limits<float>::quiet_NaN();
    px4_msgs::msg::TrajectorySetpoint msg{};
    msg.timestamp = _node.now().nanoseconds() / 1000;
    msg.position[0] = targetNed.x();
    msg.position[1] = targetNed.y();
    msg.position[2] = targetNed.z();
    msg.velocity[0] = nan;
    msg.velocity[1] = nan;
    msg.velocity[2] = nan;
    msg.acceleration[0] = nan;
    msg.acceleration[1] = nan;
    msg.acceleration[2] = nan;
    msg.jerk[0] = nan;
    msg.jerk[1] = nan;
    msg.jerk[2] = nan;
    msg.yaw = std::isfinite(_startHeadingRad) ? _startHeadingRad : nan;
    msg.yawspeed = nan;

    _trajectorySetpointPub->publish(msg);
}

void PointMissionMode::publishVelocitySetpointNed(const Eigen::Vector3f &velocityNed)
{
    if (!_trajectorySetpointPub || !isFiniteVector(velocityNed))
    {
        return;
    }

    publishOffboardControlMode(false, true);

    const float nan = std::numeric_limits<float>::quiet_NaN();
    px4_msgs::msg::TrajectorySetpoint msg{};
    msg.timestamp = _node.now().nanoseconds() / 1000;
    msg.position[0] = nan;
    msg.position[1] = nan;
    msg.position[2] = nan;
    msg.velocity[0] = velocityNed.x();
    msg.velocity[1] = velocityNed.y();
    msg.velocity[2] = velocityNed.z();
    msg.acceleration[0] = nan;
    msg.acceleration[1] = nan;
    msg.acceleration[2] = nan;
    msg.jerk[0] = nan;
    msg.jerk[1] = nan;
    msg.jerk[2] = nan;
    msg.yaw = std::isfinite(_startHeadingRad) ? _startHeadingRad : nan;
    msg.yawspeed = nan;

    _trajectorySetpointPub->publish(msg);
}

void PointMissionMode::publishVehicleCommand(uint16_t command, float param1, float param2)
{
    if (!_vehicleCommandPub)
    {
        return;
    }

    px4_msgs::msg::VehicleCommand msg{};
    msg.timestamp = _node.now().nanoseconds() / 1000;
    msg.command = command;
    msg.param1 = param1;
    msg.param2 = param2;
    msg.param3 = 0.0f;
    msg.param4 = 0.0f;
    msg.param5 = 0.0f;
    msg.param6 = 0.0f;
    msg.param7 = 0.0f;
    msg.target_system = 1;
    msg.target_component = 1;
    msg.source_system = 1;
    msg.source_component = 1;
    msg.confirmation = 0;
    msg.from_external = true;

    _vehicleCommandPub->publish(msg);
}

void PointMissionMode::publishOffboardModeCommand()
{
    publishVehicleCommand(
        px4_msgs::msg::VehicleCommand::VEHICLE_CMD_DO_SET_MODE,
        1.0f,
        6.0f);
}

void PointMissionMode::publishArmCommand()
{
    publishVehicleCommand(
        px4_msgs::msg::VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM,
        1.0f,
        0.0f);
}

void PointMissionMode::publishLandCommand()
{
    const rclcpp::Time now = _node.now();
    if (_landCommandSent &&
        ((now - _lastLandCommandTime).seconds() < static_cast<double>(_paramLandCommandIntervalSec)))
    {
        return;
    }

    publishVehicleCommand(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_NAV_LAND);

    _landCommandSent = true;
    _lastLandCommandTime = now;

    RCLCPP_WARN_THROTTLE(
        _node.get_logger(),
        *(_node.get_clock()),
        2000,
        "[PointMission] publish VEHICLE_CMD_NAV_LAND");
}

void PointMissionMode::publishDropCommand(const point_mission_mode::point &missionPoint)
{
    if (!_dropCommandPub || !missionPoint.dropCommandEnable)
    {
        return;
    }

    std_msgs::msg::String msg;
    std::ostringstream ss;
    ss << "{"
       << "\"action\":\"DROP_BALL\","
       << "\"point\":\"" << missionPoint.name << "\","
       << "\"leg_id\":" << missionPoint.dropLegId
       << "}";
    msg.data = ss.str();

    _dropCommandPub->publish(msg);

    RCLCPP_WARN(
        _node.get_logger(),
        "[PointMission] DROP notify: point=%s leg=%d",
        missionPoint.name.c_str(),
        missionPoint.dropLegId);
}

void PointMissionMode::vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg)
{
    if (!msg)
    {
        _localPositionValid = false;
        return;
    }

    if (!msg->xy_valid || !msg->z_valid)
    {
        _localPositionValid = false;
        return;
    }

    _localPositionNed = Eigen::Vector3f(msg->x, msg->y, msg->z);
    _localHeadingRad = msg->heading;

    if (!isFiniteVector(_localPositionNed) || !std::isfinite(_localHeadingRad))
    {
        _localPositionValid = false;
        return;
    }

    _localPositionValid = true;
}

void PointMissionMode::vehicleStatusCallback(const px4_msgs::msg::VehicleStatus::SharedPtr msg)
{
    if (!msg)
    {
        _vehicleStatusValid = false;
        return;
    }

    _navState = msg->nav_state;
    _armingState = msg->arming_state;
    _vehicleStatusValid = true;
}

void PointMissionMode::vehicleLandDetectedCallback(const px4_msgs::msg::VehicleLandDetected::SharedPtr msg)
{
    if (!msg)
    {
        return;
    }

    _landDetected = msg->landed;
}

void PointMissionMode::vehicleOdometryCallback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
{
    if (!msg)
    {
        return;
    }

    Eigen::Quaternionf q(
        static_cast<float>(msg->q[0]),
        static_cast<float>(msg->q[1]),
        static_cast<float>(msg->q[2]),
        static_cast<float>(msg->q[3]));

    if (!std::isfinite(q.w()) || !std::isfinite(q.x()) ||
        !std::isfinite(q.y()) || !std::isfinite(q.z()) ||
        q.norm() <= 1e-6f)
    {
        _vehicleOdomValid = false;
        return;
    }

    q.normalize();
    _vehicleQned = q;
    _vehiclePositionNed = Eigen::Vector3f(
        static_cast<float>(msg->position[0]),
        static_cast<float>(msg->position[1]),
        static_cast<float>(msg->position[2]));
    _vehicleVelocityNed = Eigen::Vector3f(
        static_cast<float>(msg->velocity[0]),
        static_cast<float>(msg->velocity[1]),
        static_cast<float>(msg->velocity[2]));

    if (!isFiniteVector(_vehiclePositionNed) || !isFiniteVector(_vehicleVelocityNed))
    {
        _vehicleOdomValid = false;
        return;
    }

    _vehicleOdomValid = true;
    updateFrameTransformerVehicleState();
}

void PointMissionMode::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
    if (!msg)
    {
        return;
    }

    const float fx = static_cast<float>(msg->k[0]);
    const float fy = static_cast<float>(msg->k[4]);
    const float cx = static_cast<float>(msg->k[2]);
    const float cy = static_cast<float>(msg->k[5]);

    if (!std::isfinite(fx) || !std::isfinite(fy) ||
        !std::isfinite(cx) || !std::isfinite(cy) ||
        fx <= 1.0f || fy <= 1.0f ||
        msg->width == 0U || msg->height == 0U)
    {
        _cameraInfoValid = false;
        _cameraIntrinsics.valid = false;
        RCLCPP_WARN_THROTTLE(
            _node.get_logger(),
            *(_node.get_clock()),
            2000,
            "[PointMission] camera_info invalid");
        return;
    }

    _cameraIntrinsics.valid = true;
    _cameraIntrinsics.width = static_cast<int>(msg->width);
    _cameraIntrinsics.height = static_cast<int>(msg->height);
    _cameraIntrinsics.fx = fx;
    _cameraIntrinsics.fy = fy;
    _cameraIntrinsics.cx = cx;
    _cameraIntrinsics.cy = cy;
    _cameraInfoValid = true;
}

float PointMissionMode::estimateProjectionRangeDownM() const
{
    float rangeDownM = _imageServoParams.fixedProjectionRangeM;

    if (_imageServoParams.useCurrentAltitudeForProjection && _startPoseValid && _localPositionValid)
    {
        const Eigen::Vector3f currentPosition = _localPositionNed;
        if (isFiniteVector(currentPosition))
        {
            // NED: z giam khi UAV bay len. Neu start tren mat dat, start.z - current.z la do cao hien tai.
            const float heightAboveStartM = _startPositionNed.z() - currentPosition.z();
            if (std::isfinite(heightAboveStartM) && heightAboveStartM > 0.0f)
            {
                rangeDownM = heightAboveStartM;
            }
        }
    }

    return std::clamp(
        rangeDownM,
        _imageServoParams.minProjectionRangeM,
        _imageServoParams.maxProjectionRangeM);
}

void PointMissionMode::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    if (!msg || !_active)
    {
        return;
    }

    if (_currentSequenceIndex >= _missionSequence.size())
    {
        _latestDetectionValid = false;
        return;
    }

    cv_bridge::CvImagePtr cvPtr;
    try
    {
        cvPtr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (const cv_bridge::Exception &exception)
    {
        _latestDetectionValid = false;
        RCLCPP_WARN_THROTTLE(
            _node.get_logger(),
            *(_node.get_clock()),
            2000,
            "[PointMission] cv_bridge error: %s",
            exception.what());
        return;
    }

    try
    {
        const point_mission_mode::point &targetPoint = missionPointAt(_currentSequenceIndex);

        if (!_cameraInfoValid || !_cameraIntrinsics.valid)
        {
            _latestDetectionValid = false;
            RCLCPP_WARN_THROTTLE(
                _node.get_logger(),
                *(_node.get_clock()),
                2000,
                "[PointMission] waiting camera_info: %s",
                _cameraInfoTopic.c_str());
            return;
        }

        if (!_vehicleOdomValid)
        {
            _latestDetectionValid = false;
            RCLCPP_WARN_THROTTLE(
                _node.get_logger(),
                *(_node.get_clock()),
                2000,
                "[PointMission] waiting vehicle_odometry: %s",
                _vehicleOdometryTopic.c_str());
            return;
        }

        const float projectionRangeDownM = estimateProjectionRangeDownM();

        cv::Mat debugImage;
        _latestDetection = _imageTargetDetector.detect(
            cvPtr->image,
            targetPoint.hsvRange,
            _cameraIntrinsics,
            projectionRangeDownM,
            _paramDebugEnable ? &debugImage : nullptr);

        _latestDetectionValid = false;
        _latestFilteredImageError = point_mission_mode::FilteredImageError{};
        _latestImageTime = _node.now();

        if (_latestDetection.valid && _latestDetection.opticalPositionValid)
        {
            updateFrameTransformerVehicleState();

            const Eigen::Vector3d targetOpticalM(
                static_cast<double>(_latestDetection.targetOpticalM.x()),
                static_cast<double>(_latestDetection.targetOpticalM.y()),
                static_cast<double>(_latestDetection.targetOpticalM.z()));

            const Eigen::Vector3d targetWorldNedD =
                _frameTransformer.opticalPositionToWorld(targetOpticalM);

            const Eigen::Vector3f targetWorldNed(
                static_cast<float>(targetWorldNedD.x()),
                static_cast<float>(targetWorldNedD.y()),
                static_cast<float>(targetWorldNedD.z()));

            const Eigen::Vector3f targetDeltaNed = targetWorldNed - _vehiclePositionNed;
            const Eigen::Vector3f targetErrorBody = worldDeltaToBodyFrd(targetDeltaNed);
            const Eigen::Vector2f rawErrorBodyXY(
                targetErrorBody.x(),
                targetErrorBody.y());

            rclcpp::Time measurementStamp = msg->header.stamp;
            if (measurementStamp.nanoseconds() == 0)
            {
                measurementStamp = _node.now();
            }

            _latestFilteredImageError = _imageErrorKalman.update(rawErrorBodyXY, measurementStamp);
            _latestDetectionValid = _latestFilteredImageError.valid;
        }

        if (_paramDebugEnable && _imageDebugPub && !debugImage.empty())
        {
            if (_latestFilteredImageError.valid)
            {
                const std::string kalmanText = "KF xy(m): " +
                    std::to_string(_latestFilteredImageError.filteredBodyXY.x()).substr(0, 6) + "," +
                    std::to_string(_latestFilteredImageError.filteredBodyXY.y()).substr(0, 6);
                cv::putText(debugImage, kalmanText, cv::Point(12, 56), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
            }

            cv_bridge::CvImage debugMsg;
            debugMsg.header = msg->header;
            debugMsg.encoding = sensor_msgs::image_encodings::BGR8;
            debugMsg.image = debugImage;
            _imageDebugPub->publish(*debugMsg.toImageMsg());
        }
    }
    catch (const std::exception &exception)
    {
        _latestDetectionValid = false;
        RCLCPP_WARN_THROTTLE(
            _node.get_logger(),
            *(_node.get_clock()),
            2000,
            "[PointMission] imageCallback failed: %s",
            exception.what());
    }
}

void PointMissionMode::handleWaitLocalPositionState()
{
    if (_localPositionValid)
    {
        setupMissionAfterLocalPositionReady();
        return;
    }

    RCLCPP_WARN_THROTTLE(
        _node.get_logger(),
        *(_node.get_clock()),
        2000,
        "[PointMission] waiting for valid /fmu/out/vehicle_local_position + heading");
}

void PointMissionMode::handleWarmupOffboardState()
{
    if (!_startPoseValid)
    {
        switchToState(State::WaitLocalPosition);
        return;
    }

    const Eigen::Vector3f takeoffTarget = takeoffAltitudeTargetNed();
    publishGotoSetpoint(takeoffTarget);
    ++_offboardWarmupCounter;

    const rclcpp::Time now = _node.now();
    const bool allowCommandByInterval =
        _lastOffboardCommandTime.nanoseconds() == 0 ||
        ((now - _lastOffboardCommandTime).seconds() >= static_cast<double>(_paramOffboardCommandIntervalSec));

    if (_offboardWarmupCounter >= _paramOffboardWarmupSetpointCount && allowCommandByInterval)
    {
        publishOffboardModeCommand();

        if (_paramAutoArm)
        {
            publishArmCommand();
        }

        _lastOffboardCommandTime = now;

        RCLCPP_WARN(
            _node.get_logger(),
            "[PointMission] request PX4 OFFBOARD + ARM, warmup_count=%d",
            _offboardWarmupCounter);
    }

    if (_offboardWarmupCounter >= _paramOffboardWarmupSetpointCount && isVehicleReadyForMission())
    {
        switchToState(State::Takeoff);
    }
}

void PointMissionMode::handleTakeoffState()
{
    if (!_startPoseValid)
    {
        switchToState(State::WaitLocalPosition);
        return;
    }

    const Eigen::Vector3f takeoffTarget = takeoffAltitudeTargetNed();
    publishGotoSetpoint(takeoffTarget);

    if (isReached(takeoffTarget))
    {
        _currentSequenceIndex = 0U;
        switchToState(State::NavigatePoints);
    }
}

void PointMissionMode::handleNavigatePointsState()
{
    if (_missionSequence.empty())
    {
        throw std::runtime_error("PointMissionMode::handleNavigatePointsState missionSequence rong");
    }

    if (_currentSequenceIndex >= _missionSequence.size())
    {
        switchToState(State::ReturnHomeAltitude);
        return;
    }

    const point_mission_mode::point &targetPoint = missionPointAt(_currentSequenceIndex);
    publishGotoSetpoint(targetPoint.absoluteNed);

    if (isReached(targetPoint.absoluteNed))
    {
        RCLCPP_INFO(
            _node.get_logger(),
            "[PointMission] reached local waypoint %s (%zu/%zu), start image servo=%d",
            targetPoint.name.c_str(),
            _currentSequenceIndex + 1U,
            _missionSequence.size(),
            static_cast<int>(targetPoint.useImageServo));

        _stableStartTime.reset();
        _dropDelayStartTime.reset();
        _latestDetectionValid = false;
        _latestFilteredImageError = point_mission_mode::FilteredImageError{};
        _imageErrorKalman.reset();
        _imageErrorVelocityController.reset();

        if (targetPoint.useImageServo)
        {
            switchToState(State::ImageServoPoint);
        }
        else
        {
            advanceAfterCurrentPoint();
        }
    }
}

void PointMissionMode::handleImageServoPointState()
{
    if (_currentSequenceIndex >= _missionSequence.size())
    {
        switchToState(State::ReturnHomeAltitude);
        return;
    }

    const point_mission_mode::point &targetPoint = missionPointAt(_currentSequenceIndex);
    const rclcpp::Time now = _node.now();
    const bool imageFresh = _latestDetectionValid &&
        ((now - _latestImageTime).seconds() <= static_cast<double>(_paramImageTimeoutSec));

    if (!imageFresh)
    {
        _stableStartTime.reset();
        publishVelocitySetpointNed(Eigen::Vector3f::Zero());
        RCLCPP_WARN_THROTTLE(
            _node.get_logger(),
            *(_node.get_clock()),
            1000,
            "[PointMission] waiting image target/camera_info for %s",
            targetPoint.name.c_str());
        return;
    }

    point_mission_mode::ImageErrorControllerInput input;
    input.filteredError = _latestFilteredImageError;

    const point_mission_mode::ImageErrorControllerOutput output = _imageErrorVelocityController.update(input);
    const Eigen::Vector3f velocityNed = bodyVelocityToNed(output.velocityBodyFrd);
    publishVelocitySetpointNed(velocityNed);

    if (!output.centered)
    {
        _stableStartTime.reset();
        publishStateDebug("image_servo_tracking");
        return;
    }

    if (!_stableStartTime.has_value())
    {
        _stableStartTime = now;
        publishStateDebug("image_centered_start_hold");
        return;
    }

    if ((now - _stableStartTime.value()).seconds() >= static_cast<double>(_imageServoParams.stableHoldSec))
    {
        publishVelocitySetpointNed(Eigen::Vector3f::Zero());

        if (targetPoint.dropCommandEnable)
        {
            _dropDelayStartTime = now;
            switchToState(State::DropDelay);
        }
        else
        {
            advanceAfterCurrentPoint();
        }
    }
}

void PointMissionMode::handleDropDelayState()
{
    publishVelocitySetpointNed(Eigen::Vector3f::Zero());

    if (_currentSequenceIndex >= _missionSequence.size())
    {
        switchToState(State::ReturnHomeAltitude);
        return;
    }

    if (!_dropDelayStartTime.has_value())
    {
        _dropDelayStartTime = _node.now();
        return;
    }

    const rclcpp::Time now = _node.now();
    if ((now - _dropDelayStartTime.value()).seconds() < static_cast<double>(_imageServoParams.dropDelaySec))
    {
        return;
    }

    const point_mission_mode::point &targetPoint = missionPointAt(_currentSequenceIndex);
    publishDropCommand(targetPoint);
    advanceAfterCurrentPoint();
}

void PointMissionMode::advanceAfterCurrentPoint()
{
    if (_currentSequenceIndex < _missionSequence.size())
    {
        const point_mission_mode::point &targetPoint = missionPointAt(_currentSequenceIndex);
        RCLCPP_INFO(
            _node.get_logger(),
            "[PointMission] completed point %s (%zu/%zu)",
            targetPoint.name.c_str(),
            _currentSequenceIndex + 1U,
            _missionSequence.size());
    }

    ++_currentSequenceIndex;
    _stableStartTime.reset();
    _dropDelayStartTime.reset();
    _latestDetectionValid = false;
    _latestFilteredImageError = point_mission_mode::FilteredImageError{};
    _imageErrorKalman.reset();
    _imageErrorVelocityController.reset();

    if (_currentSequenceIndex >= _missionSequence.size())
    {
        switchToState(State::ReturnHomeAltitude);
    }
    else
    {
        switchToState(State::NavigatePoints);
    }
}

void PointMissionMode::handleReturnHomeAltitudeState()
{
    if (!_startPoseValid)
    {
        switchToState(State::WaitLocalPosition);
        return;
    }

    const Eigen::Vector3f returnTarget = takeoffAltitudeTargetNed();
    publishGotoSetpoint(returnTarget);

    if (isReached(returnTarget))
    {
        RCLCPP_INFO(
            _node.get_logger(),
            "[PointMission] reached takeoff origin at %.2fm, start landing",
            static_cast<double>(_geometryParams.takeoffAltitudeM));
        _landCommandSent = false;
        _lastLandCommandTime = rclcpp::Time(0, 0, RCL_ROS_TIME);
        switchToState(State::Land);
    }
}

void PointMissionMode::handleLandState()
{
    if (_startPoseValid)
    {
        publishGotoSetpoint(takeoffAltitudeTargetNed());
    }

    publishLandCommand();

    if (_landDetected)
    {
        switchToState(State::Finished);
    }
}

void PointMissionMode::handleFinishedState()
{
    _active = false;
    RCLCPP_WARN_THROTTLE(
        _node.get_logger(),
        *(_node.get_clock()),
        5000,
        "[PointMission] Finished: landed detected, auto offboard mission stopped");
}

void PointMissionMode::switchToState(State state)
{
    if (_state == state)
    {
        return;
    }

    _state = state;

    if (_state != State::ImageServoPoint)
    {
        _stableStartTime.reset();
    }

    if (_state != State::DropDelay)
    {
        _dropDelayStartTime.reset();
    }

    RCLCPP_INFO(
        _node.get_logger(),
        "[PointMission] enter state: %s",
        stateName(_state).c_str());

    publishStateDebug("state_changed");
}

std::string PointMissionMode::stateName(State state) const
{
    switch (state)
    {
    case State::WaitLocalPosition:
        return "WAIT_LOCAL_POSITION";
    case State::WarmupOffboard:
        return "WARMUP_OFFBOARD";
    case State::Takeoff:
        return "TAKEOFF";
    case State::NavigatePoints:
        return "NAVIGATE_POINTS";
    case State::ImageServoPoint:
        return "IMAGE_SERVO_POINT";
    case State::DropDelay:
        return "DROP_DELAY";
    case State::ReturnHomeAltitude:
        return "RETURN_HOME_ALTITUDE";
    case State::Land:
        return "LAND";
    case State::Finished:
        return "FINISHED";
    default:
        return "UNKNOWN";
    }
}

void PointMissionMode::publishPointsDebug() const
{
    if (!_paramDebugEnable || !_pointsDebugPub)
    {
        return;
    }

    std_msgs::msg::String msg;
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "{\"points\":[";

    for (std::size_t i = 0; i < _points.size(); ++i)
    {
        const point_mission_mode::point &missionPoint = _points[i];

        if (i > 0U)
        {
            ss << ",";
        }

        ss << "{"
           << "\"name\":\"" << missionPoint.name << "\","
           << "\"relative_body_frd\":["
           << missionPoint.relativeBodyFrd.x() << ","
           << missionPoint.relativeBodyFrd.y() << ","
           << missionPoint.relativeBodyFrd.z() << "],"
           << "\"absolute_ned\":["
           << missionPoint.absoluteNed.x() << ","
           << missionPoint.absoluteNed.y() << ","
           << missionPoint.absoluteNed.z() << "],"
           << "\"hsv_range\":" << hsvRangeToJson(missionPoint.hsvRange) << ","
           << "\"use_image_servo\":" << (missionPoint.useImageServo ? "true" : "false") << ","
           << "\"drop_enable\":" << (missionPoint.dropCommandEnable ? "true" : "false") << ","
           << "\"drop_leg\":" << missionPoint.dropLegId
           << "}";
    }

    ss << "],\"mission_sequence\":[";
    for (std::size_t i = 0; i < _missionSequence.size(); ++i)
    {
        if (i > 0U)
        {
            ss << ",";
        }

        const point_mission_mode::point &missionPoint = missionPointAt(i);
        ss << "\"" << missionPoint.name << "\"";
    }

    ss << "],\"start_heading_rad\":" << _startHeadingRad << "}";
    msg.data = ss.str();
    _pointsDebugPub->publish(msg);
}

void PointMissionMode::publishStateDebug(const std::string &extra) const
{
    if (!_paramDebugEnable || !_stateDebugPub)
    {
        return;
    }

    std::string currentTargetName = "NONE";
    if (_state == State::ReturnHomeAltitude)
    {
        currentTargetName = "TAKEOFF_ORIGIN_3M";
    }
    else if (_state == State::Land)
    {
        currentTargetName = "LAND";
    }
    else if (_state == State::Finished)
    {
        currentTargetName = "FINISHED";
    }
    else if (!_missionSequence.empty() && _currentSequenceIndex < _missionSequence.size())
    {
        currentTargetName = missionPointAt(_currentSequenceIndex).name;
    }
    else if (!_missionSequence.empty())
    {
        currentTargetName = missionPointAt(_missionSequence.size() - 1U).name;
    }

    std_msgs::msg::String msg;
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3)
       << "{"
       << "\"state\":\"" << stateName(_state) << "\","
       << "\"extra\":\"" << extra << "\","
       << "\"current_sequence_index\":" << _currentSequenceIndex << ","
       << "\"current_target\":\"" << currentTargetName << "\","
       << "\"sequence_size\":" << _missionSequence.size() << ","
       << "\"start_valid\":" << (_startPoseValid ? "true" : "false") << ","
       << "\"start_heading_rad\":" << _startHeadingRad << ","
       << "\"land_detected\":" << (_landDetected ? "true" : "false") << ","
       << "\"image_valid\":" << (_latestDetectionValid ? "true" : "false") << ","
       << "\"camera_info_valid\":" << (_cameraInfoValid ? "true" : "false") << ","
       << "\"vehicle_odom_valid\":" << (_vehicleOdomValid ? "true" : "false") << ","
       << "\"local_position_valid\":" << (_localPositionValid ? "true" : "false") << ","
       << "\"vehicle_status_valid\":" << (_vehicleStatusValid ? "true" : "false") << ","
       << "\"nav_state\":" << static_cast<int>(_navState) << ","
       << "\"arming_state\":" << static_cast<int>(_armingState) << ","
       << "\"offboard_warmup_counter\":" << _offboardWarmupCounter << ","
       << "\"image_error_norm\":["
       << _latestDetection.errorNorm.x() << ","
       << _latestDetection.errorNorm.y() << "],"
       << "\"target_body_xy_m\":["
       << _latestDetection.targetBodyXYM.x() << ","
       << _latestDetection.targetBodyXYM.y() << "],"
       << "\"kalman_raw_body_xy_m\":["
       << _latestFilteredImageError.rawBodyXY.x() << ","
       << _latestFilteredImageError.rawBodyXY.y() << "],"
       << "\"kalman_filtered_body_xy_m\":["
       << _latestFilteredImageError.filteredBodyXY.x() << ","
       << _latestFilteredImageError.filteredBodyXY.y() << "],"
       << "\"kalman_velocity_body_xy_mps\":["
       << _latestFilteredImageError.velocityBodyXY.x() << ","
       << _latestFilteredImageError.velocityBodyXY.y() << "],"
       << "\"kalman_valid\":" << (_latestFilteredImageError.valid ? "true" : "false") << ","
       << "\"metric_valid\":" << (_latestDetection.metricValid ? "true" : "false") << ","
       << "\"projection_range_down_m\":" << _latestDetection.rangeDownM << ","
       << "\"image_area_px\":" << _latestDetection.areaPx << ","
       << "\"start_ned\":["
       << _startPositionNed.x() << ","
       << _startPositionNed.y() << ","
       << _startPositionNed.z() << "]"
       << "}";

    msg.data = ss.str();
    _stateDebugPub->publish(msg);
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    (void)kModeName;
    (void)kEnableDebugOutput;
    rclcpp::spin(std::make_shared<PointMissionMode>());
    rclcpp::shutdown();
    return 0;
}
