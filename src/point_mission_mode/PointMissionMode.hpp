#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_msgs/msg/string.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_land_detected.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>

#include "ControlTypes.hpp"
#include "ImageTargetDetector.hpp"
#include "FrameTransformer.hpp"
#include "ImageErrorKalman.hpp"
#include "ImageErrorVelocityController.hpp"
#include "FutureTargetPredictor.hpp"

class PointMissionMode : public rclcpp::Node
{
public:
    PointMissionMode();

private:
    void updateLoop();

private:
    enum class State
    {
        WaitLocalPosition,
        WarmupOffboard,
        Takeoff,
        NavigatePoints,
        ImageServoPoint,
        DropDelay,
        ReturnHomeAltitude,
        Land,
        Finished
    };

private:
    void loadParameters();
    void loadMissionPointsFromParams();
    void initializeFallbackMissionPoints();
    bool tryCaptureStartPose();
    void setupMissionAfterLocalPositionReady();
    Eigen::Vector3f bodyFrdToNedDelta(const Eigen::Vector3f &bodyFrd) const;
    Eigen::Vector3f bodyVelocityToNed(const Eigen::Vector3f &velocityBodyFrd) const;
    Eigen::Vector3f worldDeltaToBodyFrd(const Eigen::Vector3f &worldDeltaNed) const;
    void setupFrameTransformer();
    void updateFrameTransformerVehicleState();
    void computeAbsolutePointsNed();
    bool isFiniteVector(const Eigen::Vector3f &value) const;
    bool isReached(const Eigen::Vector3f &targetNed) const;
    bool isVehicleReadyForMission() const;
    const point_mission_mode::point &missionPointAt(std::size_t sequenceIndex) const;
    Eigen::Vector3f takeoffAltitudeTargetNed() const;

    void publishOffboardControlMode(bool usePosition, bool useVelocity);
    void publishGotoSetpoint(const Eigen::Vector3f &targetNed);
    void publishVelocitySetpointNed(const Eigen::Vector3f &velocityNed);
    void publishLandCommand();
    void publishVehicleCommand(uint16_t command, float param1 = 0.0f, float param2 = 0.0f);
    void publishOffboardModeCommand();
    void publishArmCommand();
    void publishDropCommand(const point_mission_mode::point &missionPoint);

    void vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg);
    void vehicleStatusCallback(const px4_msgs::msg::VehicleStatus::SharedPtr msg);
    void vehicleLandDetectedCallback(const px4_msgs::msg::VehicleLandDetected::SharedPtr msg);
    void vehicleOdometryCallback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg);
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    float estimateProjectionRangeDownM() const;

    void handleWaitLocalPositionState();
    void handleWarmupOffboardState();
    void handleTakeoffState();
    void handleNavigatePointsState();
    void handleImageServoPointState();
    void handleDropDelayState();
    void handleReturnHomeAltitudeState();
    void handleLandState();
    void handleFinishedState();
    void advanceAfterCurrentPoint();
    point_mission_mode::ImageTargetLockInput buildImageLockInput() const;
    point_mission_mode::FilteredImageError makeServoErrorForCurrentPoint() const;
    point_mission_mode::FutureTargetPredictorParams predictorParamsForPoint(const point_mission_mode::point &missionPoint) const;
    bool shouldDropDynamicTarget(const point_mission_mode::point &missionPoint, const point_mission_mode::ImageErrorControllerOutput &output, const rclcpp::Time &now);
    Eigen::Vector2f currentRollPitchRad() const;

    void switchToState(State state);
    std::string stateName(State state) const;
    void publishPointsDebug() const;
    void publishStateDebug(const std::string &extra = "") const;

private:
    rclcpp::Node &_node;
    rclcpp::TimerBase::SharedPtr _timer;

    rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr _offboardControlModePub;
    rclcpp::Publisher<px4_msgs::msg::TrajectorySetpoint>::SharedPtr _trajectorySetpointPub;

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr _pointsDebugPub;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr _stateDebugPub;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr _dropCommandPub;
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr _vehicleCommandPub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _imageDebugPub;

    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr _vehicleLocalPositionSub;
    rclcpp::Subscription<px4_msgs::msg::VehicleStatus>::SharedPtr _vehicleStatusSub;
    rclcpp::Subscription<px4_msgs::msg::VehicleLandDetected>::SharedPtr _vehicleLandDetectedSub;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _imageSub;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr _cameraInfoSub;
    rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr _vehicleOdometrySub;

    point_mission_mode::MissionGeometryParams _geometryParams{};
    point_mission_mode::GotoParams _gotoParams{};
    point_mission_mode::ImageDetectorParams _imageDetectorParams{};
    point_mission_mode::ImageServoParams _imageServoParams{};
    point_mission_mode::CameraMountParams _cameraMountParams{};
    point_mission_mode::ImageErrorKalmanParams _imageErrorKalmanParams{};
    point_mission_mode::ImageErrorControllerParams _imageErrorControllerParams{};

    point_mission_mode::HsvRange _redHsvRange{};
    point_mission_mode::HsvRange _yellowHsvRange{};
    point_mission_mode::HsvRange _blueHsvRange{};
    std::vector<std::string> _missionPointNames{};
    std::vector<std::string> _missionSequenceNames{};

    frame_transform::FrameTransformer _frameTransformer{};
    point_mission_mode::ImageTargetDetector _imageTargetDetector{};
    point_mission_mode::ImageErrorKalman _imageErrorKalman{};
    point_mission_mode::ImageErrorVelocityController _imageErrorVelocityController{};
    point_mission_mode::FutureTargetPredictor _futureTargetPredictor{};
    point_mission_mode::ImageTargetDetection _latestDetection{};
    point_mission_mode::FutureTargetPrediction _latestFuturePrediction{};
    point_mission_mode::FilteredImageError _latestFilteredImageError{};

    std::vector<point_mission_mode::point> _points;
    std::vector<std::size_t> _missionSequence;

    State _state{State::WaitLocalPosition};

    bool _active{true};
    bool _startPoseValid{false};
    bool _landDetected{false};
    bool _landCommandSent{false};
    bool _paramDebugEnable{true};
    bool _latestDetectionValid{false};
    bool _cameraInfoValid{false};
    bool _vehicleOdomValid{false};
    bool _localPositionValid{false};
    bool _vehicleStatusValid{false};

    std::size_t _currentSequenceIndex{0U};

    Eigen::Vector3f _startPositionNed{0.0f, 0.0f, 0.0f};
    float _startHeadingRad{0.0f};
    Eigen::Vector3f _localPositionNed{0.0f, 0.0f, 0.0f};
    float _localHeadingRad{0.0f};
    Eigen::Vector3f _vehiclePositionNed{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f _vehicleVelocityNed{0.0f, 0.0f, 0.0f};
    Eigen::Quaternionf _vehicleQned{1.0f, 0.0f, 0.0f, 0.0f};
    point_mission_mode::CameraIntrinsics _cameraIntrinsics{};

    std::string _imageTopic{"/camera_down/image_raw"};
    std::string _cameraInfoTopic{"/camera_down/camera_info"};
    std::string _imageDebugTopic{"/point_mission/image_debug"};
    std::string _dropCommandTopic{"/point_mission/drop_command"};
    std::string _vehicleOdometryTopic{"/fmu/out/vehicle_odometry"};

    float _paramLandCommandIntervalSec{1.0f};
    float _paramImageTimeoutSec{0.5f};
    float _paramTimerRateHz{50.0f};
    float _paramOffboardCommandIntervalSec{1.0f};
    int _paramOffboardWarmupSetpointCount{20};
    bool _paramAutoArm{true};
    bool _paramRequireOffboardAndArmed{false};
    point_mission_mode::FutureTargetPredictorParams _futureTargetPredictorParams{};

    rclcpp::Time _lastLandCommandTime{0, 0, RCL_ROS_TIME};
    rclcpp::Time _lastOffboardCommandTime{0, 0, RCL_ROS_TIME};
    rclcpp::Time _latestImageTime{0, 0, RCL_ROS_TIME};
    std::optional<rclcpp::Time> _stableStartTime;
    std::optional<rclcpp::Time> _dropDelayStartTime;
    std::optional<rclcpp::Time> _dynamicObserveStartTime;
    int _offboardWarmupCounter{0};
    uint8_t _navState{0U};
    uint8_t _armingState{0U};
};
