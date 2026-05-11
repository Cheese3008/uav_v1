#pragma once

#include <memory>
#include <optional>
#include <string>

#include <Eigen/Core>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>

#include <px4_msgs/msg/vehicle_attitude.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>

#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/control/setpoint_types/experimental/trajectory.hpp>

class RingPassController : public px4_ros2::ModeBase
{
public:
    explicit RingPassController(rclcpp::Node& node);

    void onActivate() override;
    void onDeactivate() override;
    void updateSetpoint(float dtSec) override;

private:
    enum class RingModeState
    {
        WaitLocalPosition = 0,
        WaitRingTarget,
        RingPass,
        Finished
    };

    struct BodyTarget
    {
        bool valid{false};
        rclcpp::Time stamp{0, 0, RCL_ROS_TIME};

        // Position tương đối của tâm vòng trong body/drone frame.
        // Quy ước giống RingDetector hiện tại:
        // x: phía trước drone, y: bên phải drone, z: xuống dưới.
        Eigen::Vector3d positionXyz{Eigen::Vector3d::Zero()};

        // Velocity tương đối do Kalman trong RingDetector ước lượng.
        // vx, vy, vz cùng frame với positionXyz.
        Eigen::Vector3d velocityXyz{Eigen::Vector3d::Zero()};
    };

private:
    void loadParameters();
    void reset();
    void publishDetectorReset();

    void vehicleAttitudeCallback(const px4_msgs::msg::VehicleAttitude::SharedPtr msg);
    void vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg);
    void targetPositionCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void targetVelocityCallback(const geometry_msgs::msg::TwistStamped::SharedPtr msg);
    void targetValidCallback(const std_msgs::msg::Bool::SharedPtr msg);

    bool attitudeReady() const;
    bool targetTimedOut() const;
    bool isReady() const;
    bool hasValidTarget() const;

    double vehicleYawRad() const;
    double applySlew(double command, double previous, double accelerationLimit, double dtSec) const;
    double computeBodyVelocityX(double yzErrorRadius) const;
    Eigen::Vector2d bodyXyToNedXy(const Eigen::Vector2d& velocityBodyXy, double yawRad) const;

    Eigen::Vector3f update(double dtSec);
    Eigen::Vector3f computeVelocityCommand(double dtSec);
    Eigen::Vector3f computeHoldCommand(double dtSec);

    void enterState(RingModeState newState);
    void publishHoldPosition();
    void publishVelocitySetpoint(const Eigen::Vector3f& velocityNed);
    double sanitizeDt(double dtSec) const;
    const char* stateToString(RingModeState state) const;

private:
    rclcpp::Node& node_;

    std::shared_ptr<px4_ros2::TrajectorySetpointType> trajectorySetpoint_;

    rclcpp::Subscription<px4_msgs::msg::VehicleAttitude>::SharedPtr vehicleAttitudeSub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr vehicleLocalPositionSub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr targetPositionSub_;
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr targetVelocitySub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr targetValidSub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr detectorResetPub_;

    px4_msgs::msg::VehicleAttitude vehicleAttitudeMsg_{};
    bool hasVehicleAttitude_{false};

    Eigen::Vector3f currentPositionNed_{0.0f, 0.0f, 0.0f};
    float currentYawRad_{0.0f};
    bool hasVehicleLocalPosition_{false};

    RingModeState state_{RingModeState::WaitLocalPosition};
    rclcpp::Time stateEnterTime_{0, 0, RCL_ROS_TIME};
    rclcpp::Time ignoreTargetUntil_{0, 0, RCL_ROS_TIME};
    rclcpp::Time ringPassStartTime_{0, 0, RCL_ROS_TIME};

    BodyTarget target_{};
    bool targetValidReceived_{false};
    bool targetValidExternal_{false};
    bool hasForwardCommandedInPass_{false};
    bool completedReported_{false};

    double vxLastNed_{0.0};
    double vyLastNed_{0.0};
    double vzLastNed_{0.0};

    // ===== Parameters =====
    double targetTimeoutSec_{0.30};
    double detectorResetIgnoreTimeSec_{0.60};
    double minTimeBeforeFinishSec_{1.00};
    double minForwardCmdBeforeFinishMps_{0.20};
    double dtMinSec_{0.005};
    double dtMaxSec_{0.100};

    // Điều khiển sai số y/z trong body frame.
    double kpY_{0.80};
    double kdY_{0.00};
    double kpZ_{0.80};
    double kdZ_{0.00};

    double yDeadbandM_{0.03};
    double zDeadbandM_{0.04};

    double vyMaxMps_{1.20};
    double vzMaxMps_{0.70};

    // Điều khiển bay xuyên vòng theo trục x body.
    double vxMinMps_{1.50};
    double vxMaxMps_{2.00};
    double yzApproachRadiusM_{0.35};
    double yzFullSpeedRadiusM_{0.10};
    double xSpeedCurveGain_{8.00};

    // z mong muốn của tâm vòng trong body frame.
    // Nếu muốn tâm vòng nằm đúng giữa camera theo chiều cao thì để 0.0.
    double targetZOffsetM_{0.40};

    // Slew rate limit cho lệnh vận tốc NED.
    double slewXyMps2_{1.50};
    double slewZMps2_{1.20};
};