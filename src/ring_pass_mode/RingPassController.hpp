#pragma once

#include <memory>
#include <optional>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>

#include <px4_msgs/msg/vehicle_attitude.hpp>
#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/control/setpoint_types/experimental/trajectory.hpp>
#include <std_msgs/msg/string.hpp>

class RingPassController
{
public:
    explicit RingPassController(rclcpp::Node& node);

    void reset();

    // Gọi mỗi vòng lặp control
    Eigen::Vector3f update(float dt_s);

    // Trạng thái
    bool isReady() const;
    bool hasValidTarget() const;
    bool isFinished() const;

private:
    struct TargetState
    {
        bool valid{false};
        rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
        Eigen::Vector3f value{Eigen::Vector3f::Zero()};
        Eigen::Vector3f velocity{Eigen::Vector3f::Zero()};
    };

private:
    void loadParameters();

    void vehicleAttitudeCallback(const px4_msgs::msg::VehicleAttitude::SharedPtr msg);
    void targetErrorCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void targetVelocityCallback(const geometry_msgs::msg::TwistStamped::SharedPtr msg);

    bool attitudeReady() const;
    bool targetTimedOut() const;

    float getVehicleYaw() const;
    float applySlew(float cmd, float prev, float accel_limit, float dt_s) const;
    float computeForwardVelocity(float center_error) const;
    Eigen::Vector2f bodyToWorldXY(const Eigen::Vector2f& body_xy, float yaw) const;

    Eigen::Vector3f computeVelocityCommand(float dt_s);
    Eigen::Vector3f computeHoldCommand(float dt_s);

private:
    rclcpp::Node& node_;

    rclcpp::Subscription<px4_msgs::msg::VehicleAttitude>::SharedPtr vehicle_attitude_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_error_sub_;
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr target_velocity_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr ring_detect_reset_pub_;

    px4_msgs::msg::VehicleAttitude vehicle_attitude_msg_{};
    float missionTargetZ() const;
    bool has_vehicle_attitude_{false};
    void publishDetectorReset();

    TargetState target_{};

    float vx_last_{0.0f};
    float vy_last_{0.0f};
    float vz_last_{0.0f};

    bool finished_{false};

    // Parameters
    float param_target_timeout_{0.3f};

    float param_kp_lat_{0.8f};
    float param_kd_lat_{0.0f};

    float param_kp_z_{0.8f};
    float param_kd_z_{0.0f};

    float param_lat_deadband_{0.03f};
    float param_z_deadband_{0.04f};

    float param_v_forward_min_{1.5f};
    float param_v_forward_max_{2.0f};
    float param_r_approach_{0.35f};
    float param_r_forward_full_{0.10f};
    float param_klog_forward_{8.0f};

    float param_v_lateral_max_{1.2f};
    float param_v_vertical_max_{0.7f};

    float param_slew_xy_{1.5f};
    float param_slew_z_{1.2f};

    float param_height_offset_{0.40f};

    // Điều kiện coi như đã xuyên tâm vòng
    float param_finish_forward_error_{0.05f};
};