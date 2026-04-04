#pragma once

#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <std_msgs/msg/string.hpp>

#include <px4_msgs/msg/vehicle_attitude.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

class TargetPoseFusionNode : public rclcpp::Node
{
public:
    TargetPoseFusionNode();

private:
    static constexpr int STATE_SIZE = 6;
    static constexpr int MEASUREMENT_SIZE = 3;

    void declareParameters();
    void initKalman();
    void resetState();

    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void resetCallback(const std_msgs::msg::String::SharedPtr msg);
    void vehicleAttitudeCallback(const px4_msgs::msg::VehicleAttitude::SharedPtr msg);
    void vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg);

    void processAndPublish();
    void predict(double dt);

    bool attitudeReady() const;
    bool localPositionReady() const;

    double getVehicleYawRad() const;
    Eigen::Matrix3d buildDeltaYawRotation(double yaw_now, double yaw_prev) const;

    Eigen::Matrix3d opticalToBodyFrdRotation() const;
    Eigen::Vector3d measurementOpticalToBody(const Eigen::Vector3d& p_opt) const;
    Eigen::Vector3d bodyToLeveledBody(const Eigen::Vector3d& p_body) const;
    Eigen::Vector3d vehicleVelocityNed() const;
    Eigen::Vector3d vehicleVelocityLeveledBody() const;

    void publishRawMeasurement(
        const Eigen::Vector3d& measurement_body_level,
        const rclcpp::Time& stamp) const;

    void publishResidual(
        const cv::Mat& residual,
        const rclcpp::Time& stamp) const;

    void publishEstimatedState(const rclcpp::Time& stamp) const;
    void publishEstimatedVelocity(const rclcpp::Time& stamp) const;

private:
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr reset_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleAttitude>::SharedPtr vehicle_attitude_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr vehicle_local_position_sub_;

    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr target_error_body_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr target_error_body_fusion_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr kalman_residual_pub_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr target_velocity_body_fusion_pub_;

    rclcpp::TimerBase::SharedPtr timer_;

    cv::KalmanFilter kf_;

    bool initialized_{false};

    rclcpp::Time last_predict_time_{0, 0, RCL_ROS_TIME};
    rclcpp::Time last_measurement_time_{0, 0, RCL_ROS_TIME};

    px4_msgs::msg::VehicleAttitude vehicle_attitude_msg_{};
    bool has_vehicle_attitude_{false};

    px4_msgs::msg::VehicleLocalPosition vehicle_local_position_msg_{};
    bool has_vehicle_local_position_{false};

    double last_level_yaw_{0.0};
    bool has_last_level_yaw_{false};

    double q_acc_x_{0.08};
    double q_acc_y_{0.15};
    double q_acc_z_{0.10};

    double r_pos_x_{0.01};
    double r_pos_y_{0.01};
    double r_pos_z_{0.015};

    double cam_offset_x_{0.12};
    double cam_offset_y_{0.03};
    double cam_offset_z_{0.242};
};