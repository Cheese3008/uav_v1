#pragma once

#include <rclcpp/rclcpp.hpp>
#include <opencv2/video/tracking.hpp>

#include "ControlTypes.hpp"

namespace point_mission_mode
{
class Kalman1D
{
public:
    void configure(const Kalman1DParams &params);
    void reset();

    // Mo ta:
    //     Cap nhat bo loc 1D bang measurement moi.
    // Input:
    //     measurement: gia tri vi tri 1 truc, don vi met.
    //     stamp: timestamp cua measurement.
    // Logic:
    //     - Lan dau: khoi tao state [pos, vel=0].
    //     - Cac lan sau: predict theo dt bang constant velocity, sau do correct bang pos.
    // Output:
    //     true neu bo loc co state hop le sau update.
    bool update(float measurement, const rclcpp::Time &stamp);

    bool predict(const rclcpp::Time &stamp);

    bool initialized() const { return initialized_; }
    float position() const;
    float velocity() const;

private:
    void applyTransition(float dt);

private:
    Kalman1DParams params_{};
    cv::KalmanFilter filter_{};
    bool initialized_{false};
    rclcpp::Time lastStamp_{0, 0, RCL_ROS_TIME};
};
} // namespace point_mission_mode
