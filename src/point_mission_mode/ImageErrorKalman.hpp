#pragma once

#include <rclcpp/rclcpp.hpp>

#include "ControlTypes.hpp"
#include "Kalman1D.hpp"

namespace point_mission_mode
{
class ImageErrorKalman
{
public:
    void configure(const ImageErrorKalmanParams &params);
    void reset();

    // Mo ta:
    //     Loc sai so target theo 2 truc body-FRD x/y bang 2 Kalman 1D doc lap.
    // Input:
    //     rawBodyXY: [x_error_m, y_error_m].
    //     stamp: timestamp measurement anh.
    // Logic:
    //     Moi truc chay Kalman 1D state [pos, vel].
    // Output:
    //     FilteredImageError gom raw, filtered position va velocity.
    FilteredImageError update(const Eigen::Vector2f &rawBodyXY, const rclcpp::Time &stamp);

    FilteredImageError latest() const;
    bool initialized() const;

private:
    Kalman1D xFilter_{};
    Kalman1D yFilter_{};
    Eigen::Vector2f latestRaw_{0.0f, 0.0f};
};
} // namespace point_mission_mode
