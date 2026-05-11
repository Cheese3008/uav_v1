#include "ImageErrorKalman.hpp"

#include <cmath>
#include <stdexcept>

namespace point_mission_mode
{
void ImageErrorKalman::configure(const ImageErrorKalmanParams &params)
{
    xFilter_.configure(params.x);
    yFilter_.configure(params.y);
    reset();
}

void ImageErrorKalman::reset()
{
    xFilter_.reset();
    yFilter_.reset();
    latestRaw_.setZero();
}

FilteredImageError ImageErrorKalman::update(const Eigen::Vector2f &rawBodyXY, const rclcpp::Time &stamp)
{
    if (!std::isfinite(rawBodyXY.x()) || !std::isfinite(rawBodyXY.y()))
    {
        return latest();
    }

    latestRaw_ = rawBodyXY;
    xFilter_.update(rawBodyXY.x(), stamp);
    yFilter_.update(rawBodyXY.y(), stamp);
    return latest();
}

FilteredImageError ImageErrorKalman::latest() const
{
    FilteredImageError output{};
    output.valid = initialized();
    output.rawBodyXY = latestRaw_;
    output.filteredBodyXY = Eigen::Vector2f(xFilter_.position(), yFilter_.position());
    output.velocityBodyXY = Eigen::Vector2f(xFilter_.velocity(), yFilter_.velocity());
    return output;
}

bool ImageErrorKalman::initialized() const
{
    return xFilter_.initialized() && yFilter_.initialized();
}
} // namespace point_mission_mode
