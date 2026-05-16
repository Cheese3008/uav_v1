#include "ImageErrorVelocityController.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace point_mission_mode
{
void ImageErrorVelocityController::configure(const ImageErrorControllerParams &params)
{
    params_ = params;

    if (!std::isfinite(params_.kpX) || params_.kpX < 0.0f)
    {
        throw std::runtime_error("ImageErrorVelocityController::configure kpX khong hop le");
    }

    if (!std::isfinite(params_.kpY) || params_.kpY < 0.0f)
    {
        throw std::runtime_error("ImageErrorVelocityController::configure kpY khong hop le");
    }

    if (!std::isfinite(params_.maxXYVelocity) || params_.maxXYVelocity <= 0.0f)
    {
        throw std::runtime_error("ImageErrorVelocityController::configure maxXYVelocity khong hop le");
    }

    if (!std::isfinite(params_.centerToleranceM) || params_.centerToleranceM < 0.0f)
    {
        throw std::runtime_error("ImageErrorVelocityController::configure centerToleranceM khong hop le");
    }
}

void ImageErrorVelocityController::reset()
{
}

ImageErrorControllerOutput ImageErrorVelocityController::update(const ImageErrorControllerInput &input) const
{
    ImageErrorControllerOutput output{};

    if (!input.filteredError.valid)
    {
        return output;
    }

    float xErrorM = input.filteredError.filteredBodyXY.x();
    float yErrorM = input.filteredError.filteredBodyXY.y();

    if (params_.invertX)
    {
        xErrorM = -xErrorM;
    }

    if (params_.invertY)
    {
        yErrorM = -yErrorM;
    }

    float vxBody = params_.kpX * xErrorM;
    float vyBody = params_.kpY * yErrorM;

    vxBody = std::clamp(vxBody, -params_.maxXYVelocity, params_.maxXYVelocity);
    vyBody = std::clamp(vyBody, -params_.maxXYVelocity, params_.maxXYVelocity);

    output.controlErrorM = Eigen::Vector2f(xErrorM, yErrorM);
    output.centered = output.controlErrorM.norm() <= params_.centerToleranceM;
    output.velocityBodyFrd = Eigen::Vector3f(vxBody, vyBody, 0.0f);

    if (output.centered)
    {
        output.velocityBodyFrd.setZero();
    }

    return output;
}
} // namespace point_mission_mode
