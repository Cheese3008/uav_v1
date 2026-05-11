#include "FutureTargetPredictor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace point_mission_mode
{
void FutureTargetPredictor::configure(const FutureTargetPredictorParams &params)
{
    params_ = params;

    if (!std::isfinite(params_.leadSec) || params_.leadSec < 0.0f)
    {
        throw std::runtime_error("FutureTargetPredictor::configure leadSec khong hop le");
    }

    if (!std::isfinite(params_.releaseLeadSec) || params_.releaseLeadSec < 0.0f)
    {
        throw std::runtime_error("FutureTargetPredictor::configure releaseLeadSec khong hop le");
    }

    if (!std::isfinite(params_.lockGatePx) || params_.lockGatePx <= 0.0f)
    {
        throw std::runtime_error("FutureTargetPredictor::configure lockGatePx khong hop le");
    }

    if (!std::isfinite(params_.minSpeedForDynamicMps) || params_.minSpeedForDynamicMps < 0.0f)
    {
        throw std::runtime_error("FutureTargetPredictor::configure minSpeedForDynamicMps khong hop le");
    }

    if (!std::isfinite(params_.maxPredictionM) || params_.maxPredictionM <= 0.0f)
    {
        throw std::runtime_error("FutureTargetPredictor::configure maxPredictionM khong hop le");
    }
}

FutureTargetPrediction FutureTargetPredictor::predict(const FutureTargetPredictorInput &input) const
{
    FutureTargetPrediction output{};

    if (!input.filteredError.valid || !isFiniteVector2(input.filteredError.filteredBodyXY) ||
        !isFiniteVector2(input.filteredError.velocityBodyXY) || !input.intrinsics.valid ||
        !std::isfinite(input.projectionRangeDownM) || input.projectionRangeDownM <= 0.0f)
    {
        return output;
    }

    const Eigen::Vector2f currentXY = input.filteredError.filteredBodyXY;
    const Eigen::Vector2f velocityXY = input.filteredError.velocityBodyXY;
    const float speedMps = velocityXY.norm();

    Eigen::Vector2f predictedXY = currentXY + velocityXY * params_.leadSec;
    Eigen::Vector2f releaseXY = currentXY + velocityXY * params_.releaseLeadSec;

    if (predictedXY.norm() > params_.maxPredictionM)
    {
        predictedXY = predictedXY.normalized() * params_.maxPredictionM;
    }

    if (releaseXY.norm() > params_.maxPredictionM)
    {
        releaseXY = releaseXY.normalized() * params_.maxPredictionM;
    }

    output.valid = true;
    output.dynamicValid = speedMps >= params_.minSpeedForDynamicMps;
    output.currentBodyXY = currentXY;
    output.velocityBodyXY = velocityXY;
    output.predictedBodyXY = predictedXY;
    output.releaseBodyXY = releaseXY;
    output.predictedPixel = projectBodyXYToPixel(predictedXY, input.intrinsics, input.projectionRangeDownM);
    output.releasePixel = projectBodyXYToPixel(releaseXY, input.intrinsics, input.projectionRangeDownM);
    output.nadirPixel = computeNadirPixel(input.intrinsics, input.rollRad, input.pitchRad);
    output.speedMps = speedMps;
    output.leadSec = params_.leadSec;
    output.releaseLeadSec = params_.releaseLeadSec;
    output.projectionRangeM = input.projectionRangeDownM;

    return output;
}

FutureTargetPredictorParams FutureTargetPredictor::params() const
{
    return params_;
}

Eigen::Vector2f FutureTargetPredictor::projectBodyXYToPixel(
    const Eigen::Vector2f &bodyXYM,
    const CameraIntrinsics &intrinsics,
    float projectionRangeDownM) const
{
    if (!intrinsics.valid || !isFiniteVector2(bodyXYM) ||
        !std::isfinite(projectionRangeDownM) || projectionRangeDownM <= 0.0f)
    {
        return Eigen::Vector2f(0.0f, 0.0f);
    }

    // Camera bung co quy uoc debug:
    // body x = -optical_y, body y = optical_x.
    const float opticalXNorm = bodyXYM.y() / projectionRangeDownM;
    const float opticalYNorm = -bodyXYM.x() / projectionRangeDownM;

    return Eigen::Vector2f(
        intrinsics.cx + intrinsics.fx * opticalXNorm,
        intrinsics.cy + intrinsics.fy * opticalYNorm);
}

Eigen::Vector2f FutureTargetPredictor::computeNadirPixel(
    const CameraIntrinsics &intrinsics,
    float rollRad,
    float pitchRad) const
{
    if (!intrinsics.valid || !std::isfinite(rollRad) || !std::isfinite(pitchRad))
    {
        return Eigen::Vector2f(0.0f, 0.0f);
    }

    const float cosRoll = std::max(0.10f, std::abs(std::cos(rollRad)));
    const float opticalXNorm = std::tan(rollRad);
    const float opticalYNorm = std::tan(pitchRad) / cosRoll;

    return Eigen::Vector2f(
        intrinsics.cx + intrinsics.fx * opticalXNorm,
        intrinsics.cy + intrinsics.fy * opticalYNorm);
}

bool FutureTargetPredictor::isFiniteVector2(const Eigen::Vector2f &value) const
{
    return std::isfinite(value.x()) && std::isfinite(value.y());
}
} // namespace point_mission_mode
