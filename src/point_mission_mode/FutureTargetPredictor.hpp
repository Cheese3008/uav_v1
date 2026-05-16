#pragma once

#include "ControlTypes.hpp"

namespace point_mission_mode
{
class FutureTargetPredictor
{
public:
    void configure(const FutureTargetPredictorParams &params);

    // Mo ta:
    //     Du doan vi tri hien tai/tuong lai cua target tu Kalman XY va velocity XY.
    // Input:
    //     filteredError: sai so XY da loc Kalman trong body-FRD.
    //     intrinsics: thong so camera de project XY len pixel.
    //     projectionRangeDownM: khoang chieu xuong dat theo met.
    //     roll/pitch: goc nghieng UAV de ve diem nadir/debug.
    // Logic:
    //     predictedBodyXY = xy + vxy * leadSec.
    //     releaseBodyXY = xy + vxy * releaseLeadSec.
    // Output:
    //     FutureTargetPrediction gom body XY, pixel du doan va pixel nadir.
    FutureTargetPrediction predict(const FutureTargetPredictorInput &input) const;

    FutureTargetPredictorParams params() const;

private:
    FutureTargetPredictorParams params_{};

    Eigen::Vector2f projectBodyXYToPixel(
        const Eigen::Vector2f &bodyXYM,
        const CameraIntrinsics &intrinsics,
        float projectionRangeDownM) const;

    Eigen::Vector2f computeNadirPixel(
        const CameraIntrinsics &intrinsics,
        float rollRad,
        float pitchRad) const;

    bool isFiniteVector2(const Eigen::Vector2f &value) const;
};
} // namespace point_mission_mode
