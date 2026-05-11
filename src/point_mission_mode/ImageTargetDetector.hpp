#pragma once

#include <opencv2/core.hpp>

#include "ControlTypes.hpp"

namespace point_mission_mode
{
class ImageTargetDetector
{
public:
    void configure(const ImageDetectorParams &params);

    // Mo ta:
    //     Detect vong tron/khoi theo HSV va CameraInfo.
    // Input:
    //     bgrImage: anh BGR tu camera.
    //     hsvRange: nguong HSV min/max cua diem F hien tai.
    //     intrinsics: fx/fy/cx/cy lay tu /camera_down/camera_info.
    //     projectionRangeDownM: do cao/chieu sau uoc luong theo huong Down, don vi met.
    //     lockInput: diem pixel du doan tu FutureTargetPredictor de lock target.
    // Logic:
    //     - Threshold HSV, morphology, find contour.
    //     - Neu co lockInput.valid thi uu tien contour gan predictedPixel.
    //     - Neu useLockGate=true thi loai contour ngoai gate de tranh nhay target.
    // Output:
    //     ImageTargetDetection va anh debug gom raw/hsv/mask/morph/contour/output.
    ImageTargetDetection detect(
        const cv::Mat &bgrImage,
        const HsvRange &hsvRange,
        const CameraIntrinsics &intrinsics,
        float projectionRangeDownM,
        const ImageTargetLockInput &lockInput,
        cv::Mat *debugImage = nullptr) const;

private:
    ImageDetectorParams params_{};

    cv::Mat buildMask(const cv::Mat &hsvImage, const HsvRange &hsvRange) const;
    bool isValidRange(const HsvRange &hsvRange) const;
    bool isValidIntrinsics(const CameraIntrinsics &intrinsics) const;
};
} // namespace point_mission_mode
