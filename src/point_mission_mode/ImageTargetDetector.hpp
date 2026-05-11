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
    //     Class xu ly anh rieng: detect vong tron/khoi theo HSV va CameraInfo.
    // Input:
    //     bgrImage: anh BGR tu camera.
    //     hsvRange: nguong HSV min/max cua diem F hien tai.
    //     intrinsics: fx/fy/cx/cy lay tu /camera_down/camera_info.
    //     projectionRangeDownM: do sau optical z uoc luong tu do cao hien tai.
    // Logic:
    //     - Threshold HSV, morphology, contour scoring.
    //     - Tinh normalized optical ray.
    //     - Tao targetOpticalM = [ray_x*z, ray_y*z, z] de dua vao FrameTransformer.
    // Output:
    //     ImageTargetDetection chua pixel, optical ray, optical position va anh debug.
    ImageTargetDetection detect(
        const cv::Mat &bgrImage,
        const HsvRange &hsvRange,
        const CameraIntrinsics &intrinsics,
        float projectionRangeDownM,
        cv::Mat *debugImage = nullptr) const;

private:
    ImageDetectorParams params_{};

    cv::Mat buildMask(const cv::Mat &hsvImage, const HsvRange &hsvRange) const;
    bool isValidRange(const HsvRange &hsvRange) const;
    bool isValidIntrinsics(const CameraIntrinsics &intrinsics) const;
};
} // namespace point_mission_mode
