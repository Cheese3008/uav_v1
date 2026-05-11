#include "ImageTargetDetector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace point_mission_mode
{
void ImageTargetDetector::configure(const ImageDetectorParams &params)
{
    params_ = params;
    params_.minAreaPx = std::max(1, params_.minAreaPx);
    params_.maxAreaPx = std::max(params_.minAreaPx + 1, params_.maxAreaPx);
    params_.morphKernelSize = std::max(1, params_.morphKernelSize);
    if ((params_.morphKernelSize % 2) == 0)
    {
        ++params_.morphKernelSize;
    }
}

ImageTargetDetection ImageTargetDetector::detect(
    const cv::Mat &bgrImage,
    const HsvRange &hsvRange,
    const CameraIntrinsics &intrinsics,
    float projectionRangeDownM,
    cv::Mat *debugImage) const
{
    try
    {
        ImageTargetDetection result{};

        if (bgrImage.empty() || !isValidRange(hsvRange))
        {
            return result;
        }

        cv::Mat hsvImage;
        cv::cvtColor(bgrImage, hsvImage, cv::COLOR_BGR2HSV);

        cv::Mat mask = buildMask(hsvImage, hsvRange);
        const cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(params_.morphKernelSize, params_.morphKernelSize));

        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        double bestScore = -std::numeric_limits<double>::infinity();
        std::vector<cv::Point> bestContour;

        const bool intrinsicsValid = isValidIntrinsics(intrinsics);
        const cv::Point2f imageCenter(
            intrinsicsValid ? intrinsics.cx : static_cast<float>(bgrImage.cols) * 0.5f,
            intrinsicsValid ? intrinsics.cy : static_cast<float>(bgrImage.rows) * 0.5f);

        for (const auto &contour : contours)
        {
            const double area = cv::contourArea(contour);
            if (area < static_cast<double>(params_.minAreaPx) ||
                area > static_cast<double>(params_.maxAreaPx))
            {
                continue;
            }

            const double perimeter = cv::arcLength(contour, true);
            if (perimeter <= 1e-6)
            {
                continue;
            }

            const double circularity = 4.0 * M_PI * area / (perimeter * perimeter);
            if (circularity < static_cast<double>(params_.minCircularity))
            {
                continue;
            }

            cv::Point2f center;
            float radius = 0.0f;
            cv::minEnclosingCircle(contour, center, radius);
            if (radius <= 1.0f)
            {
                continue;
            }

            const double circleArea = M_PI * static_cast<double>(radius) * static_cast<double>(radius);
            const double fillRatio = area / std::max(1.0, circleArea);
            if (fillRatio < static_cast<double>(params_.minFillRatio))
            {
                continue;
            }

            const double dx = static_cast<double>(center.x - imageCenter.x);
            const double dy = static_cast<double>(center.y - imageCenter.y);
            const double dist = std::sqrt(dx * dx + dy * dy);
            const double score = area + 200.0 * circularity + 100.0 * fillRatio - 0.25 * dist;

            if (score > bestScore)
            {
                bestScore = score;
                bestContour = contour;
            }
        }

        if (bestContour.empty())
        {
            if (debugImage != nullptr)
            {
                cv::cvtColor(mask, *debugImage, cv::COLOR_GRAY2BGR);
                cv::drawMarker(*debugImage, imageCenter, cv::Scalar(255, 255, 255), cv::MARKER_CROSS, 24, 2);
            }
            return result;
        }

        cv::Point2f center;
        float radius = 0.0f;
        cv::minEnclosingCircle(bestContour, center, radius);

        result.valid = true;
        result.centerPx = Eigen::Vector2f(center.x, center.y);
        result.imageCenterPx = Eigen::Vector2f(imageCenter.x, imageCenter.y);
        result.errorPx = Eigen::Vector2f(center.x - imageCenter.x, center.y - imageCenter.y);
        result.areaPx = static_cast<float>(cv::contourArea(bestContour));
        result.radiusPx = radius;
        result.cameraInfoValid = intrinsicsValid;

        if (intrinsicsValid)
        {
            result.opticalRayNorm = Eigen::Vector2f(
                result.errorPx.x() / intrinsics.fx,
                result.errorPx.y() / intrinsics.fy);
            result.errorNorm = result.opticalRayNorm;
        }
        else
        {
            result.errorNorm = Eigen::Vector2f(
                result.errorPx.x() / std::max(1.0f, static_cast<float>(bgrImage.cols) * 0.5f),
                result.errorPx.y() / std::max(1.0f, static_cast<float>(bgrImage.rows) * 0.5f));
            result.opticalRayNorm = result.errorNorm;
        }

        if (intrinsicsValid && std::isfinite(projectionRangeDownM) && projectionRangeDownM > 0.0f)
        {
            result.metricValid = true;
            result.opticalPositionValid = true;
            result.rangeDownM = projectionRangeDownM;

            const float cameraXM = result.opticalRayNorm.x() * projectionRangeDownM;
            const float cameraYM = result.opticalRayNorm.y() * projectionRangeDownM;

            // Vi tri target trong camera optical frame:
            // optical x+ sang phai anh, optical y+ xuong anh, optical z+ huong nhin camera.
            result.targetOpticalM = Eigen::Vector3f(cameraXM, cameraYM, projectionRangeDownM);

            // Gan camera bung mac dinh, debug nhanh theo body-FRD:
            // body x = -optical_y, body y = optical_x.
            result.targetBodyXYM = Eigen::Vector2f(-cameraYM, cameraXM);
        }

        if (debugImage != nullptr)
        {
            bgrImage.copyTo(*debugImage);
            cv::circle(*debugImage, center, static_cast<int>(radius), cv::Scalar(0, 255, 0), 2);
            cv::circle(*debugImage, center, 4, cv::Scalar(0, 255, 255), -1);
            cv::drawMarker(*debugImage, imageCenter, cv::Scalar(255, 255, 255), cv::MARKER_CROSS, 24, 2);

            if (result.metricValid)
            {
                const std::string text = "FRD xy(m): " +
                    std::to_string(result.targetBodyXYM.x()).substr(0, 5) + "," +
                    std::to_string(result.targetBodyXYM.y()).substr(0, 5);
                cv::putText(*debugImage, text, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            }
        }

        return result;
    }
    catch (const std::exception &exception)
    {
        throw std::runtime_error(
            std::string("ImageTargetDetector::detect failed: ") + exception.what());
    }
    catch (...)
    {
        throw std::runtime_error("ImageTargetDetector::detect failed: unknown exception");
    }
}

cv::Mat ImageTargetDetector::buildMask(const cv::Mat &hsvImage, const HsvRange &hsvRange) const
{
    cv::Mat mask;

    const int hMin = std::clamp(hsvRange.min[0], 0, 179);
    const int sMin = std::clamp(hsvRange.min[1], 0, 255);
    const int vMin = std::clamp(hsvRange.min[2], 0, 255);
    const int hMax = std::clamp(hsvRange.max[0], 0, 179);
    const int sMax = std::clamp(hsvRange.max[1], 0, 255);
    const int vMax = std::clamp(hsvRange.max[2], 0, 255);

    if (hMin <= hMax)
    {
        cv::inRange(
            hsvImage,
            cv::Scalar(hMin, sMin, vMin),
            cv::Scalar(hMax, sMax, vMax),
            mask);
        return mask;
    }

    cv::Mat lowMask;
    cv::Mat highMask;
    cv::inRange(
        hsvImage,
        cv::Scalar(0, sMin, vMin),
        cv::Scalar(hMax, sMax, vMax),
        lowMask);
    cv::inRange(
        hsvImage,
        cv::Scalar(hMin, sMin, vMin),
        cv::Scalar(179, sMax, vMax),
        highMask);
    cv::bitwise_or(lowMask, highMask, mask);

    return mask;
}

bool ImageTargetDetector::isValidRange(const HsvRange &hsvRange) const
{
    for (int i = 0; i < 3; ++i)
    {
        if (hsvRange.min[i] < 0 || hsvRange.max[i] < 0)
        {
            return false;
        }
    }

    return true;
}

bool ImageTargetDetector::isValidIntrinsics(const CameraIntrinsics &intrinsics) const
{
    return intrinsics.valid &&
        intrinsics.width > 0 &&
        intrinsics.height > 0 &&
        std::isfinite(intrinsics.fx) &&
        std::isfinite(intrinsics.fy) &&
        std::isfinite(intrinsics.cx) &&
        std::isfinite(intrinsics.cy) &&
        intrinsics.fx > 1.0f &&
        intrinsics.fy > 1.0f;
}
} // namespace point_mission_mode
