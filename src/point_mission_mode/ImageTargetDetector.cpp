#include "ImageTargetDetector.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace
{
constexpr int kDebugPanelWidth = 320;
constexpr int kDebugPanelHeight = 240;
constexpr int kDebugPanelHeaderHeight = 24;
constexpr float kPredictionDistanceWeight = 3.0f;
constexpr float kCenterDistanceWeight = 0.10f;

cv::Mat convertToBgr(const cv::Mat &image)
{
    if (image.empty())
    {
        return cv::Mat::zeros(kDebugPanelHeight, kDebugPanelWidth, CV_8UC3);
    }

    if (image.channels() == 1)
    {
        cv::Mat bgrImage;
        cv::cvtColor(image, bgrImage, cv::COLOR_GRAY2BGR);
        return bgrImage;
    }

    if (image.channels() == 3)
    {
        return image.clone();
    }

    cv::Mat normalized;
    image.convertTo(normalized, CV_8U);

    cv::Mat bgrImage;
    cv::cvtColor(normalized, bgrImage, cv::COLOR_GRAY2BGR);
    return bgrImage;
}

void drawPanelFrame(cv::Mat &panel, const std::string &title)
{
    if (panel.empty())
    {
        return;
    }

    cv::rectangle(
        panel,
        cv::Rect(0, 0, panel.cols, std::min(kDebugPanelHeaderHeight, panel.rows)),
        cv::Scalar(35, 35, 35),
        cv::FILLED);

    cv::rectangle(panel, cv::Rect(0, 0, panel.cols, panel.rows), cv::Scalar(220, 220, 220), 2);

    cv::putText(
        panel,
        title,
        cv::Point(8, 17),
        cv::FONT_HERSHEY_SIMPLEX,
        0.48,
        cv::Scalar(255, 255, 255),
        1,
        cv::LINE_AA);
}

cv::Mat makePanel(const cv::Mat &image, const std::string &title)
{
    cv::Mat bgrImage = convertToBgr(image);
    cv::Mat panel;
    cv::resize(bgrImage, panel, cv::Size(kDebugPanelWidth, kDebugPanelHeight), 0.0, 0.0, cv::INTER_AREA);
    drawPanelFrame(panel, title);
    return panel;
}

void putDebugLine(cv::Mat &panel, int lineIndex, const std::string &text)
{
    const int y = kDebugPanelHeaderHeight + 20 + lineIndex * 22;
    if (y >= panel.rows - 4)
    {
        return;
    }

    cv::putText(
        panel,
        text,
        cv::Point(10, y),
        cv::FONT_HERSHEY_SIMPLEX,
        0.48,
        cv::Scalar(255, 255, 255),
        1,
        cv::LINE_AA);
}

cv::Mat makeTextPanel(const std::vector<std::string> &lines, const std::string &title)
{
    cv::Mat panel = cv::Mat::zeros(kDebugPanelHeight, kDebugPanelWidth, CV_8UC3);
    panel.setTo(cv::Scalar(15, 15, 15));
    drawPanelFrame(panel, title);

    for (std::size_t i = 0; i < lines.size(); ++i)
    {
        putDebugLine(panel, static_cast<int>(i), lines[i]);
    }

    return panel;
}

std::string floatText(float value, int precision = 3)
{
    if (!std::isfinite(value))
    {
        return "nan";
    }

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(precision) << value;
    return ss.str();
}

cv::Mat buildDebugCanvas(const std::vector<cv::Mat> &panels)
{
    const cv::Mat emptyPanel = makeTextPanel({"empty"}, "DEBUG");
    std::vector<cv::Mat> normalizedPanels;
    normalizedPanels.reserve(6U);

    for (const auto &panel : panels)
    {
        normalizedPanels.push_back(panel.empty() ? emptyPanel.clone() : panel);
    }

    while (normalizedPanels.size() < 6U)
    {
        normalizedPanels.push_back(emptyPanel.clone());
    }

    cv::Mat row1;
    cv::Mat row2;
    cv::Mat canvas;
    cv::hconcat(std::vector<cv::Mat>{normalizedPanels[0], normalizedPanels[1], normalizedPanels[2]}, row1);
    cv::hconcat(std::vector<cv::Mat>{normalizedPanels[3], normalizedPanels[4], normalizedPanels[5]}, row2);
    cv::vconcat(row1, row2, canvas);
    return canvas;
}

cv::Point2f toPoint2f(const Eigen::Vector2f &value)
{
    return cv::Point2f(value.x(), value.y());
}

bool isFiniteVector2(const Eigen::Vector2f &value)
{
    return std::isfinite(value.x()) && std::isfinite(value.y());
}
} // namespace

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
    const ImageTargetLockInput &lockInput,
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

        const cv::Mat rawMask = buildMask(hsvImage, hsvRange);
        cv::Mat morphMask = rawMask.clone();

        const cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(params_.morphKernelSize, params_.morphKernelSize));

        cv::morphologyEx(morphMask, morphMask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(morphMask, morphMask, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(morphMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        const bool intrinsicsValid = isValidIntrinsics(intrinsics);
        const cv::Point2f imageCenter(
            intrinsicsValid ? intrinsics.cx : static_cast<float>(bgrImage.cols) * 0.5f,
            intrinsicsValid ? intrinsics.cy : static_cast<float>(bgrImage.rows) * 0.5f);

        const bool lockValid = lockInput.valid && isFiniteVector2(lockInput.predictedPixel);
        const cv::Point2f lockPixel = toPoint2f(lockInput.predictedPixel);

        cv::Mat contourDebug = bgrImage.clone();
        cv::Mat finalDebug = bgrImage.clone();
        cv::drawMarker(contourDebug, imageCenter, cv::Scalar(255, 255, 255), cv::MARKER_CROSS, 24, 2);
        cv::drawMarker(finalDebug, imageCenter, cv::Scalar(255, 255, 255), cv::MARKER_CROSS, 24, 2);

        if (lockValid)
        {
            cv::drawMarker(contourDebug, lockPixel, cv::Scalar(0, 140, 255), cv::MARKER_TILTED_CROSS, 22, 2);
            cv::circle(contourDebug, lockPixel, static_cast<int>(std::round(lockInput.lockGatePx)), cv::Scalar(0, 140, 255), 2);
        }

        if (lockInput.valid && isFiniteVector2(lockInput.releasePixel))
        {
            cv::drawMarker(contourDebug, toPoint2f(lockInput.releasePixel), cv::Scalar(255, 0, 255), cv::MARKER_DIAMOND, 22, 2);
        }

        if (lockInput.valid && isFiniteVector2(lockInput.nadirPixel))
        {
            cv::drawMarker(contourDebug, toPoint2f(lockInput.nadirPixel), cv::Scalar(255, 0, 180), cv::MARKER_DIAMOND, 20, 2);
        }

        double bestScore = -std::numeric_limits<double>::infinity();
        std::vector<cv::Point> bestContour;
        float bestLockDistancePx = 0.0f;

        for (const auto &contour : contours)
        {
            cv::drawContours(contourDebug, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(160, 160, 160), 1);

            const double area = cv::contourArea(contour);
            if (area < static_cast<double>(params_.minAreaPx) || area > static_cast<double>(params_.maxAreaPx))
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

            const double centerDx = static_cast<double>(center.x - imageCenter.x);
            const double centerDy = static_cast<double>(center.y - imageCenter.y);
            const double centerDist = std::sqrt(centerDx * centerDx + centerDy * centerDy);

            double lockDist = 0.0;
            if (lockValid)
            {
                const double dx = static_cast<double>(center.x - lockPixel.x);
                const double dy = static_cast<double>(center.y - lockPixel.y);
                lockDist = std::sqrt(dx * dx + dy * dy);

                if (lockInput.useLockGate && lockDist > static_cast<double>(lockInput.lockGatePx))
                {
                    cv::circle(contourDebug, center, static_cast<int>(std::round(radius)), cv::Scalar(0, 0, 255), 2);
                    continue;
                }
            }

            cv::circle(contourDebug, center, static_cast<int>(std::round(radius)), cv::Scalar(255, 0, 0), 2);

            double score = area + 200.0 * circularity + 100.0 * fillRatio - kCenterDistanceWeight * centerDist;
            if (lockValid && lockInput.usePredictionScore)
            {
                score -= kPredictionDistanceWeight * lockDist;
            }

            if (score > bestScore)
            {
                bestScore = score;
                bestContour = contour;
                bestLockDistancePx = static_cast<float>(lockDist);
            }
        }

        bool hasBestContour = !bestContour.empty();
        cv::Point2f center(0.0f, 0.0f);
        float radius = 0.0f;

        if (hasBestContour)
        {
            cv::minEnclosingCircle(bestContour, center, radius);

            result.valid = true;
            result.centerPx = Eigen::Vector2f(center.x, center.y);
            result.imageCenterPx = Eigen::Vector2f(imageCenter.x, imageCenter.y);
            result.errorPx = Eigen::Vector2f(center.x - imageCenter.x, center.y - imageCenter.y);
            result.areaPx = static_cast<float>(cv::contourArea(bestContour));
            result.radiusPx = radius;
            result.cameraInfoValid = intrinsicsValid;
            result.lockDistancePx = bestLockDistancePx;
            result.selectedByLock = lockValid;

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

                result.targetOpticalM = Eigen::Vector3f(cameraXM, cameraYM, projectionRangeDownM);
                result.targetBodyXYM = Eigen::Vector2f(-cameraYM, cameraXM);
            }

            cv::drawContours(contourDebug, std::vector<std::vector<cv::Point>>{bestContour}, -1, cv::Scalar(0, 255, 0), 3);
            cv::circle(finalDebug, center, static_cast<int>(std::round(radius)), cv::Scalar(0, 255, 0), 2);
            cv::circle(finalDebug, center, 4, cv::Scalar(0, 255, 255), -1);
            cv::line(finalDebug, imageCenter, center, cv::Scalar(0, 255, 255), 2);
        }

        if (lockValid)
        {
            cv::drawMarker(finalDebug, lockPixel, cv::Scalar(0, 140, 255), cv::MARKER_TILTED_CROSS, 24, 2);
            cv::circle(finalDebug, lockPixel, static_cast<int>(std::round(lockInput.lockGatePx)), cv::Scalar(0, 140, 255), 2);
        }

        if (lockInput.valid && isFiniteVector2(lockInput.releasePixel))
        {
            cv::drawMarker(finalDebug, toPoint2f(lockInput.releasePixel), cv::Scalar(255, 0, 255), cv::MARKER_DIAMOND, 24, 2);
        }

        if (lockInput.valid && isFiniteVector2(lockInput.nadirPixel))
        {
            cv::drawMarker(finalDebug, toPoint2f(lockInput.nadirPixel), cv::Scalar(255, 0, 180), cv::MARKER_DIAMOND, 20, 2);
        }

        if (debugImage != nullptr)
        {
            cv::Mat hsvBgr;
            cv::cvtColor(hsvImage, hsvBgr, cv::COLOR_HSV2BGR);

            std::vector<std::string> metricLines;
            metricLines.push_back(std::string("valid: ") + (result.valid ? "true" : "false"));
            metricLines.push_back("contours: " + std::to_string(contours.size()));
            metricLines.push_back("best_area_px: " + floatText(result.areaPx, 1));
            metricLines.push_back("radius_px: " + floatText(result.radiusPx, 1));
            metricLines.push_back("err_px: " + floatText(result.errorPx.x(), 1) + ", " + floatText(result.errorPx.y(), 1));
            metricLines.push_back("body_xy_m: " + floatText(result.targetBodyXYM.x(), 3) + ", " + floatText(result.targetBodyXYM.y(), 3));
            metricLines.push_back("pred_px: " + floatText(lockInput.predictedPixel.x(), 1) + ", " + floatText(lockInput.predictedPixel.y(), 1));
            metricLines.push_back("rel_px: " + floatText(lockInput.releasePixel.x(), 1) + ", " + floatText(lockInput.releasePixel.y(), 1));
            metricLines.push_back("lock_dist_px: " + floatText(result.lockDistancePx, 1));
            metricLines.push_back(std::string("camera_info: ") + (intrinsicsValid ? "true" : "false"));

            *debugImage = buildDebugCanvas(
                {
                    makePanel(bgrImage, "01 RAW BGR"),
                    makePanel(hsvBgr, "02 BGR->HSV"),
                    makePanel(rawMask, "03 HSV MASK RAW"),
                    makePanel(morphMask, "04 MORPH OPEN/CLOSE"),
                    makePanel(contourDebug, "05 CONTOUR + LOCK"),
                    hasBestContour ? makePanel(finalDebug, "06 TARGET + FUTURE") : makeTextPanel(metricLines, "06 TARGET + FUTURE")});

            cv::Mat metricPanel = makeTextPanel(metricLines, "METRIC DEBUG");
            const cv::Rect metricRoi(kDebugPanelWidth * 2, kDebugPanelHeight, kDebugPanelWidth, kDebugPanelHeight);
            metricPanel.copyTo((*debugImage)(metricRoi));
        }

        return result;
    }
    catch (const std::exception &exception)
    {
        throw std::runtime_error(std::string("ImageTargetDetector::detect failed: ") + exception.what());
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
        cv::inRange(hsvImage, cv::Scalar(hMin, sMin, vMin), cv::Scalar(hMax, sMax, vMax), mask);
        return mask;
    }

    cv::Mat lowMask;
    cv::Mat highMask;
    cv::inRange(hsvImage, cv::Scalar(0, sMin, vMin), cv::Scalar(hMax, sMax, vMax), lowMask);
    cv::inRange(hsvImage, cv::Scalar(hMin, sMin, vMin), cv::Scalar(179, sMax, vMax), highMask);
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
