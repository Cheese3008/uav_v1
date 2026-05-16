#pragma once

#include <opencv2/core.hpp>

namespace yellow_pencil
{

struct DetectParams
{
	cv::Scalar hsv_low{15, 60, 60};
	cv::Scalar hsv_high{45, 255, 255};
	int morph_kernel_px = 5;
	double min_area_px = 800.0;
};

struct DetectResult
{
	bool valid = false;
	cv::Rect bbox;
	cv::Point2f center{0.f, 0.f};
	double area = 0.0;
};

bool detectYellowPencil(const cv::Mat &bgr, DetectResult &out, const DetectParams &p = {});

} // namespace yellow_pencil
