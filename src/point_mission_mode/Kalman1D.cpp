#include "Kalman1D.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace point_mission_mode
{
void Kalman1D::configure(const Kalman1DParams &params)
{
    params_ = params;

    if (!std::isfinite(params_.qAcc) || params_.qAcc < 0.0f)
    {
        throw std::runtime_error("Kalman1D::configure qAcc khong hop le");
    }

    if (!std::isfinite(params_.rPos) || params_.rPos <= 0.0f)
    {
        throw std::runtime_error("Kalman1D::configure rPos khong hop le");
    }

    params_.initialPositionVariance = std::max(1e-6f, params_.initialPositionVariance);
    params_.initialVelocityVariance = std::max(1e-6f, params_.initialVelocityVariance);
    params_.maxPredictDt = std::max(1e-3f, params_.maxPredictDt);

    filter_ = cv::KalmanFilter(2, 1, 0, CV_64F);
    filter_.transitionMatrix = cv::Mat::eye(2, 2, CV_64F);
    filter_.measurementMatrix = cv::Mat::zeros(1, 2, CV_64F);
    filter_.measurementMatrix.at<double>(0, 0) = 1.0;
    filter_.measurementNoiseCov = cv::Mat::eye(1, 1, CV_64F);
    filter_.measurementNoiseCov.at<double>(0, 0) = static_cast<double>(params_.rPos);
    filter_.statePost = cv::Mat::zeros(2, 1, CV_64F);
    filter_.statePre = cv::Mat::zeros(2, 1, CV_64F);

    reset();
}

void Kalman1D::reset()
{
    initialized_ = false;
    lastStamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);

    filter_.statePost = cv::Mat::zeros(2, 1, CV_64F);
    filter_.statePre = cv::Mat::zeros(2, 1, CV_64F);
    filter_.errorCovPost = cv::Mat::eye(2, 2, CV_64F);
    filter_.errorCovPost.at<double>(0, 0) = static_cast<double>(params_.initialPositionVariance);
    filter_.errorCovPost.at<double>(1, 1) = static_cast<double>(params_.initialVelocityVariance);
}

bool Kalman1D::update(float measurement, const rclcpp::Time &stamp)
{
    if (!std::isfinite(measurement))
    {
        return initialized_;
    }

    if (!initialized_)
    {
        filter_.statePost.at<double>(0, 0) = static_cast<double>(measurement);
        filter_.statePost.at<double>(1, 0) = 0.0;
        filter_.statePre = filter_.statePost.clone();
        initialized_ = true;
        lastStamp_ = stamp;
        return true;
    }

    if (stamp.nanoseconds() > 0 && lastStamp_.nanoseconds() > 0)
    {
        double dt = (stamp - lastStamp_).seconds();
        if (dt < 0.0)
        {
            return true;
        }

        dt = std::min(dt, static_cast<double>(params_.maxPredictDt));
        if (dt > 1e-6)
        {
            applyTransition(static_cast<float>(dt));
            filter_.predict();
        }
    }

    cv::Mat measurementMat(1, 1, CV_64F);
    measurementMat.at<double>(0, 0) = static_cast<double>(measurement);
    filter_.correct(measurementMat);
    lastStamp_ = stamp;
    return true;
}

bool Kalman1D::predict(const rclcpp::Time &stamp)
{
    if (!initialized_ || stamp.nanoseconds() == 0 || lastStamp_.nanoseconds() == 0)
    {
        return initialized_;
    }

    double dt = (stamp - lastStamp_).seconds();
    if (dt < 0.0)
    {
        return initialized_;
    }

    dt = std::min(dt, static_cast<double>(params_.maxPredictDt));
    if (dt > 1e-6)
    {
        applyTransition(static_cast<float>(dt));
        filter_.predict();
        filter_.statePost = filter_.statePre.clone();
        lastStamp_ = stamp;
    }

    return initialized_;
}

float Kalman1D::position() const
{
    if (!initialized_)
    {
        return 0.0f;
    }

    return static_cast<float>(filter_.statePost.at<double>(0, 0));
}

float Kalman1D::velocity() const
{
    if (!initialized_)
    {
        return 0.0f;
    }

    return static_cast<float>(filter_.statePost.at<double>(1, 0));
}

void Kalman1D::applyTransition(float dt)
{
    const double dtValue = static_cast<double>(dt);
    const double dt2 = dtValue * dtValue;
    const double dt3 = dt2 * dtValue;
    const double dt4 = dt3 * dtValue;
    const double q = static_cast<double>(params_.qAcc);

    filter_.transitionMatrix = cv::Mat::eye(2, 2, CV_64F);
    filter_.transitionMatrix.at<double>(0, 1) = dtValue;

    filter_.processNoiseCov = cv::Mat::zeros(2, 2, CV_64F);
    filter_.processNoiseCov.at<double>(0, 0) = 0.25 * dt4 * q;
    filter_.processNoiseCov.at<double>(0, 1) = 0.50 * dt3 * q;
    filter_.processNoiseCov.at<double>(1, 0) = 0.50 * dt3 * q;
    filter_.processNoiseCov.at<double>(1, 1) = dt2 * q;
}
} // namespace point_mission_mode
