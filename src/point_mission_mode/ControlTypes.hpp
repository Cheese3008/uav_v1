#pragma once

#include <array>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace point_mission_mode
{
// ==== HSV threshold types ====
struct HsvRange
{
    // OpenCV HSV: H [0..179], S [0..255], V [0..255].
    // Neu min[0] > max[0], detector se tach hue thanh 2 khoang de ho tro mau do bi wrap.
    std::array<int, 3> min{0, 0, 0};
    std::array<int, 3> max{179, 255, 255};
};

// ==== Camera model types ====
struct CameraIntrinsics
{
    // CameraInfo theo ROS: K = [fx, 0, cx, 0, fy, cy, 0, 0, 1].
    bool valid{false};
    int width{0};
    int height{0};
    float fx{0.0f};
    float fy{0.0f};
    float cx{0.0f};
    float cy{0.0f};
};

// ==== Transform / vehicle state types ====
struct CameraMountParams
{
    // Camera gan cung duoi bung UAV, offset nam trong body FRD cua PX4.
    // body x+: truoc, body y+: phai, body z+: xuong.
    float camOffsetX{0.0f};
    float camOffsetY{0.0f};
    float camOffsetZ{-0.10f};
};

struct VehicleStateNed
{
    bool valid{false};
    Eigen::Vector3f positionNed{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f velocityNed{0.0f, 0.0f, 0.0f};
    Eigen::Quaternionf worldFromBody{1.0f, 0.0f, 0.0f, 0.0f};
};

// ==== Kalman 1D / XY target error types ====
struct Kalman1DParams
{
    // Mo hinh 1D constant velocity: state [position, velocity].
    float qAcc{0.04f};
    float rPos{0.02f};
    float initialPositionVariance{0.20f};
    float initialVelocityVariance{2.0f};
    float maxPredictDt{0.20f};
};

struct ImageErrorKalmanParams
{
    Kalman1DParams x{};
    Kalman1DParams y{};
};

struct FilteredImageError
{
    bool valid{false};
    Eigen::Vector2f rawBodyXY{0.0f, 0.0f};
    Eigen::Vector2f filteredBodyXY{0.0f, 0.0f};
    Eigen::Vector2f velocityBodyXY{0.0f, 0.0f};
};

// ==== Future target prediction types ====
struct FutureTargetPredictorParams
{
    // leadSec: thoi gian du doan target de ve/lock anh.
    // releaseLeadSec: thoi gian den luc bong/chan cham muc tieu sau khi kich tha.
    float leadSec{0.25f};
    float releaseLeadSec{0.35f};
    float lockGatePx{90.0f};
    float minSpeedForDynamicMps{0.05f};
    float maxPredictionM{3.0f};
    bool usePredictionLock{true};
};

struct FutureTargetPrediction
{
    bool valid{false};
    bool dynamicValid{false};
    Eigen::Vector2f currentBodyXY{0.0f, 0.0f};
    Eigen::Vector2f velocityBodyXY{0.0f, 0.0f};
    Eigen::Vector2f predictedBodyXY{0.0f, 0.0f};
    Eigen::Vector2f releaseBodyXY{0.0f, 0.0f};
    Eigen::Vector2f predictedPixel{0.0f, 0.0f};
    Eigen::Vector2f releasePixel{0.0f, 0.0f};
    Eigen::Vector2f nadirPixel{0.0f, 0.0f};
    float speedMps{0.0f};
    float leadSec{0.0f};
    float releaseLeadSec{0.0f};
    float projectionRangeM{0.0f};
};

struct FutureTargetPredictorInput
{
    FilteredImageError filteredError{};
    CameraIntrinsics intrinsics{};
    float projectionRangeDownM{0.0f};
    float rollRad{0.0f};
    float pitchRad{0.0f};
};

// ==== Image detector types ====
struct ImageDetectorParams
{
    int minAreaPx{250};
    int maxAreaPx{250000};
    float minCircularity{0.55f};
    float minFillRatio{0.45f};
    int morphKernelSize{5};
};

struct ImageTargetLockInput
{
    bool valid{false};
    bool useLockGate{true};
    bool usePredictionScore{true};
    Eigen::Vector2f predictedPixel{0.0f, 0.0f};
    Eigen::Vector2f releasePixel{0.0f, 0.0f};
    Eigen::Vector2f nadirPixel{0.0f, 0.0f};
    float lockGatePx{90.0f};
};

struct ImageTargetDetection
{
    bool valid{false};

    // Ket qua pixel.
    Eigen::Vector2f centerPx{0.0f, 0.0f};
    Eigen::Vector2f imageCenterPx{0.0f, 0.0f};
    Eigen::Vector2f errorPx{0.0f, 0.0f};

    // errorNorm = normalized optical ray [(u-cx)/fx, (v-cy)/fy] khi co CameraInfo.
    Eigen::Vector2f errorNorm{0.0f, 0.0f};
    bool cameraInfoValid{false};
    Eigen::Vector2f opticalRayNorm{0.0f, 0.0f};

    // Vi tri target trong camera optical frame, don vi met.
    // optical x+: phai anh, optical y+: xuong anh, optical z+: huong nhin camera.
    bool opticalPositionValid{false};
    Eigen::Vector3f targetOpticalM{0.0f, 0.0f, 0.0f};

    // Uoc luong nhanh trong body-FRD de debug.
    bool metricValid{false};
    Eigen::Vector2f targetBodyXYM{0.0f, 0.0f};
    float rangeDownM{0.0f};

    float areaPx{0.0f};
    float radiusPx{0.0f};
    float lockDistancePx{0.0f};
    bool selectedByLock{false};
};

// ==== Visual servo control types ====
struct ImageErrorControllerParams
{
    // Dieu khien theo sai so XY da loc Kalman trong body-FRD.
    float kpX{0.6f};
    float kpY{0.6f};
    float maxXYVelocity{0.45f};
    float centerToleranceM{0.12f};
    float stableHoldSec{1.0f};
    float dropDelaySec{1.0f};
    bool invertX{false};
    bool invertY{false};
};

struct ImageErrorControllerInput
{
    FilteredImageError filteredError{};
};

struct ImageErrorControllerOutput
{
    Eigen::Vector3f velocityBodyFrd{0.0f, 0.0f, 0.0f};
    bool centered{false};
    Eigen::Vector2f controlErrorM{0.0f, 0.0f};
};

struct ImageServoParams
{
    // Giu lai nhom tham so chung de quan ly timeout/projection.
    float stableHoldSec{1.0f};
    float dropDelaySec{1.0f};
    bool useCurrentAltitudeForProjection{true};
    float fixedProjectionRangeM{3.0f};
    float minProjectionRangeM{0.3f};
    float maxProjectionRangeM{10.0f};
};


struct MissionGeometryParams
{
    float takeoffAltitudeM{3.0f};
    float xDistanceM{8.0f};
    float centerOffsetXM{2.5f};
    float lateralOffsetYM{2.5f};
    float f5OffsetXM{2.5f};
    float lateralSign{-1.0f};
};

// ==== Mission point types ====
struct PointDynamicParams
{
    // movingTarget=true: dung Kalman velocity de tao releaseBodyXY = xy + v * releaseLeadSec.
    bool movingTarget{false};
    float observeSec{0.35f};
    float predictionLeadSec{0.25f};
    float releaseLeadSec{0.35f};
    float dynamicDropDelaySec{0.10f};
    float releaseGateM{0.18f};
    float minSpeedMps{0.05f};
    bool usePredictedErrorForServo{true};
};

struct point
{
    std::string name;

    // Toa do tuong doi trong he BODY-FRD tai thoi diem kich node.
    // x: truc X body-FRD cua UAV.
    // y: truc Y body-FRD cua UAV, ben trai la gia tri am neu body Y+ la ben phai.
    // z: Down cua UAV, nen bay len 3m se la z = -3.0.
    Eigen::Vector3f relativeBodyFrd{0.0f, 0.0f, 0.0f};

    // Toa do absolute NED duoc tinh sau khi capture vi tri + heading ban dau.
    Eigen::Vector3f absoluteNed{0.0f, 0.0f, 0.0f};

    HsvRange hsvRange{};
    bool useImageServo{true};
    bool dropCommandEnable{false};
    int dropLegId{0};
    PointDynamicParams dynamic{};
};

// ==== Coordinate navigator types ====
struct GotoParams
{
    float maxHorizontalSpeed{1.2f};
    float maxVerticalSpeed{0.6f};
    float acceptanceXY{0.20f};
    float acceptanceZ{0.15f};
};

} // namespace point_mission_mode
