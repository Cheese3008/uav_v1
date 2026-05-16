#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <string>

namespace frame_transform
{

enum class MountMode
{
    BellyFixedCamera = 0,
    BellyFixedCameraLeft90 = 1,
    BellyFixedCameraRight90 = 2,
    BellyGimbalCamera = 3,
    FrontFixedCamera = 4
};

struct VehicleStateData
{
    Eigen::Vector3d positionWorld = Eigen::Vector3d::Zero();

    // Quaternion quay tu body FRD sang world/local NED.
    Eigen::Quaterniond worldFromBody = Eigen::Quaterniond::Identity();
};

struct TransformConfig
{
    MountMode mountMode{MountMode::BellyFixedCamera};

    std::string mountModeString{"belly_fixed_camera"};

    // Offset camera trong body FRD:
    //   X+: phia truoc UAV
    //   Y+: ben phai UAV
    //   Z+: huong xuong UAV
    Eigen::Vector3d cameraOffsetBody = Eigen::Vector3d::Zero();

    // Ma tran quay tu camera optical frame sang mount/body frame.
    //
    // Camera optical frame theo ROS/OpenCV:
    //   optical X+: ben phai anh
    //   optical Y+: xuong duoi anh
    //   optical Z+: huong nhin camera
    Eigen::Matrix3d opticalToMountRotation = Eigen::Matrix3d::Identity();
};

class FrameTransformer
{
public:
    FrameTransformer();

    /**
     * Mo ta:
     *     Cap nhat cau hinh bien doi frame.
     *
     * Input:
     *     config: cau hinh transform gom mount mode, camera offset va ma tran quay.
     *
     * Logic:
     *     Luu config vao noi bo de cac ham transform su dung.
     *
     * Output:
     *     Khong co.
     */
    void setConfig(const TransformConfig &config);

    /**
     * Mo ta:
     *     Cap nhat trang thai UAV trong he world/local NED.
     *
     * Input:
     *     vehicleState:
     *         - positionWorld: vi tri UAV trong local NED
     *         - worldFromBody: quaternion body FRD -> local NED
     *
     * Logic:
     *     Luu state va normalize quaternion.
     *
     * Output:
     *     Khong co.
     */
    void setVehicleState(const VehicleStateData &vehicleState);

    /**
     * Mo ta:
     *     Cap nhat quaternion mount -> body, dung cho camera gimbal.
     *
     * Input:
     *     bodyFromMount: quaternion mount -> body.
     *
     * Logic:
     *     Normalize quaternion, neu khong hop le thi identity.
     *
     * Output:
     *     Khong co.
     */
    void setBodyFromMountQuaternion(const Eigen::Quaterniond &bodyFromMount);

    /**
     * Mo ta:
     *     Cap nhat quaternion mount -> body bang Euler degree.
     *
     * Input:
     *     yawDeg, pitchDeg, rollDeg: goc Euler don vi do.
     *
     * Logic:
     *     Tao quaternion theo thu tu Z-Y-X roi normalize.
     *
     * Output:
     *     Khong co.
     */
    void setBodyFromMountEulerDeg(double yawDeg, double pitchDeg, double rollDeg);

    /**
     * Mo ta:
     *     Bien doi vi tri target tu camera optical frame sang world/local NED.
     *
     * Input:
     *     opticalPosition: toa do target trong camera optical frame.
     *
     * Logic:
     *     optical -> mount/body -> world/local NED, co cong camera offset.
     *
     * Output:
     *     Toa do target trong world/local NED.
     */
    Eigen::Vector3d opticalPositionToWorld(const Eigen::Vector3d &opticalPosition) const;

    /**
     * Mo ta:
     *     Bien doi quaternion target tu camera optical frame sang world/local NED.
     *
     * Input:
     *     opticalOrientation: quaternion target trong optical frame.
     *
     * Logic:
     *     optical -> mount/body -> world/local NED.
     *
     * Output:
     *     Quaternion target trong world/local NED.
     */
    Eigen::Quaterniond opticalOrientationToWorld(
        const Eigen::Quaterniond &opticalOrientation) const;

    /**
     * Mo ta:
     *     Tao config cho camera gan duoi bung, huong nhin xuong, anh canh tren la phia truoc UAV.
     *
     * Input:
     *     cameraOffsetBody: offset camera trong body FRD.
     *
     * Logic:
     *     Mapping mac dinh:
     *         optical X ->  body Y
     *         optical Y -> -body X
     *         optical Z ->  body Z
     *
     * Output:
     *     TransformConfig.
     */
    static TransformConfig makeBellyFixedCameraConfig(
        const Eigen::Vector3d &cameraOffsetBody);

    /**
     * Mo ta:
     *     Tao config cho camera gan duoi bung, huong nhin xuong,
     *     nhung camera bi xoay trai 90 do theo cach lap cua ban hien tai.
     *
     * Input:
     *     cameraOffsetBody: offset camera trong body FRD.
     *
     * Logic:
     *     Mapping:
     *         optical X -> body X
     *         optical Y -> body Y
     *         optical Z -> body Z
     *
     * Output:
     *     TransformConfig.
     */
    static TransformConfig makeBellyFixedCameraLeft90Config(
        const Eigen::Vector3d &cameraOffsetBody);

    /**
     * Mo ta:
     *     Tao config cho camera gan duoi bung, huong nhin xuong,
     *     nhung camera bi xoay phai 90 do so voi config mac dinh.
     *
     * Input:
     *     cameraOffsetBody: offset camera trong body FRD.
     *
     * Logic:
     *     Mapping:
     *         optical X -> -body X
     *         optical Y -> -body Y
     *         optical Z ->  body Z
     *
     * Output:
     *     TransformConfig.
     */
    static TransformConfig makeBellyFixedCameraRight90Config(
        const Eigen::Vector3d &cameraOffsetBody);

    /**
     * Mo ta:
     *     Tao config cho camera gimbal gan duoi bung.
     *
     * Input:
     *     cameraOffsetBody: offset camera trong body FRD.
     *
     * Logic:
     *     optical -> mount, sau do mount -> body lay tu quaternion gimbal.
     *
     * Output:
     *     TransformConfig.
     */
    static TransformConfig makeBellyGimbalCameraConfig(
        const Eigen::Vector3d &cameraOffsetBody);

    /**
     * Mo ta:
     *     Tao config cho camera gan phia truoc UAV.
     *
     * Input:
     *     cameraOffsetBody: offset camera trong body FRD.
     *
     * Logic:
     *     Camera nhin thang theo body X.
     *
     * Output:
     *     TransformConfig.
     */
    static TransformConfig makeFrontFixedCameraConfig(
        const Eigen::Vector3d &cameraOffsetBody);

    static MountMode parseMountMode(const std::string &modeString);

    static std::string mountModeToString(MountMode mountMode);

private:
    Eigen::Quaterniond bodyFromMountQuaternion() const;

    TransformConfig config_{};

    VehicleStateData vehicleState_{};

    Eigen::Quaterniond bodyFromMount_{Eigen::Quaterniond::Identity()};
};

} // namespace frame_transform