#pragma once

#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace kalman_filter_data
{

/**
 * Mô tả:
 *     Kiểu mount camera đang sử dụng.
 *
 * Ý nghĩa:
 *     - BellyFixedCamera: camera gắn cố định dưới bụng UAV, nhìn xuống.
 *     - BellyGimbalCamera: camera dưới bụng nhưng có gimbal.
 *     - FrontFixedCamera: camera gắn phía trước UAV.
 */
enum class MountMode
{
    BellyFixedCamera = 0,
    BellyGimbalCamera = 1,
    FrontFixedCamera = 2
};

/**
 * Mô tả:
 *     Cấu hình biến đổi hệ trục cho camera.
 *
 * Logic hệ trục:
 *     camera optical frame
 *         -> mount frame
 *         -> body FRD frame
 *         -> world/NED frame
 *
 * Trong hệ PX4:
 *     body FRD:
 *         X: forward
 *         Y: right
 *         Z: down
 *
 *     world NED:
 *         X: North
 *         Y: East
 *         Z: Down
 */
struct TransformConfig
{
    MountMode mountMode{MountMode::BellyFixedCamera};

    std::string mountModeString{"belly_fixed_camera"};

    // Vị trí camera trong body frame FRD.
    // Nếu camera nằm dưới bụng 9 cm thì cam_offset_z nên là +0.09.
    Eigen::Vector3d cameraOffsetBody{0.0, 0.0, 0.0};

    // Ma trận quay từ camera optical frame sang mount frame.
    Eigen::Matrix3d opticalToMountRotation{Eigen::Matrix3d::Identity()};
};

/**
 * Mô tả:
 *     Trạng thái UAV dùng cho FrameTransformer.
 *
 * positionWorld:
 *     Vị trí UAV trong world frame.
 *     Với code mới dùng PX4 thì đây là PX4 local NED:
 *         x = North
 *         y = East
 *         z = Down
 *
 * worldFromBody:
 *     Quaternion quay vector từ body FRD sang world/NED.
 */
struct VehicleStateData
{
    Eigen::Vector3d positionWorld{0.0, 0.0, 0.0};

    Eigen::Quaterniond worldFromBody{Eigen::Quaterniond::Identity()};
};

} // namespace kalman_filter_data