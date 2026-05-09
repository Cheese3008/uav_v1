#pragma once

#include <memory>
#include <optional>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/control/setpoint_types/experimental/trajectory.hpp>

#include <Eigen/Core>

#include "RingPassController.hpp"

class MissionNode : public px4_ros2::ModeBase
{
public:
    explicit MissionNode(rclcpp::Node& node);

    void onActivate() override;
    void onDeactivate() override;
    void updateSetpoint(float dt_s) override;

private:
    enum class RingModeState
    {
        WAIT_LOCAL_POSITION = 0,
        WAIT_RING_TARGET,
        RING_PASS,
        FINISHED
    };

private:
    /**
     * Mô tả:
     *     Đọc parameter tối thiểu cho mode bay xuyên vòng.
     *
     * Input:
     *     Không có input trực tiếp.
     *
     * Output:
     *     Cập nhật các biến parameter nội bộ của node.
     */
    void loadParameters();

    /**
     * Mô tả:
     *     Khởi tạo subscriber lấy local position từ PX4.
     *
     * Input:
     *     Không có input trực tiếp.
     *
     * Output:
     *     Tạo subscriber /fmu/out/vehicle_local_position.
     */
    void setupInterfaces();

    /**
     * Mô tả:
     *     Callback nhận vị trí local và heading hiện tại của UAV từ PX4.
     *
     * Input:
     *     msg: VehicleLocalPosition::SharedPtr, dữ liệu vị trí local NED.
     *
     * Output:
     *     Cập nhật currentPosition_, currentYaw_ và cờ hasVehicleLocalPosition_.
     */
    void vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg);

    /**
     * Mô tả:
     *     Chuyển trạng thái nội bộ của mode và log tên trạng thái mới.
     *
     * Input:
     *     newState: trạng thái mới cần chuyển sang.
     *
     * Output:
     *     Cập nhật state_, stateEnterTime_ và stateFirstTick_.
     */
    void enterState(RingModeState newState);

    /**
     * Mô tả:
     *     Gửi setpoint giữ nguyên vị trí hiện tại.
     *
     * Input:
     *     Không có input trực tiếp.
     *
     * Output:
     *     Publish position setpoint bằng currentPosition_.
     */
    void publishHoldPosition();

    /**
     * Mô tả:
     *     Gửi velocity setpoint NED cho PX4.
     *
     * Input:
     *     velocityNed: Eigen::Vector3f, vận tốc mong muốn trong hệ NED.
     *
     * Output:
     *     Publish trajectory velocity setpoint, giữ yaw hiện tại.
     */
    void publishVelocitySetpoint(const Eigen::Vector3f& velocityNed);

    /**
     * Mô tả:
     *     Ép dt vào khoảng an toàn để tránh slew quá lớn khi callback bị trễ.
     *
     * Input:
     *     dtSec: thời gian vòng lặp do PX4 ROS2 mode truyền vào.
     *
     * Output:
     *     dt đã được giới hạn trong [dtMin, dtMax].
     */
    float sanitizeDt(float dtSec) const;

    const char* ringModeStateToString(RingModeState state) const;

private:
    rclcpp::Node& node_;
    RingPassController ringPassController_;

    std::shared_ptr<px4_ros2::TrajectorySetpointType> trajectorySetpoint_;

    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr vehicleLocalPositionSub_;

    rclcpp::Time stateEnterTime_{0, 0, RCL_ROS_TIME};

    Eigen::Vector3f currentPosition_{0.0f, 0.0f, 0.0f};
    float currentYaw_{0.0f};
    bool hasVehicleLocalPosition_{false};

    RingModeState state_{RingModeState::WAIT_LOCAL_POSITION};
    bool stateFirstTick_{true};

    float paramDtMinSec_{0.005f};
    float paramDtMaxSec_{0.10f};
};
