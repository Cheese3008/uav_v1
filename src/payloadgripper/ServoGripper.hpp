#pragma once

#include <string>

#include <px4_msgs/msg/vehicle_command.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>

class ServoGripper
{
public:
    explicit ServoGripper(rclcpp::Node &node);

    /**
     * Mô tả:
     *     Kích gripper sang trạng thái gắp và giữ cứng.
     *
     * Input:
     *     forceCommand: true nếu muốn gửi lại lệnh dù trạng thái nội bộ đã là gắp.
     *
     * Logic:
     *     - Nếu đã gắp và không ép gửi lại thì chỉ publish debug.
     *     - Nếu cần gắp thì gửi VEHICLE_CMD_DO_SET_ACTUATOR = 187.
     *     - Giá trị gắp mặc định là +1.0, tương tự lệnh test param1: 1.0.
     *     - Cập nhật latch grabbed_=true để giữ trạng thái.
     *
     * Output/topic publish:
     *     - /fmu/in/vehicle_command: px4_msgs::msg::VehicleCommand.
     *     - /payload_gripper/gripper_grabbed_debug: std_msgs::msg::Bool true.
     */
    void grabAndHold(bool forceCommand = false);

    /**
     * Mô tả:
     *     Kích gripper sang trạng thái nhả.
     *
     * Input:
     *     forceCommand: true nếu muốn gửi lại lệnh dù trạng thái nội bộ đã là nhả.
     *
     * Logic:
     *     - Nếu đã nhả và không ép gửi lại thì chỉ publish debug.
     *     - Nếu cần nhả thì gửi VEHICLE_CMD_DO_SET_ACTUATOR = 187.
     *     - Giá trị nhả mặc định là -1.0, tương tự lệnh test param1: -1.0.
     *     - Cập nhật latch grabbed_=false.
     *
     * Output/topic publish:
     *     - /fmu/in/vehicle_command: px4_msgs::msg::VehicleCommand.
     *     - /payload_gripper/gripper_grabbed_debug: std_msgs::msg::Bool false.
     */
    void release(bool forceCommand = false);

    /**
     * Mô tả:
     *     Reset trạng thái latch nội bộ mà không gửi lệnh xuống PX4.
     *
     * Input:
     *     grabbed: trạng thái phần mềm mới.
     *
     * Logic:
     *     - Chỉ cập nhật biến grabbed_.
     *     - Dùng khi cần đồng bộ trạng thái debug mà không làm servo di chuyển.
     *
     * Output/topic publish:
     *     - /payload_gripper/gripper_grabbed_debug.
     */
    void resetState(bool grabbed);

    /**
     * Mô tả:
     *     Publish lại trạng thái debug hiện tại.
     *
     * Input:
     *     Không có.
     *
     * Logic:
     *     - true nghĩa là đang gắp/giữ.
     *     - false nghĩa là đang nhả/mở.
     *
     * Output/topic publish:
     *     - /payload_gripper/gripper_grabbed_debug.
     */
    void publishDebugState() const;

    /**
     * Mô tả:
     *     Trả về trạng thái gắp hiện tại.
     *
     * Input:
     *     Không có.
     *
     * Logic:
     *     - Đọc biến latch grabbed_.
     *
     * Output:
     *     - true nếu đang gắp, false nếu đang nhả.
     */
    bool isGrabbed() const;

private:
    /**
     * Mô tả:
     *     Đọc parameter điều khiển gripper.
     *
     * Input:
     *     Không có.
     *
     * Logic:
     *     - Đọc topic VehicleCommand và topic debug.
     *     - Đọc actuator_index, grab_value, release_value, repeat_command_count.
     *     - Clamp actuator_index trong khoảng 1..6 và value trong khoảng -1..1.
     *
     * Output:
     *     - Cập nhật biến cấu hình nội bộ.
     */
    void loadParameters();

    /**
     * Mô tả:
     *     Gửi lệnh actuator xuống PX4 bằng VehicleCommand.
     *
     * Input:
     *     actuatorValue: giá trị chuẩn hóa trong khoảng [-1.0, 1.0].
     *
     * Logic:
     *     - Dùng VEHICLE_CMD_DO_SET_ACTUATOR = 187.
     *     - actuator_index=1 thì ghi actuatorValue vào param1.
     *     - actuator_index=2 thì ghi actuatorValue vào param2, tương tự tới param6.
     *     - Các param actuator còn lại để 0.0 giống lệnh test thủ công.
     *
     * Output/topic publish:
     *     - /fmu/in/vehicle_command.
     */
    void publishActuatorValue(float actuatorValue) const;

private:
    rclcpp::Node &node_;

    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr vehicleCommandPublisher_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr debugStatePublisher_;

    std::string vehicleCommandTopic_{"/fmu/in/vehicle_command"};
    std::string gripperDebugTopic_{"/payload_gripper/gripper_grabbed_debug"};

    int actuatorIndex_{1};
    float grabValue_{1.0f};
    float releaseValue_{-1.0f};
    int repeatCommandCount_{3};

    bool grabbed_{false};
};
