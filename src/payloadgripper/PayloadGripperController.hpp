#pragma once

#include <memory>
#include <optional>
#include <string>

#include <Eigen/Core>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/control/setpoint_types/experimental/trajectory.hpp>
#include <px4_ros2/odometry/local_position.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>

#include "ServoGripper.hpp"

class PayloadGripperController : public px4_ros2::ModeBase
{
public:
    explicit PayloadGripperController(rclcpp::Node &node);

    void onActivate() override;
    void onDeactivate() override;
    void updateSetpoint(float dt_s) override;

private:
    enum class State
    {
        SearchObject,
        ApproachObject,
        DescendToGrabHeight,
        GrabReady,
        ClimbToSavedAltitude
    };

    struct ObjectWorldState
    {
        Eigen::Vector3f positionWorld{0.0f, 0.0f, 0.0f};
        rclcpp::Time timestamp{0, 0, RCL_ROS_TIME};
        bool validPose{false};
    };

private:
    /**
     * Mô tả:
     *     Đọc toàn bộ parameter dùng cho controller bộ gắp.
     *
     * Input:
     *     Không có.
     *
     * Logic:
     *     - Đọc topic object pose, object valid và parameter state machine
     *     - Đọc gain điều khiển XY để đưa UAV vào tâm vật
     *     - Đọc bán kính vùng tâm, điều kiện ổn định, độ cao gắp, vận tốc hạ và servo
     *
     * Output:
     *     Cập nhật các biến parameter nội bộ.
     */
    void loadParameters();

    /**
     * Mô tả:
     *     Reset toàn bộ biến runtime cho một lượt gắp mới.
     *
     * Input:
     *     Không có.
     *
     * Logic:
     *     - Xóa pose object cũ
     *     - Reset timer của từng state
     *     - Reset vận tốc lọc, cờ đã gắp, cờ completed và độ cao lưu
     *
     * Output:
     *     Controller trở về trạng thái runtime sạch trước khi vào SearchObject.
     */
    void resetMissionRuntimeState();

    /**
     * Mô tả:
     *     Luôn gửi lệnh mở gripper ở đầu mode.
     *
     * Input:
     *     Không có.
     *
     * Logic:
     *     - Gọi ServoGripper::release(true) để ép gửi lệnh release xuống PX4
     *     - Không phụ thuộc latch nội bộ grabbed_
     *
     * Output/topic publish:
     *     - /fmu/in/vehicle_command: lệnh DO_SET_ACTUATOR release
     *     - /payload_gripper/gripper_grabbed_debug: false
     */
    void releaseGripperForNewMission();

    /**
     * Mô tả:
     *     Báo hoàn thành mode đúng một lần.
     *
     * Input:
     *     Không có.
     *
     * Logic:
     *     - Nếu đã completed rồi thì bỏ qua
     *     - Nếu chưa thì gọi ModeBase::completed(Success)
     *
     * Output:
     *     PX4 nhận kết quả custom mode thành công.
     */
    void completeMissionOnce();

    /**
     * Mô tả:
     *     Callback nhận pose vật thể trong hệ world/NED.
     *
     * Input:
     *     msg: PoseStamped của vật thể. pose.position phải cùng hệ với PX4 local position.
     *
     * Logic:
     *     - Nếu controller chưa active thì bỏ qua
     *     - Nếu có topic object_valid và đang false thì bỏ qua pose
     *     - Lưu vị trí và timestamp mới nhất của vật thể
     *
     * Output:
     *     Cập nhật objectWorld_.
     */
    void objectPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

    /**
     * Mô tả:
     *     Callback nhận trạng thái hợp lệ của object detector/tracker.
     *
     * Input:
     *     msg: Bool, true là đang thấy vật, false là mất vật.
     *
     * Logic:
     *     - Khi false thì xóa pose hiện tại
     *
     * Output:
     *     Cập nhật objectValidExternal_ và objectWorld_.validPose.
     */
    void objectValidCallback(const std_msgs::msg::Bool::SharedPtr msg);

    /**
     * Mô tả:
     *     Giữ UAV tại chỗ bằng velocity setpoint bằng 0.
     *
     * Input:
     *     Không có.
     *
     * Logic:
     *     - Gửi vận tốc XYZ bằng 0
     *
     * Output:
     *     Publish trajectory setpoint giữ vị trí.
     */
    void holdPosition();

    /**
     * Mô tả:
     *     Kiểm tra object pose có bị mất hoặc quá timeout không.
     *
     * Input:
     *     Không có.
     *
     * Logic:
     *     - Nếu chưa có pose thì lost
     *     - Nếu pose cũ hơn object_timeout thì lost
     *
     * Output:
     *     true nếu mất object, false nếu object còn hợp lệ.
     */
    bool checkObjectLost() const;

    /**
     * Mô tả:
     *     Đổi trạng thái của state machine.
     *
     * Input:
     *     state: trạng thái mới.
     *
     * Logic:
     *     - In log transition
     *     - Reset timer liên quan khi vào state mới
     *
     * Output:
     *     Cập nhật state_.
     */
    void switchToState(State state);

    std::string stateName(State state) const;

    /**
     * Mô tả:
     *     Tính vận tốc XY đơn giản để đưa UAV vào tâm vật.
     *
     * Input:
     *     errorXY: sai số ngang world/NED giữa object và UAV
     *     dt_s: chu kỳ cập nhật mode
     *
     * Logic:
     *     - Deadband sai số nhỏ để giảm rung
     *     - P-control: velocity = kp * error
     *     - Giới hạn vận tốc max
     *     - Slew-rate để tránh lệnh vận tốc đổi quá gắt
     *
     * Output:
     *     Vận tốc XY setpoint trong hệ NED/world.
     */
    Eigen::Vector2f computeVelocityXY(const Eigen::Vector2f &errorXY, float dt_s);

    /**
     * Mô tả:
     *     Tính vận tốc Z hạ xuống độ cao gắp theo local Z.
     *
     * Input:
     *     Không có.
     *
     * Logic:
     *     - Nếu abs(local_z) cao hơn grab_target_altitude thì hạ với descend_velocity
     *     - Nếu đã tới hoặc thấp hơn độ cao gắp thì dừng Z
     *
     * Output:
     *     Vận tốc Z setpoint. Theo NED: vz > 0 là đi xuống.
     */
    float computeFixedDescentVelocity() const;

    /**
     * Mô tả:
     *     Lấy độ cao hiện tại của UAV theo trị tuyệt đối local NED z.
     *
     * Input:
     *     Không có.
     *
     * Logic:
     *     - PX4 local position thường có z âm khi UAV bay lên
     *     - Dùng abs(z) để ra độ cao tương đối
     *     - Nếu EKF local z đang dùng/ràng buộc bởi rangefinder thì giá trị này bám theo nguồn range
     *
     * Output:
     *     Độ cao hiện tại của UAV, đơn vị mét.
     */
    float getVehicleAltitude() const;

    /**
     * Mô tả:
     *     Kiểm tra UAV đã hạ tới độ cao gắp cố định chưa.
     *
     * Input:
     *     Không có.
     *
     * Logic:
     *     - So sánh độ cao hiện tại với grab_target_altitude + tolerance
     *
     * Output:
     *     true nếu đã tới vùng độ cao gắp.
     */
    bool isAtGrabAltitude() const;

    /**
     * Mô tả:
     *     Tính vận tốc Z để bay lên lại độ cao đã lưu trước khi hạ gắp.
     *
     * Input:
     *     Không có.
     *
     * Logic:
     *     - Nếu chưa lên đủ cao thì trả về vận tốc âm theo NED để bay lên
     *     - Nếu đã tới độ cao lưu thì dừng Z
     *
     * Output:
     *     Vận tốc Z setpoint. Theo NED: vz < 0 là bay lên.
     */
    float computeClimbVelocityToSavedAltitude() const;

    /**
     * Mô tả:
     *     Kiểm tra UAV đã bay lên lại độ cao đã lưu trước khi hạ chưa.
     *
     * Input:
     *     Không có.
     *
     * Logic:
     *     - Nếu chưa có độ cao lưu thì coi như đã xong để tránh kẹt state
     *     - So sánh độ cao hiện tại với savedReturnAltitude_ - tolerance
     *
     * Output:
     *     true nếu UAV đã lên lại độ cao cần trả về.
     */
    bool isAtSavedReturnAltitude() const;

    /**
     * Mô tả:
     *     Áp giới hạn gia tốc lên lệnh vận tốc.
     *
     * Input:
     *     commandVelocity: lệnh vận tốc mong muốn
     *     previousVelocity: lệnh vận tốc trước đó
     *     accelLimit: giới hạn gia tốc
     *     dt_s: chu kỳ cập nhật
     *
     * Logic:
     *     Giới hạn độ thay đổi vận tốc trong một chu kỳ.
     *
     * Output:
     *     Lệnh vận tốc đã được làm mượt.
     */
    float applySlew(float commandVelocity, float previousVelocity, float accelLimit, float dt_s) const;

    void handleSearchObjectState(bool objectLost);
    void handleApproachObjectState(float dt_s, bool objectLost);

    /**
     * Mô tả:
     *     Hạ UAV xuống độ cao gắp theo local Z. Nếu lệch tâm thì tạm dừng hạ.
     *
     * Input:
     *     dt_s: chu kỳ cập nhật mode.
     *
     * Logic:
     *     - Nếu còn thấy object thì tiếp tục chỉnh XY
     *     - Nếu object vẫn nằm trong vùng tâm thì mới cho hạ Z
     *     - Nếu object lệch tâm hoặc mất object thì vz = 0 để tránh hạ lệch
     *     - Khi tới grab_target_altitude thì chuyển sang GrabReady
     *     - Nếu quá timeout mà chưa tới độ cao thì giữ vị trí, không kích gắp
     *
     * Output:
     *     Publish velocity setpoint XYZ cho PX4.
     */
    void handleDescendToGrabHeightState(float dt_s);

    void handleGrabReadyState(float dt_s);
    void handleClimbToSavedAltitudeState(float dt_s);

private:
    rclcpp::Node &node_;

    std::shared_ptr<px4_ros2::TrajectorySetpointType> trajectorySetpoint_;
    std::shared_ptr<px4_ros2::OdometryLocalPosition> vehicleLocalPosition_;
    std::unique_ptr<ServoGripper> servoGripper_;

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr objectPoseSub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr objectValidSub_;

    std::string objectPoseTopic_;
    std::string objectValidTopic_;

    float paramObjectTimeout_{1.0f};

    float paramXyKp_{0.8f};
    float paramXyDeadband_{0.02f};
    float paramXyMaxVelocity_{1.5f};
    float paramSlewAcc_{0.9f};

    float paramCenterGateRadius_{0.08f};
    float paramCenterStableCommandVelocity_{0.08f};
    float paramApproachSettleTime_{0.80f};

    float paramGrabTargetAltitude_{1.0f};
    float paramGrabAltitudeTolerance_{0.05f};
    float paramDescendVelocity_{0.20f};

    bool paramDescendPauseWhenUnstable_{true};

    float paramDescendTimeout_{25.0f};
    bool paramAllowGrabOnDescendTimeout_{false};

    float paramGrabReadySettleTime_{0.50f};
    float paramClimbVelocity_{0.25f};
    float paramClimbAltitudeTolerance_{0.05f};
    float paramClimbTimeout_{8.0f};

    bool paramReleaseOnDeactivate_{false};
    bool paramForceGrabCommand_{true};

    ObjectWorldState objectWorld_;

    State state_{State::SearchObject};

    bool controllerActive_{false};
    bool objectLostPrev_{true};
    bool objectValidReceived_{false};
    bool objectValidExternal_{true};
    bool grabCommandSent_{false};
    bool completedReported_{false};
    bool hasSavedReturnAltitude_{false};

    float approachSettledTime_{0.0f};
    float descendTime_{0.0f};
    float grabReadySettledTime_{0.0f};
    float climbTime_{0.0f};
    float savedReturnAltitude_{0.0f};

    float vxFiltered_{0.0f};
    float vyFiltered_{0.0f};
};