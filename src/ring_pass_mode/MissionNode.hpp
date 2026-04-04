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
	enum class MissionState
	{
		WAIT_TAKEOFF_POINT = 0,
		INIT_MISSION_POINTS,
		MOVE_TO_P1,
		RING_PASS_FORWARD,
		MOVE_TO_P2_AND_CLIMB,
		PICK_OBJECT_WAIT,
		MOVE_TO_P3,
		ALIGN_YAW_BACKWARD,
		RING_PASS_BACKWARD,
		MOVE_TO_P4_AND_CLIMB,
		DROP_OBJECT_WAIT,
		RETURN_TO_P1,
		ALIGN_YAW_FORWARD_AGAIN,
		RETURN_TO_LAND_WAIT,
		FINISHED
	};

	struct MissionPoint
	{
		Eigen::Vector3f pos{0.0f, 0.0f, 0.0f};
		bool valid{false};
	};

private:
	void loadParameters();
	void setupInterfaces();

	void vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg);

	void enterState(MissionState new_state);
	void initMissionPoints();

	bool isNearPoint(
		const Eigen::Vector3f& target,
		float xy_tol,
		float z_tol) const;

	Eigen::Vector3f computeGoToVelocity(
		const Eigen::Vector3f& target_pos,
		float max_xy,
		float max_z) const;

	float computeYawError(
		float target_yaw_rad,
		float current_yaw_rad) const;

	bool alignYawDone(float target_yaw_rad) const;

	bool waitStateDone(
		double wait_s,
		const std::string& message);

	const char* missionStateToString(MissionState state) const;

private:
	rclcpp::Node& _node;
	RingPassController ring_pass_controller_;

	std::shared_ptr<px4_ros2::TrajectorySetpointType> trajectory_setpoint_;

	rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr vehicle_local_position_sub_;

	rclcpp::Time state_enter_time_{0, 0, RCL_ROS_TIME};

	Eigen::Vector3f current_position_{0.0f, 0.0f, 0.0f};
	float current_yaw_{0.0f};
	bool has_vehicle_local_position_{false};

	Eigen::Vector3f takeoff_position_{0.0f, 0.0f, 0.0f};
	bool takeoff_position_saved_{false};

	MissionPoint point1_{};
	MissionPoint point2_{};
	MissionPoint point3_{};
	MissionPoint point4_{};

	MissionState state_{MissionState::WAIT_TAKEOFF_POINT};
	bool state_first_tick_{true};
    float missionTargetZ() const;

	int cycle_count_{0};
	int max_cycle_{3};

	float param_mission_height_{3.0f};

	float param_wp_kp_xy_{0.8f};
	float param_wp_kp_z_{0.8f};
	float param_wp_max_xy_{1.0f};
	float param_wp_max_z_{0.6f};

	float param_xy_tolerance_{0.25f};
	float param_z_tolerance_{0.20f};
	float param_yaw_tolerance_deg_{8.0f};

	float param_p1_x_{0.0f};
	float param_p1_y_{3.0f};

	float param_p2_x_{8.0f};
	float param_p2_y_{3.0f};

	float param_p3_x_{8.0f};
	float param_p3_y_{0.0f};

	float param_p4_x_{0.0f};
	float param_p4_y_{0.0f};
};