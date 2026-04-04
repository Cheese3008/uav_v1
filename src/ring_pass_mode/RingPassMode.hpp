#pragma once

#include <memory>
#include <optional>

#include <Eigen/Core>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <px4_msgs/msg/vehicle_attitude.hpp>
#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/control/setpoint_types/experimental/trajectory.hpp>
#include <rclcpp/rclcpp.hpp>

class RingPassMode : public px4_ros2::ModeBase
{
public:
	explicit RingPassMode(rclcpp::Node& node);

	void onActivate() override;
	void onDeactivate() override;
	void updateSetpoint(float dt_s) override;

private:
	struct TargetError
	{
		Eigen::Vector3f value{0.0f, 0.0f, 0.0f}; // body_level_frd: x=fwd, y=right, z=down
		rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
		bool valid{false};
	};

	void loadParameters();
	void resetController();

	void vehicleAttitudeCallback(const px4_msgs::msg::VehicleAttitude::SharedPtr msg);
	void targetErrorCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

	bool attitudeReady() const;
	bool targetTimedOut() const;

	float getVehicleYaw() const;
	float computeForwardAlpha(float center_error) const;
	float applySlew(float cmd, float prev, float accel_limit, float dt_s) const;

	Eigen::Vector2f bodyToWorldXY(const Eigen::Vector2f& body_xy, float yaw) const;
	Eigen::Vector3f computeVelocityCommand(float dt_s);
	void publishHold(float dt_s);

private:
	rclcpp::Node& _node;

	std::shared_ptr<px4_ros2::TrajectorySetpointType> _trajectory_setpoint;

	rclcpp::Subscription<px4_msgs::msg::VehicleAttitude>::SharedPtr _vehicle_attitude_sub;
	rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr _target_error_sub;
	rclcpp::TimerBase::SharedPtr _debug_timer;

	px4_msgs::msg::VehicleAttitude _vehicle_attitude_msg{};
	bool _has_vehicle_attitude{false};

	TargetError _target{};

	float _param_target_timeout{0.3f};

	float _param_kp_lat{0.8f};
	float _param_kp_z{0.8f};

	float _param_lat_deadband{0.05f};
	float _param_z_deadband{0.05f};

	float _param_v_forward_max{2.0f};
	float _param_v_lateral_max{1.0f};
	float _param_v_vertical_max{0.5f};

	float _param_center_error_enter_forward{0.35f};
	float _param_center_error_full_forward{0.10f};

	float _param_slew_xy{1.5f};
	float _param_slew_z{0.8f};

	float _param_height_offset{0.0f};

	float _vx_last{0.0f};
	float _vy_last{0.0f};
	float _vz_last{0.0f};

	bool _yaw_sp_init{false};
	float _yaw_sp_fixed{0.0f};
};