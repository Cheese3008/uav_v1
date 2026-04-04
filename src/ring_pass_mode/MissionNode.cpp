#include "MissionNode.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include <px4_ros2/components/node_with_mode.hpp>

namespace
{
constexpr char kVehicleLocalPositionTopic[] = "/fmu/out/vehicle_local_position";

const std::string kModeName = "MissionRingNode";
constexpr bool kEnableDebugOutput = true;

float wrapPi(float angle)
{
	while (angle > static_cast<float>(M_PI))
	{
		angle -= 2.0f * static_cast<float>(M_PI);
	}

	while (angle < -static_cast<float>(M_PI))
	{
		angle += 2.0f * static_cast<float>(M_PI);
	}

	return angle;
}
}

MissionNode::MissionNode(rclcpp::Node& node)
	: ModeBase(node, kModeName)
	, _node(node)
	, ring_pass_controller_(node)
{
	loadParameters();
	setupInterfaces();

	trajectory_setpoint_ =
		std::make_shared<px4_ros2::TrajectorySetpointType>(*this);

	modeRequirements().manual_control = false;

	state_enter_time_ = _node.now();

	RCLCPP_INFO(_node.get_logger(), "[MissionNode] external mode created");
}

void MissionNode::loadParameters()
{
	_node.declare_parameter<float>("mission_height", 3.0f);

	_node.declare_parameter<float>("mission_xy_tolerance", 0.25f);
	_node.declare_parameter<float>("mission_z_tolerance", 0.20f);
	_node.declare_parameter<float>("mission_yaw_tolerance_deg", 8.0f);

	_node.declare_parameter<int>("mission_cycle_count", 1);

    _node.declare_parameter<float>("mission_p1_x", -3.5f);
    _node.declare_parameter<float>("mission_p1_y", 2.5f);

    _node.declare_parameter<float>("mission_p2_x", -3.5f);
    _node.declare_parameter<float>("mission_p2_y", 18.5f);

    _node.declare_parameter<float>("mission_p3_x", 3.5f);
    _node.declare_parameter<float>("mission_p3_y", 18.5f);

    _node.declare_parameter<float>("mission_p4_x", 3.5f);
    _node.declare_parameter<float>("mission_p4_y", 2.5f);

	_node.get_parameter("mission_height", param_mission_height_);

	_node.get_parameter("mission_xy_tolerance", param_xy_tolerance_);
	_node.get_parameter("mission_z_tolerance", param_z_tolerance_);
	_node.get_parameter("mission_yaw_tolerance_deg", param_yaw_tolerance_deg_);

	_node.get_parameter("mission_cycle_count", max_cycle_);

	_node.get_parameter("mission_p1_x", param_p1_x_);
	_node.get_parameter("mission_p1_y", param_p1_y_);

	_node.get_parameter("mission_p2_x", param_p2_x_);
	_node.get_parameter("mission_p2_y", param_p2_y_);

	_node.get_parameter("mission_p3_x", param_p3_x_);
	_node.get_parameter("mission_p3_y", param_p3_y_);

	_node.get_parameter("mission_p4_x", param_p4_x_);
	_node.get_parameter("mission_p4_y", param_p4_y_);

	max_cycle_ = std::max(max_cycle_, 1);

	RCLCPP_INFO(
		_node.get_logger(),
		"[MissionNode] params: mission_height=%.2f P1(%.2f %.2f) P2(%.2f %.2f) P3(%.2f %.2f) P4(%.2f %.2f)",
		param_mission_height_,
		param_p1_x_, param_p1_y_,
		param_p2_x_, param_p2_y_,
		param_p3_x_, param_p3_y_,
		param_p4_x_, param_p4_y_);
}

void MissionNode::setupInterfaces()
{
	vehicle_local_position_sub_ =
		_node.create_subscription<px4_msgs::msg::VehicleLocalPosition>(
			kVehicleLocalPositionTopic,
			rclcpp::QoS(10).best_effort(),
			std::bind(&MissionNode::vehicleLocalPositionCallback, this, std::placeholders::_1));
}

void MissionNode::vehicleLocalPositionCallback(
	const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg)
{
	current_position_.x() = msg->x;
	current_position_.y() = msg->y;
	current_position_.z() = msg->z;
	current_yaw_ = msg->heading;

	has_vehicle_local_position_ = true;
}

void MissionNode::onActivate()
{
	cycle_count_ = 0;
	takeoff_position_saved_ = false;
	state_first_tick_ = true;
	state_enter_time_ = _node.now();
	state_ = MissionState::WAIT_TAKEOFF_POINT;

	ring_pass_controller_.reset();

	RCLCPP_INFO(_node.get_logger(), "[MissionNode] activated");
}

void MissionNode::onDeactivate()
{
	ring_pass_controller_.reset();
	RCLCPP_INFO(_node.get_logger(), "[MissionNode] deactivated");
}

void MissionNode::enterState(MissionState new_state)
{
	state_ = new_state;
	state_enter_time_ = _node.now();
	state_first_tick_ = true;

	RCLCPP_INFO(
		_node.get_logger(),
		"[MissionNode] enter state: %s",
		missionStateToString(state_));
}

void MissionNode::initMissionPoints()
{
	// DUNG TOA DO WORLD TUYET DOI, KHONG CONG VOI TAKEOFF_POSITION_
	point1_.pos = Eigen::Vector3f(param_p1_x_, param_p1_y_, 0.0f);
	point2_.pos = Eigen::Vector3f(param_p2_x_, param_p2_y_, 0.0f);
	point3_.pos = Eigen::Vector3f(param_p3_x_, param_p3_y_, 0.0f);
	point4_.pos = Eigen::Vector3f(param_p4_x_, param_p4_y_, 0.0f);

	point1_.valid = true;
	point2_.valid = true;
	point3_.valid = true;
	point4_.valid = true;

	RCLCPP_INFO(
		_node.get_logger(),
		"[MissionNode] world points: P1(%.2f %.2f) P2(%.2f %.2f) P3(%.2f %.2f) P4(%.2f %.2f) START(%.2f %.2f)",
		point1_.pos.x(), point1_.pos.y(),
		point2_.pos.x(), point2_.pos.y(),
		point3_.pos.x(), point3_.pos.y(),
		point4_.pos.x(), point4_.pos.y(),
		takeoff_position_.x(), takeoff_position_.y());
}

bool MissionNode::isNearPoint(
	const Eigen::Vector3f& target,
	float xy_tol,
	float z_tol) const
{
	const float dx = target.x() - current_position_.x();
	const float dy = target.y() - current_position_.y();
	const float dz = target.z() - current_position_.z();

	const float err_xy = std::sqrt(dx * dx + dy * dy);

	return (err_xy <= xy_tol) && (std::abs(dz) <= z_tol);
}

Eigen::Vector3f MissionNode::computeGoToVelocity(
	const Eigen::Vector3f& target_pos,
	float max_xy,
	float max_z) const
{
	(void)target_pos;
	(void)max_xy;
	(void)max_z;
	return Eigen::Vector3f::Zero();
}

float MissionNode::missionTargetZ() const
{
	return -param_mission_height_;
}

float MissionNode::computeYawError(
	float target_yaw_rad,
	float current_yaw_rad) const
{
	return wrapPi(target_yaw_rad - current_yaw_rad);
}

bool MissionNode::alignYawDone(float target_yaw_rad) const
{
	const float yaw_tol_rad =
		param_yaw_tolerance_deg_ * static_cast<float>(M_PI) / 180.0f;

	const float yaw_error = computeYawError(target_yaw_rad, current_yaw_);

	return std::abs(yaw_error) <= yaw_tol_rad;
}

bool MissionNode::waitStateDone(
	double wait_s,
	const std::string& message)
{
	if (state_first_tick_)
	{
		RCLCPP_WARN(_node.get_logger(), "%s", message.c_str());
		state_first_tick_ = false;
	}

	return (_node.now() - state_enter_time_).seconds() >= wait_s;
}

const char* MissionNode::missionStateToString(MissionState state) const
{
	switch (state)
	{
	case MissionState::WAIT_TAKEOFF_POINT:
		return "WAIT_TAKEOFF_POINT";
	case MissionState::INIT_MISSION_POINTS:
		return "INIT_MISSION_POINTS";
	case MissionState::MOVE_TO_P1:
		return "MOVE_TO_P1";
	case MissionState::RING_PASS_FORWARD:
		return "RING_PASS_FORWARD";
	case MissionState::MOVE_TO_P2_AND_CLIMB:
		return "MOVE_TO_P2";
	case MissionState::PICK_OBJECT_WAIT:
		return "PICK_OBJECT_WAIT";
	case MissionState::MOVE_TO_P3:
		return "MOVE_TO_P3";
	case MissionState::ALIGN_YAW_BACKWARD:
		return "ALIGN_YAW_BACKWARD";
	case MissionState::RING_PASS_BACKWARD:
		return "RING_PASS_BACKWARD";
	case MissionState::MOVE_TO_P4_AND_CLIMB:
		return "MOVE_TO_P4";
	case MissionState::DROP_OBJECT_WAIT:
		return "DROP_OBJECT_WAIT";
	case MissionState::RETURN_TO_P1:
		return "RETURN_TO_START";
	case MissionState::ALIGN_YAW_FORWARD_AGAIN:
		return "ALIGN_YAW_FORWARD_AGAIN";
	case MissionState::RETURN_TO_LAND_WAIT:
		return "RETURN_TO_LAND_WAIT";
	case MissionState::FINISHED:
		return "FINISHED";
	default:
		return "UNKNOWN";
	}
}

void MissionNode::updateSetpoint(float dt_s)
{
	(void)dt_s;

	if (!has_vehicle_local_position_)
	{
		RCLCPP_WARN_THROTTLE(
			_node.get_logger(),
			*(_node.get_clock()),
			1000,
			"[MissionNode] waiting for /fmu/out/vehicle_local_position");

		trajectory_setpoint_->update(
			Eigen::Vector3f::Zero(),
			std::nullopt,
			std::nullopt,
			std::nullopt);
		return;
	}

	switch (state_)
	{
	case MissionState::WAIT_TAKEOFF_POINT:
	{
		if (!takeoff_position_saved_)
		{
			takeoff_position_ = current_position_;
			takeoff_position_saved_ = true;

			RCLCPP_INFO(
				_node.get_logger(),
				"[MissionNode] saved start point: (%.2f, %.2f, %.2f)",
				takeoff_position_.x(),
				takeoff_position_.y(),
				takeoff_position_.z());

			enterState(MissionState::INIT_MISSION_POINTS);
		}

		trajectory_setpoint_->updatePosition(current_position_);
		break;
	}

	case MissionState::INIT_MISSION_POINTS:
	{
		initMissionPoints();
		enterState(MissionState::MOVE_TO_P1);
		trajectory_setpoint_->updatePosition(current_position_);
		break;
	}

	case MissionState::MOVE_TO_P1:
	{
		Eigen::Vector3f target = point1_.pos;
		target.z() = missionTargetZ();

		trajectory_setpoint_->updatePosition(target);

		RCLCPP_INFO_THROTTLE(
			_node.get_logger(),
			*(_node.get_clock()),
			500,
			"[MissionNode] MOVE_TO_P1 current=(%.2f %.2f %.2f) target=(%.2f %.2f %.2f)",
			current_position_.x(), current_position_.y(), current_position_.z(),
			target.x(), target.y(), target.z());

		if (isNearPoint(target, param_xy_tolerance_, param_z_tolerance_))
		{
			RCLCPP_INFO(_node.get_logger(), "[MissionNode] reached P1");
			enterState(MissionState::MOVE_TO_P2_AND_CLIMB);
		}
		break;
	}

	case MissionState::MOVE_TO_P2_AND_CLIMB:
	{
		Eigen::Vector3f target = point2_.pos;
		target.z() = missionTargetZ();

		trajectory_setpoint_->updatePosition(target);

		RCLCPP_INFO_THROTTLE(
			_node.get_logger(),
			*(_node.get_clock()),
			500,
			"[MissionNode] MOVE_TO_P2 current=(%.2f %.2f %.2f) target=(%.2f %.2f %.2f)",
			current_position_.x(), current_position_.y(), current_position_.z(),
			target.x(), target.y(), target.z());

		if (isNearPoint(target, param_xy_tolerance_, param_z_tolerance_))
		{
			RCLCPP_INFO(_node.get_logger(), "[MissionNode] reached P2");
			enterState(MissionState::MOVE_TO_P3);
		}
		break;
	}

	case MissionState::MOVE_TO_P3:
	{
		Eigen::Vector3f target = point3_.pos;
		target.z() = missionTargetZ();

		trajectory_setpoint_->updatePosition(target);

		RCLCPP_INFO_THROTTLE(
			_node.get_logger(),
			*(_node.get_clock()),
			500,
			"[MissionNode] MOVE_TO_P3 current=(%.2f %.2f %.2f) target=(%.2f %.2f %.2f)",
			current_position_.x(), current_position_.y(), current_position_.z(),
			target.x(), target.y(), target.z());

		if (isNearPoint(target, param_xy_tolerance_, param_z_tolerance_))
		{
			RCLCPP_INFO(_node.get_logger(), "[MissionNode] reached P3");
			enterState(MissionState::MOVE_TO_P4_AND_CLIMB);
		}
		break;
	}

	case MissionState::MOVE_TO_P4_AND_CLIMB:
	{
		Eigen::Vector3f target = point4_.pos;
		target.z() = missionTargetZ();

		trajectory_setpoint_->updatePosition(target);

		RCLCPP_INFO_THROTTLE(
			_node.get_logger(),
			*(_node.get_clock()),
			500,
			"[MissionNode] MOVE_TO_P4 current=(%.2f %.2f %.2f) target=(%.2f %.2f %.2f)",
			current_position_.x(), current_position_.y(), current_position_.z(),
			target.x(), target.y(), target.z());

		if (isNearPoint(target, param_xy_tolerance_, param_z_tolerance_))
		{
			RCLCPP_INFO(_node.get_logger(), "[MissionNode] reached P4");
			enterState(MissionState::RETURN_TO_P1);
		}
		break;
	}

	case MissionState::RETURN_TO_P1:
	{
		Eigen::Vector3f target = takeoff_position_;
		target.z() = missionTargetZ();

		trajectory_setpoint_->updatePosition(target);

		RCLCPP_INFO_THROTTLE(
			_node.get_logger(),
			*(_node.get_clock()),
			500,
			"[MissionNode] RETURN_TO_START current=(%.2f %.2f %.2f) target=(%.2f %.2f %.2f)",
			current_position_.x(), current_position_.y(), current_position_.z(),
			target.x(), target.y(), target.z());

		if (isNearPoint(target, param_xy_tolerance_, param_z_tolerance_))
		{
			RCLCPP_INFO(_node.get_logger(), "[MissionNode] reached START");
			enterState(MissionState::FINISHED);
		}
		break;
	}

	case MissionState::RING_PASS_FORWARD:
	case MissionState::PICK_OBJECT_WAIT:
	case MissionState::ALIGN_YAW_BACKWARD:
	case MissionState::RING_PASS_BACKWARD:
	case MissionState::DROP_OBJECT_WAIT:
	case MissionState::ALIGN_YAW_FORWARD_AGAIN:
	case MissionState::RETURN_TO_LAND_WAIT:
	{
		trajectory_setpoint_->updatePosition(current_position_);
		break;
	}

	case MissionState::FINISHED:
	default:
	{
		trajectory_setpoint_->updatePosition(current_position_);
		break;
	}
	}

	if (state_ == MissionState::FINISHED)
	{
		completed(px4_ros2::Result::Success);
	}
}

int main(int argc, char* argv[])
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<px4_ros2::NodeWithMode<MissionNode>>(kModeName, kEnableDebugOutput));
	rclcpp::shutdown();
	return 0;
}