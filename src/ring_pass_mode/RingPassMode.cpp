#include "RingPassMode.hpp"

#include <algorithm>
#include <cmath>

#include <px4_ros2/components/node_with_mode.hpp>

namespace
{
const std::string kModeName = "RingPassMode";
constexpr bool kEnableDebugOutput = true;

constexpr char kVehicleAttitudeTopic[] = "/fmu/out/vehicle_attitude";
constexpr char kTargetErrorFusionTopic[] = "/ring_detect/target_error_body_filtered";
} // namespace

RingPassMode::RingPassMode(rclcpp::Node& node)
	: ModeBase(node, kModeName),
	  _node(node)
{
	_trajectory_setpoint = std::make_shared<px4_ros2::TrajectorySetpointType>(*this);

	_vehicle_attitude_sub =
		_node.create_subscription<px4_msgs::msg::VehicleAttitude>(
			kVehicleAttitudeTopic,
			rclcpp::QoS(10).best_effort(),
			std::bind(&RingPassMode::vehicleAttitudeCallback, this, std::placeholders::_1));

	_target_error_sub =
		_node.create_subscription<geometry_msgs::msg::PoseStamped>(
			kTargetErrorFusionTopic,
			rclcpp::QoS(1).best_effort(),
			std::bind(&RingPassMode::targetErrorCallback, this, std::placeholders::_1));

	loadParameters();

	modeRequirements().manual_control = false;

	_debug_timer = _node.create_wall_timer(
		std::chrono::milliseconds(500),
		[this]()
		{
			if (!attitudeReady()) {
				RCLCPP_WARN_THROTTLE(
					_node.get_logger(),
					*(_node.get_clock()),
					2000,
					"[RingPassMode] waiting for vehicle attitude");
				return;
			}

			RCLCPP_INFO_THROTTLE(
				_node.get_logger(),
				*(_node.get_clock()),
				1000,
				"[RingPassMode] target_valid=%d err=(%.3f, %.3f, %.3f) h_offset=%.3f yaw=%.3f",
				_target.valid ? 1 : 0,
				_target.value.x(),
				_target.value.y(),
				_target.value.z(),
				_param_height_offset,
				getVehicleYaw());
		});
}
void RingPassMode::loadParameters()
{
	// =========================
	// 1) Timeout dữ liệu target
	// =========================

	// Nếu quá thời gian này mà không nhận được target mới thì coi như mất mục tiêu.
	// Mode sẽ không tiếp tục lao qua vòng mà chuyển sang giữ vận tốc 0.
	// Tăng lên:
	// - nếu detector / kalman publish hơi chậm
	// - nếu đôi lúc mất target ngắn nhưng bạn vẫn muốn bay liên tục
	// Giảm xuống:
	// - nếu muốn mode phản ứng nhanh khi mất mục tiêu
	_node.declare_parameter<float>("target_timeout", 0.3f);

	// =========================
	// 2) Gain điều khiển P
	// =========================

	// Gain sửa lệch ngang trái/phải.
	// Sai số lateral càng lớn thì lệnh vy càng lớn.
	// Tăng lên:
	// - drone kéo ngang nhanh hơn để căn tâm vòng
	// - phù hợp khi thấy drone sửa ngang quá chậm
	// Giảm xuống:
	// - nếu drone lắc ngang, overshoot, qua lại trái phải nhiều
	_node.declare_parameter<float>("ringpass_kp_lat", 0.8f);

	// Gain sửa độ cao.
	// Trong frame hiện tại z là "down", nên gain này quyết định độ mạnh khi chỉnh lên/xuống.
	// Tăng lên:
	// - căn cao độ nhanh hơn
	// Giảm xuống:
	// - nếu drone nhấp nhô, rung theo trục z
	_node.declare_parameter<float>("ringpass_kp_z", 0.8f);

	// =========================
	// 3) Deadband
	// =========================

	// Vùng chết trục ngang.
	// Nếu sai số ngang nhỏ hơn ngưỡng này thì xem như đã đủ đúng, không cần sửa nữa.
	// Tăng lên:
	// - giảm rung lắc nhỏ quanh tâm
	// - bay mượt hơn
	// Giảm xuống:
	// - muốn căn chính xác hơn
	// - nhưng có thể rung nhiều hơn gần tâm
	_node.declare_parameter<float>("ringpass_lat_deadband", 0.05f);

	// Vùng chết trục z.
	// Nếu sai số độ cao nhỏ hơn ngưỡng này thì không chỉnh nữa.
	// Tăng lên:
	// - đỡ rung lên xuống gần tâm vòng
	// Giảm xuống:
	// - muốn bám đúng độ cao hơn
	_node.declare_parameter<float>("ringpass_z_deadband", 0.05f);

	// =========================
	// 4) Giới hạn vận tốc từng trục
	// =========================

	// Tốc độ tiến tối đa theo trục forward.
	// Đây là thông số ảnh hưởng mạnh nhất đến việc drone "lao qua vòng" nhanh hay chậm.
	// Tăng lên:
	// - drone bay tới vòng nhanh hơn
	// Giảm xuống:
	// - an toàn hơn, dễ tune hơn
	// - phù hợp khi perception hoặc control chưa ổn định
	_node.declare_parameter<float>("ringpass_v_forward_max", 1.0f);

	// Tốc độ sửa ngang tối đa.
	// Dù kp_lat lớn, vận tốc ngang vẫn không vượt quá ngưỡng này.
	// Tăng lên:
	// - căn trái/phải nhanh hơn
	// Giảm xuống:
	// - tránh giật ngang mạnh
	_node.declare_parameter<float>("ringpass_v_lateral_max", 1.0f);

	// Tốc độ sửa cao độ tối đa.
	// Dù kp_z lớn, vận tốc z vẫn không vượt quá ngưỡng này.
	// Tăng lên:
	// - căn độ cao nhanh hơn
	// Giảm xuống:
	// - tránh chúi / ngóc mạnh, tránh nhấp nhô
	_node.declare_parameter<float>("ringpass_v_vertical_max", 0.5f);

	// =========================
	// 5) Điều kiện cho phép tiến tới trước
	// =========================

	// Khi sai số tâm (kết hợp lệch ngang + lệch cao độ) lớn hơn ngưỡng này
	// thì chưa cho lao tới trước hoặc cho rất ít.
	// Tăng lên:
	// - drone bắt đầu tiến sớm hơn dù chưa căn rất chuẩn
	// - giúp bay nhanh hơn
	// Giảm xuống:
	// - drone sẽ ưu tiên căn tâm kỹ hơn rồi mới lao
	// - an toàn hơn nhưng chậm hơn
	_node.declare_parameter<float>("ringpass_center_error_enter_forward", 0.35f);

	// Khi sai số tâm nhỏ hơn ngưỡng này thì forward đạt gần mức tối đa.
	// Tăng lên:
	// - forward mạnh sớm hơn
	// - cảm giác bay quyết đoán hơn
	// Giảm xuống:
	// - chỉ khi rất chuẩn tâm mới lao tối đa
	_node.declare_parameter<float>("ringpass_center_error_full_forward", 0.10f);

	// =========================
	// 6) Slew rate - làm mượt lệnh
	// =========================

	// Giới hạn tốc độ thay đổi lệnh XY.
	// Không phải giới hạn vận tốc, mà là giới hạn "độ tăng giảm" của lệnh.
	// Tăng lên:
	// - phản ứng nhanh hơn, bớt ì
	// Giảm xuống:
	// - mượt hơn nhưng chậm hơn
	// - nếu quá thấp thì drone có cảm giác lười, không chịu tăng tốc
	_node.declare_parameter<float>("ringpass_slew_xy", 1.5f);

	// Giới hạn tốc độ thay đổi lệnh z.
	// Tăng lên:
	// - chỉnh lên/xuống nhanh hơn
	// Giảm xuống:
	// - mượt hơn theo phương đứng
	_node.declare_parameter<float>("ringpass_slew_z", 0.8f);

	// =========================
	// 7) Offset độ cao bay qua vòng
	// =========================

	// Offset theo trục down.
	// err_down = target_down - height_offset
	//
	// Nếu height_offset > 0:
	// - drone sẽ cố bay CAO HƠN tâm vòng
	// - hữu ích khi bên dưới có bộ gắp
	//
	// Nếu height_offset < 0:
	// - drone sẽ cố bay THẤP HƠN tâm vòng
	//
	// Đây là param hình học/chính sách bay, không phải gain.
	_node.declare_parameter<float>("ringpass_height_offset", 0.0f);

	// =========================
	// 8) Read back
	// =========================

	_node.get_parameter("target_timeout", _param_target_timeout);

	_node.get_parameter("ringpass_kp_lat", _param_kp_lat);
	_node.get_parameter("ringpass_kp_z", _param_kp_z);

	_node.get_parameter("ringpass_lat_deadband", _param_lat_deadband);
	_node.get_parameter("ringpass_z_deadband", _param_z_deadband);

	_node.get_parameter("ringpass_v_forward_max", _param_v_forward_max);
	_node.get_parameter("ringpass_v_lateral_max", _param_v_lateral_max);
	_node.get_parameter("ringpass_v_vertical_max", _param_v_vertical_max);

	_node.get_parameter("ringpass_center_error_enter_forward", _param_center_error_enter_forward);
	_node.get_parameter("ringpass_center_error_full_forward", _param_center_error_full_forward);

	_node.get_parameter("ringpass_slew_xy", _param_slew_xy);
	_node.get_parameter("ringpass_slew_z", _param_slew_z);

	_node.get_parameter("ringpass_height_offset", _param_height_offset);
}

void RingPassMode::resetController()
{
	_vx_last = 0.0f;
	_vy_last = 0.0f;
	_vz_last = 0.0f;

	_yaw_sp_init = false;
	_yaw_sp_fixed = 0.0f;
}

void RingPassMode::vehicleAttitudeCallback(
	const px4_msgs::msg::VehicleAttitude::SharedPtr msg)
{
	_vehicle_attitude_msg = *msg;
	_has_vehicle_attitude = true;
}

void RingPassMode::targetErrorCallback(
	const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
	_target.value.x() = static_cast<float>(msg->pose.position.x);
	_target.value.y() = static_cast<float>(msg->pose.position.y);
	_target.value.z() = static_cast<float>(msg->pose.position.z);

	_target.stamp = (msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0)
		? _node.now()
		: rclcpp::Time(msg->header.stamp);

	_target.valid = true;
}

bool RingPassMode::attitudeReady() const
{
	return _has_vehicle_attitude;
}

bool RingPassMode::targetTimedOut() const
{
	if (!_target.valid) {
		return true;
	}

	const double age_s = (_node.now() - _target.stamp).seconds();
	return age_s > static_cast<double>(_param_target_timeout);
}

float RingPassMode::getVehicleYaw() const
{
	const auto& q = _vehicle_attitude_msg.q;

	const float w = q[0];
	const float x = q[1];
	const float y = q[2];
	const float z = q[3];

	const float siny_cosp = 2.0f * (w * z + x * y);
	const float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);

	return std::atan2(siny_cosp, cosy_cosp);
}

float RingPassMode::computeForwardAlpha(float center_error) const
{
	if (center_error >= _param_center_error_enter_forward) {
		return 0.0f;
	}

	if (center_error <= _param_center_error_full_forward) {
		return 1.0f;
	}

	const float denom =
		std::max(_param_center_error_enter_forward - _param_center_error_full_forward, 1e-3f);

	float alpha = (_param_center_error_enter_forward - center_error) / denom;
	alpha = std::clamp(alpha, 0.0f, 1.0f);

	return alpha * alpha;
}

float RingPassMode::applySlew(float cmd, float prev, float accel_limit, float dt_s) const
{
	const float dt = std::max(dt_s, 1e-3f);
	const float max_delta = accel_limit * dt;
	const float delta = std::clamp(cmd - prev, -max_delta, max_delta);
	return prev + delta;
}

Eigen::Vector2f RingPassMode::bodyToWorldXY(const Eigen::Vector2f& body_xy, float yaw) const
{
	const float c = std::cos(yaw);
	const float s = std::sin(yaw);

	Eigen::Vector2f world_xy;
	world_xy.x() = c * body_xy.x() - s * body_xy.y();
	world_xy.y() = s * body_xy.x() + c * body_xy.y();
	return world_xy;
}

Eigen::Vector3f RingPassMode::computeVelocityCommand(float dt_s)
{
	const float err_forward = _target.value.x();
	const float err_lateral = _target.value.y();
	const float err_down = _target.value.z() - _param_height_offset;

	const float center_error = std::sqrt(err_lateral * err_lateral + err_down * err_down);

	float vx_body_cmd = computeForwardAlpha(center_error) * _param_v_forward_max;
	if (err_forward < 0.0f) {
		vx_body_cmd = 0.0f;
	}

	float vy_body_cmd = _param_kp_lat * err_lateral;
	if (std::abs(err_lateral) < _param_lat_deadband) {
		vy_body_cmd = 0.0f;
	}
	vy_body_cmd = std::clamp(vy_body_cmd, -_param_v_lateral_max, _param_v_lateral_max);

	float vz_cmd = _param_kp_z * err_down;
	if (std::abs(err_down) < _param_z_deadband) {
		vz_cmd = 0.0f;
	}
	vz_cmd = std::clamp(vz_cmd, -_param_v_vertical_max, _param_v_vertical_max);

	const float yaw_now = getVehicleYaw();
	const Eigen::Vector2f world_xy =
		bodyToWorldXY(Eigen::Vector2f(vx_body_cmd, vy_body_cmd), yaw_now);

	_vx_last = applySlew(world_xy.x(), _vx_last, _param_slew_xy, dt_s);
	_vy_last = applySlew(world_xy.y(), _vy_last, _param_slew_xy, dt_s);
	_vz_last = applySlew(vz_cmd, _vz_last, _param_slew_z, dt_s);

	return Eigen::Vector3f(_vx_last, _vy_last, _vz_last);
}

void RingPassMode::publishHold(float dt_s)
{
	_vx_last = applySlew(0.0f, _vx_last, _param_slew_xy, dt_s);
	_vy_last = applySlew(0.0f, _vy_last, _param_slew_xy, dt_s);
	_vz_last = applySlew(0.0f, _vz_last, _param_slew_z, dt_s);

	_trajectory_setpoint->update(
		Eigen::Vector3f(_vx_last, _vy_last, _vz_last),
		std::nullopt,
		std::nullopt);
}

void RingPassMode::onActivate()
{
	resetController();

	if (attitudeReady()) {
		_yaw_sp_fixed = getVehicleYaw();
		_yaw_sp_init = true;
	}
}

void RingPassMode::onDeactivate()
{
	RCLCPP_INFO(_node.get_logger(), "[RingPassMode] deactivated");
}

void RingPassMode::updateSetpoint(float dt_s)
{
	if (!attitudeReady()) {
		publishHold(dt_s);
		return;
	}

	if (!_yaw_sp_init) {
		_yaw_sp_fixed = getVehicleYaw();
		_yaw_sp_init = true;
	}

	if (targetTimedOut()) {
		publishHold(dt_s);
		return;
	}

	const Eigen::Vector3f vel_sp = computeVelocityCommand(dt_s);

	_trajectory_setpoint->update(
		vel_sp,
		std::nullopt,
		std::nullopt);
}

int main(int argc, char* argv[])
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<px4_ros2::NodeWithMode<RingPassMode>>(kModeName, kEnableDebugOutput));
	rclcpp::shutdown();
	return 0;
}