from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )

    enable_camera_arg = DeclareLaunchArgument(
        'enable_camera',
        default_value='false',
        description='Enable UDP H264 camera node'
    )

    camera_udp_port_arg = DeclareLaunchArgument(
        'camera_udp_port',
        default_value='5600',
        description='UDP port for H264 camera stream'
    )

    camera_image_topic_arg = DeclareLaunchArgument(
        'camera_image_topic',
        default_value='/camera_down/image_raw',
        description='Downward camera image topic'
    )

    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/camera_down/camera_info',
        description='Downward camera info topic'
    )

    camera_frame_id_arg = DeclareLaunchArgument(
        'camera_frame_id',
        default_value='camera_down_link',
        description='Downward camera frame id'
    )

    point_param_file_arg = DeclareLaunchArgument(
        'point_param_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('point_mission_mode'),
            'cfg',
            'params.yaml'
        ]),
        description='Parameter file for point_mission_mode'
    )

    camera_node = Node(
        package='udp_h264_camera',
        executable='udp_h264_camera_node',
        name='udp_h264_camera_down',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_camera')),
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'image_topic': LaunchConfiguration('camera_image_topic')},
            {'camera_info_topic': LaunchConfiguration('camera_info_topic')},
            {'frame_id': LaunchConfiguration('camera_frame_id')},
            {'udp_port': LaunchConfiguration('camera_udp_port')},
        ],
    )

    point_mission_node = Node(
        package='point_mission_mode',
        executable='point_mission_mode',
        name='point_mission_mode',
        output='screen',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            LaunchConfiguration('point_param_file'),
        ],
    )

    return LaunchDescription([
        use_sim_time_arg,
        enable_camera_arg,
        camera_udp_port_arg,
        camera_image_topic_arg,
        camera_info_topic_arg,
        camera_frame_id_arg,
        point_param_file_arg,
        camera_node,
        point_mission_node,
    ])
