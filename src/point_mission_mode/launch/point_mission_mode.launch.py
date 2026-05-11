from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )

    return LaunchDescription([
        use_sim_time_arg,

        Node(
            package='point_mission_mode',
            executable='point_mission_mode',
            name='point_mission_mode',
            output='screen',
            parameters=[
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
                PathJoinSubstitution([FindPackageShare('point_mission_mode'), 'cfg', 'params.yaml'])
            ]
        ),
    ])
