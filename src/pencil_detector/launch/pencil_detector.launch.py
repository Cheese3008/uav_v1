#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    share = FindPackageShare('pencil_detector')
    params = PathJoinSubstitution([share, 'cfg', 'pencil_params.yaml'])
    return LaunchDescription([
        Node(
            package='pencil_detector',
            executable='pencil_detector',
            name='pencil_detector_node',
            output='screen',
            parameters=[params],
        )
    ])
