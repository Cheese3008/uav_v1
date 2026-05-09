#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
import os

def generate_launch_description():
    # Get the package share directory
    cube_detector_share = FindPackageShare('cube_detector')
    
    # Path to the parameters file
    params_file = PathJoinSubstitution([cube_detector_share, 'cfg', 'params.yaml'])
    
    # Create the node
    cube_detector_node = Node(
        package='cube_detector',
        executable='cube_detector',
        name='cube_detector',
        output='screen',
        parameters=[params_file],
        remappings=[
            # You can remap topics here if needed
            # ('/image', '/camera/image_raw'),
            # ('/camera_info', '/camera/camera_info'),
        ]
    )
    
    return LaunchDescription([
        cube_detector_node
    ])
