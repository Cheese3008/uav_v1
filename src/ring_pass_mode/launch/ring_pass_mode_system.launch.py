from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    enable_viz_arg = DeclareLaunchArgument(
        'enable_gazebo_viz',
        default_value='true',
        description='Enable Gazebo marker visualization (simulation only)'
    )

    return LaunchDescription([
        enable_viz_arg,

        # # Bridge camera image from Gazebo -> ROS2
        # Node(
        #     package='ros_gz_bridge',
        #     executable='parameter_bridge',
        #     name='image_bridge',
        #     arguments=[
        #         '/world/default/model/x500_mono_cam_0/link/camera_link/sensor/imager/image@sensor_msgs/msg/Image@gz.msgs.Image'
        #     ],
        #     output='screen',
        # ),

        # # Bridge camera info from Gazebo -> ROS2
        # Node(
        #     package='ros_gz_bridge',
        #     executable='parameter_bridge',
        #     name='camera_info_bridge',
        #     arguments=[
        #         '/world/default/model/x500_mono_cam_0/link/camera_link/sensor/imager/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo'
        #     ],
        #     output='screen',
        # ),

        # # Bridge processed image from ROS2 -> Gazebo
        # Node(
        #     package='ros_gz_bridge',
        #     executable='parameter_bridge',
        #     name='image_proc_bridge',
        #     arguments=[
        #         '/image_proc@sensor_msgs/msg/Image[gz.msgs.Image'
        #     ],
        #     parameters=[{
        #         'qos_overrides./image_proc.subscription.reliability': 'best_effort',
        #         'qos_overrides./image_proc.publisher.reliability': 'best_effort'
        #     }],
        #     output='screen',
        # ),

        # UDP Camera note
        Node(
            package='udp_h264_camera',
            executable='udp_h264_camera_node',
            name='udp_h264_camera',
            output='screen',
        ),

        # Ring detector node
        Node(
            package='ring_detector',
            executable='ring_detector',
            name='ring_detector',
            output='screen',
        ),

        # Target pose fusion node
        Node(
            package='target_pose_fusion',
            executable='target_pose_fusion_node',
            name='target_pose_fusion',
            output='screen',
        ),

        # Ring pass mode node
        Node(
            package='ring_pass_mode',
            executable='ring_pass_mode',
            name='ring_pass_mode',
            output='screen',
        ),
    ])