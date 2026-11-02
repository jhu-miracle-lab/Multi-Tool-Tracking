from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("arms", default_value="['psm1', 'psm2']"),
            DeclareLaunchArgument("allow_primary_switch", default_value="false"),
            DeclareLaunchArgument("n_particles", default_value="300"),
            DeclareLaunchArgument("warmup_frames", default_value="50"),
            DeclareLaunchArgument("image_topic", default_value="/camera/left/image_raw"),
            DeclareLaunchArgument("camera_frame", default_value="left_camera_optical_frame"),
            Node(
                package="surgical_tool_tracking_ros2",
                executable="tracking_node",
                name="dual_arm_tracking",
                output="screen",
                parameters=[
                    {
                        "arms": LaunchConfiguration("arms"),
                        "allow_primary_switch": LaunchConfiguration("allow_primary_switch"),
                        "n_particles": LaunchConfiguration("n_particles"),
                        "warmup_frames": LaunchConfiguration("warmup_frames"),
                        "image_topic": LaunchConfiguration("image_topic"),
                        "camera_frame": LaunchConfiguration("camera_frame"),
                    }
                ],
            ),
        ]
    )
