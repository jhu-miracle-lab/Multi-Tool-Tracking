"""Launch the dvrk_display unixfd stereo capture publisher together with
the dual-arm tracking node, wired to the same topic names by default.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("handeye", default_value=""),
            DeclareLaunchArgument("arms", default_value="['psm1', 'psm2']"),
            DeclareLaunchArgument("allow_primary_switch", default_value="false"),
            DeclareLaunchArgument("shaft_backend", default_value="elsed"),
            DeclareLaunchArgument("n_particles", default_value="300"),
            DeclareLaunchArgument("warmup_frames", default_value="50"),
            DeclareLaunchArgument("left_topic", default_value="/camera/left/image_raw"),
            DeclareLaunchArgument("right_topic", default_value="/camera/right/image_raw"),
            DeclareLaunchArgument("camera_frame", default_value="left_camera_optical_frame"),
            DeclareLaunchArgument("socket_path", default_value=""),
            DeclareLaunchArgument("dekr_model_path", default_value=""),
            DeclareLaunchArgument("dekr_fp16", default_value="true"),
            DeclareLaunchArgument("dekr_compile", default_value="true"),
            DeclareLaunchArgument("dekr_use_identity_head", default_value="true"),
            DeclareLaunchArgument("no_display", default_value="false"),
            DeclareLaunchArgument("display_scale", default_value="2.0"),
            Node(
                package="surgical_tool_tracking_ros2",
                executable="dvrk_stereo_publisher",
                name="dvrk_stereo_publisher",
                output="screen",
                parameters=[
                    {
                        "socket_path": LaunchConfiguration("socket_path"),
                        "left_topic": LaunchConfiguration("left_topic"),
                        "right_topic": LaunchConfiguration("right_topic"),
                    }
                ],
            ),
            Node(
                package="surgical_tool_tracking_ros2",
                executable="tracking_node",
                name="dual_arm_tracking",
                output="screen",
                parameters=[
                    {
                        "handeye": LaunchConfiguration("handeye"),
                        "arms": LaunchConfiguration("arms"),
                        "allow_primary_switch": LaunchConfiguration("allow_primary_switch"),
                        "shaft_backend": LaunchConfiguration("shaft_backend"),
                        "n_particles": LaunchConfiguration("n_particles"),
                        "warmup_frames": LaunchConfiguration("warmup_frames"),
                        "image_topic": LaunchConfiguration("left_topic"),
                        "right_image_topic": LaunchConfiguration("right_topic"),
                        "camera_frame": LaunchConfiguration("camera_frame"),
                        "dekr_model_path": LaunchConfiguration("dekr_model_path"),
                        "dekr_fp16": LaunchConfiguration("dekr_fp16"),
                        "dekr_compile": LaunchConfiguration("dekr_compile"),
                        "dekr_use_identity_head": LaunchConfiguration("dekr_use_identity_head"),
                        "no_display": LaunchConfiguration("no_display"),
                        "display_scale": LaunchConfiguration("display_scale"),
                    }
                ],
            ),
        ]
    )
