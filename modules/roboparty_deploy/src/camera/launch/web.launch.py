from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_save_dir = Path.home() / "Pictures" / "roboparty_camera"

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "color_topic",
                default_value="/camera/d455/color/image_raw",
            ),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="/camera/d455/aligned_depth_to_color/image_raw",
            ),
            DeclareLaunchArgument("bind_host", default_value="0.0.0.0"),
            DeclareLaunchArgument("port", default_value="8080"),
            DeclareLaunchArgument("save_dir", default_value=str(default_save_dir)),
            DeclareLaunchArgument("jpeg_quality", default_value="80"),
            Node(
                package="roboparty_camera",
                executable="camera_web_server",
                name="camera_web_server",
                output="screen",
                parameters=[
                    {
                        "color_topic": LaunchConfiguration("color_topic"),
                        "depth_topic": LaunchConfiguration("depth_topic"),
                        "bind_host": LaunchConfiguration("bind_host"),
                        "port": LaunchConfiguration("port"),
                        "save_dir": LaunchConfiguration("save_dir"),
                        "jpeg_quality": LaunchConfiguration("jpeg_quality"),
                    }
                ],
            ),
        ]
    )
