from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    realsense_launch = Path(
        get_package_share_directory("realsense2_camera")
    ) / "launch" / "rs_launch.py"

    arguments = {
        "camera_namespace": LaunchConfiguration("camera_namespace"),
        "camera_name": LaunchConfiguration("camera_name"),
        "serial_no": LaunchConfiguration("serial_no"),
        "enable_color": "true",
        "enable_depth": "true",
        "rgb_camera.color_profile": LaunchConfiguration("color_profile"),
        "depth_module.depth_profile": LaunchConfiguration("depth_profile"),
        "enable_sync": "true",
        "align_depth.enable": "true",
        "pointcloud.enable": "false",
        "enable_gyro": "false",
        "enable_accel": "false",
        "publish_tf": "true",
    }

    return LaunchDescription(
        [
            DeclareLaunchArgument("camera_namespace", default_value="camera"),
            DeclareLaunchArgument("camera_name", default_value="d455"),
            DeclareLaunchArgument("serial_no", default_value="_245022302750"),
            DeclareLaunchArgument("color_profile", default_value="640x480x15"),
            DeclareLaunchArgument("depth_profile", default_value="640x480x15"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(realsense_launch)),
                launch_arguments=arguments.items(),
            ),
        ]
    )
