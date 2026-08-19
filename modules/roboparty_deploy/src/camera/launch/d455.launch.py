from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    default_config = Path(
        get_package_share_directory("roboparty_camera")
    ) / "config" / "d455.yaml"
    realsense_launch = Path(
        get_package_share_directory("realsense2_camera")
    ) / "launch" / "rs_launch.py"

    arguments = {
        "camera_namespace": LaunchConfiguration("camera_namespace"),
        "camera_name": LaunchConfiguration("camera_name"),
        "config_file": LaunchConfiguration("config_file"),
    }

    return LaunchDescription(
        [
            DeclareLaunchArgument("camera_namespace", default_value="camera"),
            DeclareLaunchArgument("camera_name", default_value="d455"),
            DeclareLaunchArgument("config_file", default_value=str(default_config)),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(realsense_launch)),
                launch_arguments=arguments.items(),
            ),
        ]
    )
