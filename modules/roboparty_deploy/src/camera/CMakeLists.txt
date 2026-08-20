cmake_minimum_required(VERSION 3.8)
project(roboparty_camera)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/roboparty_camera_web.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME camera_web_server
)

ament_package()
