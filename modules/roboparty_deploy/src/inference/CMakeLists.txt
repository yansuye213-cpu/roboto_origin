# SPDX-License-Identifier: GPL-3.0
# Copyright (C) 2025-2026 Luo1imasi

cmake_minimum_required(VERSION 3.16)
project(roboparty_inference)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)

set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(roboparty_motors REQUIRED)
find_package(roboparty_imu REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(ocs2_sqp QUIET)
find_package(ocs2_centroidal_model QUIET)
find_package(ocs2_pinocchio_interface QUIET)
find_package(ocs2_robotic_tools QUIET)

option(YAML_BUILD_SHARED_LIBS "Build shared library" OFF)
option(YAML_CPP_BUILD_TESTS "Build tests" OFF)
option(YAML_CPP_BUILD_TOOLS "Build tools" OFF)
add_subdirectory(thirdparty)

set(PUBLIC_DEPENDENCIES
    ${Boost_LIBRARIES} pthread Eigen3::Eigen fmt::fmt spdlog::spdlog ${ONNXRUNTIME_LIB_FILE} yaml-cpp)

set(PINOCCHIO_DEPENDENCIES)
set(PINOCCHIO_INCLUDE_DIRS)
message(STATUS "Pinocchio found: enabling whole-body robot model")
if(TARGET pinocchio::pinocchio)
  list(APPEND PINOCCHIO_DEPENDENCIES pinocchio::pinocchio)
else()
  list(APPEND PINOCCHIO_DEPENDENCIES ${pinocchio_LIBRARIES})
  list(APPEND PINOCCHIO_INCLUDE_DIRS ${pinocchio_INCLUDE_DIRS})
endif()

set(OCS2_SOURCES)
set(OCS2_DEPENDENCIES)
set(ROBOPARTY_WITH_OCS2 OFF)
if(ocs2_sqp_FOUND AND TARGET ocs2_sqp::ocs2_sqp
   AND ocs2_centroidal_model_FOUND
   AND TARGET ocs2_centroidal_model::ocs2_centroidal_model
   AND ocs2_pinocchio_interface_FOUND
   AND TARGET ocs2_pinocchio_interface::ocs2_pinocchio_interface
   AND ocs2_robotic_tools_FOUND
   AND TARGET ocs2_robotic_tools::ocs2_robotic_tools)
  find_package(Boost REQUIRED COMPONENTS system filesystem log log_setup)
  message(STATUS "OCS2 found: enabling full Pinocchio centroidal MPC backend")
  set(ROBOPARTY_WITH_OCS2 ON)
  list(APPEND OCS2_SOURCES
    src/whole_body_mpc/model/centroidal_model_info.cpp
    src/whole_body_mpc/reference/mode_schedule_adapter.cpp
    src/whole_body_mpc/reference/swing_trajectory.cpp
    src/whole_body_mpc/reference/switched_model_reference_manager.cpp
    src/whole_body_mpc/mpc/ocs2_centroidal_mpc.cpp
  )
  list(APPEND OCS2_DEPENDENCIES
    ocs2_sqp::ocs2_sqp
    ocs2_centroidal_model::ocs2_centroidal_model
    ocs2_pinocchio_interface::ocs2_pinocchio_interface
    ocs2_robotic_tools::ocs2_robotic_tools
  )
else()
  message(STATUS "Full OCS2 centroidal dependencies not found: stand_wbc_mpc_backend=ocs2 will report a runtime error")
endif()

add_subdirectory(src)

add_library(robot STATIC
  src/robot_interface.cpp
)

target_include_directories(robot
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(robot PUBLIC ${PUBLIC_DEPENDENCIES} utils cnpy
  roboparty_motors::roboparty_motors
  roboparty_imu::roboparty_imu
)

pybind11_add_module(robot_py src/pybind_module.cpp)
target_link_libraries(robot_py PUBLIC robot)

add_executable(inference_node
  src/inference_node.cpp
  src/run_logger.cpp
  src/ros_interface.cpp
  src/obs_manager.cpp
  src/standing_stabilizer.cpp
  src/whole_body_mpc/estimation/base_state_estimator.cpp
  src/whole_body_mpc/model/robot_model.cpp
  src/whole_body_mpc/mpc/centroidal_mpc.cpp
  ${OCS2_SOURCES}
  src/whole_body_mpc/reference/contact_schedule.cpp
  src/whole_body_mpc/reference/foot_placement_planner.cpp
  src/whole_body_mpc/reference/recovery_gait_planner.cpp
  src/whole_body_mpc/wbc/contact_force_qp.cpp
  src/whole_body_mpc/wbc/whole_body_wbc.cpp
  src/whole_body_mpc/whole_body_mpc_controller.cpp
)
target_include_directories(inference_node
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${CNPY_INCLUDE_DIR}
  ${PINOCCHIO_INCLUDE_DIRS}
)
target_link_libraries(inference_node PUBLIC
  ${PUBLIC_DEPENDENCIES}
  ${PINOCCHIO_DEPENDENCIES}
  ${OCS2_DEPENDENCIES}
  utils cnpy robot)
if(ROBOPARTY_WITH_OCS2)
  target_compile_definitions(inference_node PRIVATE ROBOPARTY_WITH_OCS2=1)
endif()
ament_target_dependencies(inference_node PUBLIC rclcpp sensor_msgs geometry_msgs std_srvs)

set_target_properties(inference_node PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")

install(TARGETS inference_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(TARGETS robot utils
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY launch 
  DESTINATION share/${PROJECT_NAME})
install(DIRECTORY robots
  DESTINATION share/${PROJECT_NAME}
)
install(FILES ${ONNXRUNTIME_LIB_FILE} ${ONNXRUNTIME_LIB_FILE}.1 ${ONNXRUNTIME_LIB_FILE}.1.21.0
        DESTINATION lib)

set(PYTHON_INSTALL_DIR "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")

install(TARGETS robot_py
  LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}
  ARCHIVE DESTINATION ${PYTHON_INSTALL_DIR}
  RUNTIME DESTINATION ${PYTHON_INSTALL_DIR})

ament_export_dependencies(rclcpp sensor_msgs geometry_msgs)
ament_export_libraries(robot)

ament_package()
