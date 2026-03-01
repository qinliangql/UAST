cmake_minimum_required(VERSION 2.8.3)
project(so3_control)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  nav_msgs
  quadrotor_msgs
  tf
  nodelet
  cmake_utils
)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

catkin_package()

find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_library(SO3Control src/SO3Control.cpp)
add_library(so3_control_nodelet src/so3_control_nodelet.cpp)

target_link_libraries(so3_control_nodelet
  ${catkin_LIBRARIES}
  SO3Control
)

# 普通PID控制器
add_executable(control_example src/control_example.cpp)
target_link_libraries(control_example
  ${catkin_LIBRARIES}
)

# 接收加速度直接转为姿态，并叠加干扰观测器作为内环输入
add_executable(network_control_node src/NetworkControl.cpp src/network_control_node.cpp)
target_link_libraries(network_control_node
  ${catkin_LIBRARIES}
  SO3Control
)
