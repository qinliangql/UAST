cmake_minimum_required(VERSION 3.0.2)
project(sensor_simulator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS
        roscpp
        nav_msgs
        std_msgs
        sensor_msgs
        cv_bridge
)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP)
find_package(yaml-cpp REQUIRED)
message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

# CUDA
find_package(CUDA REQUIRED)
# Query the GPU architecture
cuda_select_nvcc_arch_flags(ARCH_FLAGS)
if("${ARCH_FLAGS}" STREQUAL "")
  message(FATAL_ERROR
    "Failed to automatically detect CUDA architecture. "
    "Please manually set ARCH_FLAGS in CMakeList.txt (line 25) based on your device, for example:\n"
    "  set(ARCH_FLAGS \"-gencode arch=compute_86,code=sm_86\")"
  )
endif()
message(WARNING "CUDA Architecture: ${ARCH_FLAGS}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${ARCH_FLAGS}")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

## Declare a catkin package
catkin_package(
        INCLUDE_DIRS include
        LIBRARIES sensor_simulator
        CATKIN_DEPENDS roscpp
        DEPENDS OpenCV)

## Specify additional locations of header files
include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}  # Eigen库路径
        ${PCL_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

add_executable(sensor_simulator src/test_simulator.cpp src/sensor_simulator.cpp)
target_link_libraries(sensor_simulator ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX yaml-cpp)
target_compile_definitions(sensor_simulator PRIVATE CONFIG_FILE_PATH="${CMAKE_SOURCE_DIR}/config/config.yaml")


cuda_add_library(raycast_cuda src/sensor_simulator.cu)
target_link_libraries(raycast_cuda ${CUDA_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(sensor_simulator_cuda src/test_simulator_cuda.cpp src/maps.cpp src/perlinnoise.cpp)
target_link_libraries(sensor_simulator_cuda ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} raycast_cuda yaml-cpp)
target_compile_definitions(sensor_simulator_cuda PRIVATE CONFIG_FILE_PATH="${CMAKE_SOURCE_DIR}/config/config.yaml")

add_executable(sensor_simulator_cuda_multi src/test_simulator_cuda_multi.cpp src/maps.cpp src/perlinnoise.cpp)
target_link_libraries(sensor_simulator_cuda_multi ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} raycast_cuda yaml-cpp)
target_compile_definitions(sensor_simulator_cuda_multi PRIVATE CONFIG_FILE_PATH="${CMAKE_SOURCE_DIR}/config/config.yaml")

add_executable(dataset_generator src/dataset_generator.cpp src/maps.cpp src/perlinnoise.cpp)
target_link_libraries(dataset_generator ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} raycast_cuda yaml-cpp)
target_compile_definitions(dataset_generator PRIVATE CONFIG_FILE_PATH="${CMAKE_SOURCE_DIR}/config/config.yaml")