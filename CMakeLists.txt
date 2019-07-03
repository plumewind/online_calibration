cmake_minimum_required(VERSION 2.8)
project(online_calibration)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

add_compile_options(-std=c++11)
# Check C++11 or C++0x support
#add_definitions(-DCOMPILEDWITHC11)
#if( CMAKE_BUILD_TYPE MATCHES "Debug" )
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -msse3 -std=c++11 -pthread -g -march=native -Wno-deprecated-declarations")
#else( CMAKE_BUILD_TYPE MATCHES "Release" )
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -msse3 -std=c++11 -pthread -O3 -march=native -Wno-deprecated-declarations")
#endif( CMAKE_BUILD_TYPE MATCHES "Debug" )


set(PCL_DIR /usr/local/pcl180/share/pcl-1.8)
find_package(PCL 1.8 REQUIRED)

set(OpenCV_DIR /usr/local/opencv320/share/OpenCV)
find_package(OpenCV 3.2 REQUIRED)

find_package(Ceres REQUIRED)
#find_package(iSAM REQUIRED)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
find_package(Eigen3 REQUIRED)
find_package(Glog REQUIRED)

#find_package(Pangolin REQUIRED)


include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PCL_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIR}
${OpenCV_INCLUDE_DIRS}
${Ceres_LIBRARY_DIRS}
${Glog_INCLUDE_DIRS}
"/usr/local/include/isam/"
${CHOLMOD_INCLUDE_DIR}#nvidia
#${Pangolin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cpp
src/Tracking.cpp
src/Initializer.cpp
src/Viewer.cpp
src/Tools.cpp
src/Calibration.cpp
src/CameraFrame.cpp
src/LaserScan.cpp
src/Solver.cpp
)

target_link_libraries(${PROJECT_NAME}
${PCL_LIBRARIES}
${EIGEN3_LIBS}
${OpenCV_LIBRARIES}
${CERES_LIBRARIES}
${Glog_LIBRARIES}
#${CHOLMOD_LIBS}
#${ISAM_LIBRARIES}
#${Pangolin_LIBRARIES}
)

#add_executable(calib_test test/test.cpp)
#target_link_libraries(calib_test ${PROJECT_NAME})
#set_target_properties(calib_test PROPERTIES COMPILE_FLAGS "-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-O3")

add_executable(odometry_test exmaples/kitti/odometry_test.cpp)
target_link_libraries(odometry_test ${PROJECT_NAME} isam cholmod)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-O3")

# Build tools

