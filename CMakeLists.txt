project(cudaPcl)
cmake_minimum_required(VERSION 2.4.6)

set(POD_NAME cudaPcl)
include(cmake/pods.cmake)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)


set(CMAKE_COLOR_MAKEFILE ON)
IF(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug) # Release; debug; ...
  set(CMAKE_BUILD_TYPE Release) # Release; debug; ...
ENDIF()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "default install path" FORCE )
endif()

# add flags depending on build type
set(CMAKE_CXX_FLAGS_RELEASE "-g -Wall -Wno-deprecated -O2 -finline-functions -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wno-deprecated")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\")

find_package (Eigen3 REQUIRED)
find_package (CUDA REQUIRED)
find_package (OpenCV REQUIRED)
find_package (PCL REQUIRED)
FIND_PACKAGE (Boost 1.54 REQUIRED)
#FIND_PACKAGE(OpenCL)

message(${EIGEN3_INCLUDE_DIR})
message(${CUDA_INCLUDE_DIRS})
message(${CUDA_TOOLKIT_INCLUDE})
message(${CUDA_TOOLKIT_ROOT_DIR})
message(${CUDA_LIBRARIES})
message(${Boost_LIBRARY_DIRS})
#message(${PCL_INCLUDE_DIRS})

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${CUDA_TOOLKIT_INCLUDE}
#  include/nvidia/
#  ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
#  ${CUDA_TOOLKIT_ROOT_DIR}/samples/3_Imaging/convolutionSeparable/
#  /usr/include
#  $ENV{WORKSPACE_HOME}/3rdparty/include/pcl-1.7/
#  /usr/include/ni
#  /usr/include/vtk-5.8/
  ${Boost_INCLUDE_DIRS}
#  $ENV{WORKSPACE_HOME}/3rdparty/include/
#  $ENV{WORKSPACE_HOME}/3rdparty/include/libfreenect2/tinythread/
)

link_directories(
  ${PCL_LIBRARY_DIRS}
  ${Boost_LIBRARY_DIRS}
#  $ENV{WORKSPACE_HOME}/3rdparty/lib
)

add_executable(pclNormals ./src/pclNormals.cpp)
target_link_libraries (pclNormals
  boost_system boost_thread  boost_program_options
  pcl_common pcl_io pcl_search pcl_visualization pcl_features
  vtkCommon vtkFiltering vtkRendering
  OpenNI XnVNite
  opencv_core opencv_highgui)

cuda_add_executable(openniSmoothDepth ./src/openniSmoothDepth.cpp
  ./cuda/cuda_pcl_helpers.cu
  ./cuda/integral.cu
  ./cuda/guidedFilter.cu
  ./cuda/convolutionSeparable.cu
  )
target_link_libraries (openniSmoothDepth boost_system boost_thread pcl_common
 boost_program_options
  opencv_core opencv_highgui opencv_contrib  opencv_imgproc
  pcl_io pcl_visualization OpenNI XnVNite
  vtkCommon vtkFiltering vtkRendering
  )

cuda_add_executable(openniSmoothNormals ./src/openniSmoothNormals.cpp
  ./cuda/cuda_pcl_helpers.cu
  ./cuda/integral.cu
  ./cuda/guidedFilter.cu
  ./cuda/convolutionSeparable.cu
  )
target_link_libraries (openniSmoothNormals
  boost_system boost_thread pcl_common boost_program_options
  opencv_core opencv_highgui opencv_contrib opencv_imgproc
  pcl_io pcl_visualization OpenNI XnVNite
  vtkCommon vtkFiltering vtkRendering
  )

#add_executable(freenect2Vis ./src/freenect2Visualizer.cpp)
#target_link_libraries (freenect2Vis
#  boost_system boost_thread freenect2 boost_program_options
#  opencv_highgui opencv_core
#  )

add_executable(pclGrabber ./src/pclGrabber.cpp)
target_link_libraries (pclGrabber boost_system boost_thread pcl_common pcl_io pcl_visualization OpenNI XnVNite)

add_executable(pclBenchmark ./src/pclBenchmark.cpp)
target_link_libraries (pclBenchmark boost_system boost_thread pcl_common pcl_io pcl_visualization OpenNI XnVNite)

cuda_add_library(cudaPcl SHARED
  ./cuda/cuda_pcl_helpers.cu
  ./cuda/integral.cu
  ./cuda/guidedFilter.cu
  ./src/pcl_helpers.cpp
  ./cuda/convolutionSeparable.cu
  ./cuda/bilateralFilter.cu
  ./cuda/optimizationSO3.cu
)

target_link_libraries (cudaPcl
  boost_system boost_thread pcl_common
  opencv_core opencv_highgui opencv_contrib opencv_imgproc
  pcl_io pcl_visualization OpenNI XnVNite
  vtkCommon vtkFiltering vtkRendering
  )


INSTALL(TARGETS cudaPcl
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  )

pods_install_libraries(cudaPcl)
pods_install_executables(pclGrabber pclBenchmark pclNormals
  openniSmoothNormals openniSmoothDepth)

pods_install_headers(
  ./include/cudaPcl/convolutionSeparable_common.h 
  ./include/cudaPcl/convolutionSeparable_common_small.h
  ./include/cudaPcl/cuda_pc_helpers.h
  ./include/cudaPcl/cudaSphereHelpers.h
  ./include/cudaPcl/cv_helpers.hpp
  ./include/cudaPcl/depthGuidedFilter.hpp
  ./include/cudaPcl/freenect2Grabber.hpp
  ./include/cudaPcl/normalExtractSimpleGpu.hpp
  ./include/cudaPcl/openniGrabber.hpp
  ./include/cudaPcl/openniSmoothDepth.hpp
  ./include/cudaPcl/openniSmoothDepthGpu.hpp
  ./include/cudaPcl/openniSmoothNormalsGpu.hpp
  ./include/cudaPcl/openniVisualizer.hpp
  ./include/cudaPcl/pcl_helpers.hpp
  ./include/cudaPcl/root_includes.hpp
  ./include/cudaPcl/dirSeg.hpp
  DESTINATION cudaPcl)

#install(DIRECTORY include/cudaPcl/ DESTINATION include/cudaPcl/
#          FILES_MATCHING PATTERN "*.h")
#install(DIRECTORY include/cudaPcl/ DESTINATION include/cudaPcl/
#          FILES_MATCHING PATTERN "*.hpp")
