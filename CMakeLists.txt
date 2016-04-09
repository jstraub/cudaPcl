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

set(DRC OFF)
set(OPENNI ON)

#set(VTK_DIR /usr/bin/)

find_package (Eigen3 REQUIRED)
find_package (CUDA REQUIRED)
#find_package (VTK REQUIRED)
FIND_PACKAGE (Boost 1.54 REQUIRED)
message(${EIGEN3_INCLUDE_DIR})
message(${CUDA_INCLUDE_DIRS})
message(${CUDA_TOOLKIT_INCLUDE})
message(${CUDA_TOOLKIT_ROOT_DIR})
message(${CUDA_LIBRARIES})
message(${Boost_LIBRARY_DIRS})
message("VTK ")
#message(${VTK_LIBRARY_DIRS})
#message(${VTK_INCLUDE_DIRS})

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${CUDA_TOOLKIT_INCLUDE}
  ${Boost_INCLUDE_DIRS}
#  ${VTK_INCLUDE_DIRS}
#  /usr/include/
)

link_directories(
  ${Boost_LIBRARY_DIRS}
  ${VTK_LIBRARY_DIRS}
  /usr/lib/
)

if(NOT DRC)
  find_package (OpenCV REQUIRED)
  find_package (PCL REQUIRED)
  message(${OpenCV_LIBS})
  message(${OpenCV_INCLUDE_DIRS})
  message(${PCL_INCLUDE_DIRS})

  include_directories(
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    )

  link_directories(
    ${PCL_LIBRARY_DIRS}
    /usr/lib/
    )
endif()

#add_executable(pclNormals ./src/pclNormals.cpp)
#target_link_libraries (pclNormals
#  boost_system boost_thread  boost_program_options
#  pcl_common pcl_io pcl_search pcl_visualization pcl_features
#  vtkCommon vtkFiltering vtkRendering
#  OpenNI XnVNite
#  opencv_core opencv_highgui)

if(OPENNI)
cuda_add_library(cudaPcl SHARED
  ./cuda/cuda_pcl_helpers.cu
  ./cuda/integral.cu
  ./cuda/guidedFilter.cu
  ./cuda/convolutionSeparable.cu
  ./cuda/surfelRender.cu
  ./src/pinhole.cpp
#  ./src/pcl_helpers.cpp
#  ./cuda/bilateralFilter.cu
#  ./cuda/optimizationSO3.cu
)

target_link_libraries (cudaPcl
  boost_system boost_thread pcl_common
  opencv_core opencv_highgui opencv_contrib opencv_imgproc
  pcl_io pcl_visualization OpenNI #XnVNite
  vtkCommon vtkFiltering vtkRendering
  )
  pods_install_libraries(cudaPcl)
endif()

if(NOT DRC AND OPENNI)

  cuda_add_executable(openniPyramid ./src/openniPyramid.cpp
    )
  target_link_libraries (openniPyramid
    cudaPcl
    boost_system boost_thread pcl_common
    boost_program_options
    opencv_core opencv_highgui opencv_contrib  opencv_imgproc
    pcl_io pcl_visualization OpenNI #XnVNite
    vtkCommon vtkFiltering vtkRendering
    )

  cuda_add_executable(openniSmoothDepth ./src/openniSmoothDepth.cpp
    )
  target_link_libraries (openniSmoothDepth 
    cudaPcl
    boost_system boost_thread pcl_common
    boost_program_options
    opencv_core opencv_highgui opencv_contrib  opencv_imgproc
    pcl_io pcl_visualization OpenNI #XnVNite
    vtkCommon vtkFiltering vtkRendering
    )

  cuda_add_executable(openniSmoothNormals ./src/openniSmoothNormals.cpp
    )
  target_link_libraries (openniSmoothNormals
    cudaPcl
    boost_system boost_thread pcl_common boost_program_options
    opencv_core opencv_highgui opencv_contrib opencv_imgproc
    pcl_io pcl_visualization OpenNI #XnVNite
    vtkCommon vtkFiltering vtkRendering
    )

  cuda_add_executable(openniSmoothNormals_file
    ./src/openniSmoothNormals_file.cpp
    )
  target_link_libraries (openniSmoothNormals_file
    cudaPcl
    boost_system boost_thread pcl_common boost_program_options
    opencv_core opencv_highgui opencv_contrib opencv_imgproc
    pcl_io pcl_visualization OpenNI #XnVNite
    vtkCommon vtkFiltering vtkRendering
    )

  add_executable(pclGrabber ./src/pclGrabber.cpp)
  target_link_libraries (pclGrabber boost_system boost_thread
    pcl_common pcl_io pcl_visualization OpenNI #XnVNite
    )

  add_executable(pclBenchmark ./src/pclBenchmark.cpp)
  target_link_libraries (pclBenchmark boost_system boost_thread
    pcl_common pcl_io pcl_visualization OpenNI #XnVNite
    )

  add_executable(rndTransformPc ./src/randomlyTransformPointClouds.cpp)
  target_link_libraries(rndTransformPc boost_system boost_thread
    boost_program_options
    pcl_common pcl_io pcl_visualization)

  pods_install_executables(pclGrabber openniSmoothNormals openniPyramid
    openniSmoothNormals_file
    pclBenchmark 
    openniSmoothDepth)
endif()
#add_executable(freenect2Vis ./src/freenect2Visualizer.cpp)
#target_link_libraries (freenect2Vis
#  boost_system boost_thread freenect2 boost_program_options
#  opencv_highgui opencv_core
#  )


  add_executable(renderPcFromPc ./src/renderPcFromPc.cpp
    ./src/pinhole.cpp)
  target_link_libraries (renderPcFromPc boost_system boost_thread
    boost_program_options pcl_filters
    pcl_common  pcl_features pcl_search pcl_io pcl_visualization)

  add_executable(rndTransformPc ./src/randomlyTransformPointClouds.cpp)
  target_link_libraries(rndTransformPc boost_system boost_thread
    boost_program_options
    pcl_common pcl_io pcl_visualization )

  cuda_add_executable(testSurfelRender 
    ./test/surfelRender.cpp)
  target_link_libraries(testSurfelRender
    jsCore cudaPcl 
    ${OpenCV_LIBS}
    )

  pods_install_executables( rndTransformPc renderPcFromPc
    testSurfelRender)

  cuda_add_executable(realSenseSmoothNormalsGpuApp
    ./src/realSenseSmoothNormalsGpuApp.cpp
    )
  target_link_libraries (realSenseSmoothNormalsGpuApp
    cudaPcl realsense rgbdGrabber
#    boost_system boost_thread pcl_common
#    boost_program_options
    opencv_core opencv_highgui opencv_contrib  opencv_imgproc
#    pcl_io pcl_visualization OpenNI #XnVNite
#    vtkCommon vtkFiltering vtkRendering
    )

pods_install_headers(
  ./include/cudaPcl/convolutionSeparable_common.h 
  ./include/cudaPcl/convolutionSeparable_common_small.h
  ./include/cudaPcl/cuda_pc_helpers.h
  ./include/cudaPcl/cudaSphereHelpers.h
  ./include/cudaPcl/cv_helpers.hpp
  ./include/cudaPcl/depthGuidedFilter.hpp
  ./include/cudaPcl/depthGuidedFilter_impl.hpp
  ./include/cudaPcl/freenect2Grabber.hpp
  ./include/cudaPcl/normalExtractSimpleGpu.hpp
  ./include/cudaPcl/normalExtractSimpleGpu_impl.hpp
  ./include/cudaPcl/openniGrabber.hpp
  ./include/cudaPcl/openniSmoothDepth.hpp
  ./include/cudaPcl/openniSmoothDepthGpu.hpp
  ./include/cudaPcl/openniSmoothNormalsGpu.hpp
  ./include/cudaPcl/openniVisualizer.hpp
  ./include/cudaPcl/pcl_helpers.hpp
  ./include/cudaPcl/root_includes.hpp
  ./include/cudaPcl/dirSeg.hpp
  ./include/cudaPcl/pcSeg.hpp
  ./include/cudaPcl/pinhole.h
  ./include/cudaPcl/openniPyramid.hpp
  ./include/cudaPcl/surfel.hpp
  ./include/cudaPcl/smoothNormalsGpu.hpp
  ./include/cudaPcl/realSenseSmoothNormalsGpu.hpp
  DESTINATION cudaPcl)

#install(DIRECTORY include/cudaPcl/ DESTINATION include/cudaPcl/
#          FILES_MATCHING PATTERN "*.h")
#install(DIRECTORY include/cudaPcl/ DESTINATION include/cudaPcl/
#          FILES_MATCHING PATTERN "*.hpp")
