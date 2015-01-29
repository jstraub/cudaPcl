project(cudaPcl)                                                                    
cmake_minimum_required(VERSION 2.4.6)                                           

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
FIND_PACKAGE (Boost 1.52 REQUIRED)

message(${EIGEN3_INCLUDE_DIR})
message(${CUDA_INCLUDE_DIRS})
message(${CUDA_TOOLKIT_INCLUDE})
message(${CUDA_TOOLKIT_ROOT_DIR})
message(${Boost_LIBRARY_DIRS})
message(${CUDA_LIBRARIES})
message(${PCL_INCLUDE_DIRS})

include_directories(                                                            
  include
  ${EIGEN3_INCLUDE_DIR}                                       
  ${PCL_INCLUDE_DIRS}                                       
  ${CUDA_TOOLKIT_INCLUDE}
  ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
  ${CUDA_TOOLKIT_ROOT_DIR}/samples/3_Imaging/convolutionSeparable/
  /usr/include                                        
#  $ENV{WORKSPACE_HOME}/3rdparty/include/pcl-1.7/
  /usr/include/ni
  /usr/include/vtk-5.8/
  ${Boost_INCLUDE_DIRS}
)                                                                               
                                                                                
link_directories(
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
  ./cuda/convolutionSeparable_wrap.cu
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
  ./cuda/convolutionSeparable_wrap.cu
  ) 
target_link_libraries (openniSmoothNormals boost_system boost_thread pcl_common
 boost_program_options
  opencv_core opencv_highgui opencv_contrib opencv_imgproc
  pcl_io pcl_visualization OpenNI XnVNite
  vtkCommon vtkFiltering vtkRendering 
  )

add_executable(pclGrabber ./src/pclGrabber.cpp) 
target_link_libraries (pclGrabber boost_system boost_thread pcl_common pcl_io pcl_visualization OpenNI XnVNite)
                                                                                
add_executable(pclBenchmark ./src/pclBenchmark.cpp) 
target_link_libraries (pclBenchmark boost_system boost_thread pcl_common pcl_io pcl_visualization OpenNI XnVNite)

cuda_add_library(cudaPcl SHARED
  ./cuda/cuda_pcl_helpers.cu 
  ./cuda/integral.cu
  ./cuda/guidedFilter.cu
  ./src/pcl_helpers.cpp
  ./cuda/convolutionSeparable_wrap.cu
  ./cuda/cuda_pcl_helpers.cu 
  ./cuda/bilateralFilter.cu
  ./cuda/optimizationSO3.cu
)

target_link_libraries (cudaPcl
  boost_system boost_thread pcl_common
  opencv_core opencv_highgui opencv_contrib opencv_imgproc
  pcl_io pcl_visualization OpenNI XnVNite
  vtkCommon vtkFiltering vtkRendering 
  )

cuda_add_executable(testReduction
  ./src/testReduction.cpp
  ./cuda/testReduction.cu
  ./cuda/cuda_pcl_helpers.cu 
  )

add_subdirectory(./libRos)

INSTALL(TARGETS cudaPcl #realtimeMF_openni
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  )

install(DIRECTORY include/ DESTINATION include/
          FILES_MATCHING PATTERN "*.h")
install(DIRECTORY include/ DESTINATION include/
          FILES_MATCHING PATTERN "*.hpp")
