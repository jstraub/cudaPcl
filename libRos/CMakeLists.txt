project(librealtimeMF_ros)  
cmake_minimum_required(VERSION 2.4.6)                                           

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

                                                                                
set(CMAKE_COLOR_MAKEFILE ON)                                                    
IF(NOT CMAKE_BUILD_TYPE)                                                        
  set(CMAKE_BUILD_TYPE Release) # Release; debug; ...                           
  #set(CMAKE_BUILD_TYPE Debug) # Release; debug; ...                             
ENDIF()                                                                         
                                                                                
# add flags depending on build type                                             
set(CMAKE_CXX_FLAGS_RELEASE "-g -Wall -Wno-deprecated -O2 -finline-functions -DNDEBUG")         
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wno-deprecated")                                                 
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\")
                                                                                
find_package (Eigen3 REQUIRED)
find_package (CUDA REQUIRED)
find_package (OpenCV REQUIRED)

include_directories(                                                            
  ../include
  ${EIGEN_INCLUDE_DIRS}                                                         
  ${CUDA_TOOLKIT_INCLUDE}
  ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
  /usr/include                                                                  
  #$ENV{WORKSPACE_HOME}/3rdparty/include                                         
  #/usr/include/pcl-1.7/
  /usr/include/ni
  /usr/include/vtk-5.8/
  /opt/ros/groovy/include/pcl-1.6/
)                                                                               
                                                                                
link_directories(
  #$ENV{WORKSPACE_HOME}/3rdparty/lib
  /opt/ros/groovy/lib/
)

cuda_add_library(cudaPcl_ros SHARED
  ../src/pcl_helpers.cpp
  ${CUDA_TOOLKIT_ROOT_DIR}/samples/3_Imaging/convolutionSeparable/convolutionSeparable.cu 
#  ../cuda/convolutionSeparable.cu 
#  ../cuda/convolutionSeparable_small.cu 
  ../cuda/cuda_pcl_helpers.cu)

target_link_libraries(cudaPcl_ros
    opencv_core opencv_highgui                                                    
      pcl_common pcl_io pcl_visualization pcl_filters                               
  )


INSTALL(TARGETS cudaPcl_ros LIBRARY DESTINATION lib)
