
#
#  Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
#
#  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
#  rights in and to this software, related documentation and any modifications thereto.
#  Any use, reproduction, disclosure or distribution of this software and related
#  documentation without an express license agreement from NVIDIA Corporation is strictly
#  prohibited.
#
#  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
#  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
#  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
#  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
#  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
#  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
#  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
#  SUCH DAMAGES
#

# Finds sutil's copies of the libraries before looking around for the system libraries on Windows platforms.

IF (WIN32)
  SET(GLUT_ROOT_PATH $ENV{GLUT_ROOT_PATH})
ENDIF (WIN32)

find_package(OpenGL)

if(WIN32)
  # For whatever reason, cmake doesn't detect that a library is 32 or 64 bits,
  # so we have to selectively look for it in one of two places.
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(dir win64)
  else() # 32 bit
    set(dir win32)
  endif()
  find_library(GLUT_glut_LIBRARY names freeglut
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/support/freeglut/${dir}/Release
    NO_DEFAULT_PATH
    )
  find_file(GLUT_glut_DLL names freeglut.dll
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/support/freeglut/${dir}/Release
    NO_DEFAULT_PATH
    )
  find_path(GLUT_INCLUDE_DIR GL/glut.h
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/support/freeglut/include
    )
  if( GLUT_glut_LIBRARY AND
      GLUT_glut_DLL     AND
      GLUT_INCLUDE_DIR
      )
    # We need to set some of the same variables that FindGLUT.cmake does.
    set(GLUT_FOUND TRUE)
    set(GLUT_LIBRARIES "${GLUT_glut_LIBRARY}"
      #winmm.lib
      )
    set(sources ${sources} ${GLUT_INCLUDE_DIR}/GL/glut.h)

  endif() # All the components were found

  # Mark the libraries as advanced
  mark_as_advanced(
    GLUT_glut_LIBRARY
    GLUT_glut_DLL
    GLUT_INCLUDE_DIR
    )
else() # Now for everyone else
  find_package(GLUT REQUIRED)
  # Some systems don't need Xmu for glut.  Remove it if it wasn't found.
  if( DEFINED GLUT_Xmu_LIBRARY AND NOT GLUT_Xmu_LIBRARY)
    list(REMOVE_ITEM GLUT_LIBRARIES ${GLUT_Xmu_LIBRARY})
  endif()
endif()
