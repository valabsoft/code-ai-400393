cmake_minimum_required(VERSION 3.14)
project(mrcv-morphologyImage LANGUAGES CXX)

include("../../cmake/utils.cmake")
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

if(is_top_level)
    find_package( mrcv REQUIRED )
    find_package( OpenCV REQUIRED )
    IF (WIN32)
      set( OpenCV_INCLUDE_DIRS "C:/opencv-4.10.0-build/install/include/" )
      set( OpenCV_LIB_DIR "C:/opencv-4.10.0-build/install/x64/vc17/lib/" )
    ELSE()
      set( OpenCV_INCLUDE_DIRS "/usr/local/include/opencv4/" )
      set( OpenCV_LIB_DIR "/usr/local/lib/" )
    ENDIF()
endif()

set(sources main.cpp)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

add_executable(mrcv-morphologyImage)
target_sources(mrcv-morphologyImage PRIVATE ${sources})
target_link_libraries(mrcv-morphologyImage PRIVATE mrcv::mrcv)

if(NOT is_top_level)
    win_copy_deps_to_target_dir(mrcv-morphologyImage mrcv::mrcv)
endif()
