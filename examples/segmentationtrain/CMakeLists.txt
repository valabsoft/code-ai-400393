﻿cmake_minimum_required(VERSION 3.14)
project(mrcv-segmentationtrain LANGUAGES CXX)

include("../../cmake/utils.cmake")
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

if(is_top_level)    
    find_package(mrcv REQUIRED)
    find_package(OpenCV REQUIRED)
    find_package(Torch REQUIRED)

    set(CMAKE_PREFIX_PATH "C:/libtorch-12.4/Release/")

    # OpenCV DIRs
    IF (WIN32)
        set( OpenCV_INCLUDE_DIRS "C:/opencv-4.10.0-build/install/include/" )
        set( OpenCV_LIB_DIR "C:/opencv-4.10.0-build/install/x64/vc17/lib/" )
    ELSE()
        set( OpenCV_INCLUDE_DIRS "/usr/local/include/opencv4/" )
        set( OpenCV_LIB_DIR "/usr/local/lib/" )
    ENDIF()

    # Torch DIRs
    IF (WIN32)
        set( Torch_INCLUDE_DIRS "C:/libtorch-12.4/Release/include/" )
        set( Torch_LIB_DIR "C:/libtorch-12.4/Release/lib/" )
    ELSE()
        set( Torch_INCLUDE_DIRS "/usr/local/include/torch/" )
        set( Torch_LIB_DIR "/usr/local/lib/" )
    ENDIF()
endif()

set(sources main.cpp)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

add_executable(mrcv-segmentationtrain)
target_sources(mrcv-segmentationtrain PRIVATE ${sources})
target_link_libraries(mrcv-segmentationtrain PRIVATE mrcv::mrcv)

if(NOT is_top_level)
    win_copy_deps_to_target_dir(mrcv-segmentationtrain mrcv::mrcv)
endif()
