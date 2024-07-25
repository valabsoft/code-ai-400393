cmake_minimum_required(VERSION 3.14)
project(mrcv-libfirst LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
include("../../cmake/utils.cmake")
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

if(is_top_level)
    find_package(mrcv REQUIRED)
endif()
###############################################################################
# Определение зависимостей библиотеки
###############################################################################

find_package( OpenCV REQUIRED )

#set( OpenCV_INCLUDE_DIRS "C:/opencv-4.10.0-build/install/include/" )
#set( OpenCV_LIB_DIR "C:/opencv-4.10.0-build/install/x64/vc17/lib/" )

include_directories( ${OpenCV_INCLUDE_DIRS} )
link_directories(${OpenCV_LIB_DIR})



###############################################################################
# Исходники библиотеки
###############################################################################
set(sources main.cpp)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

add_executable(mrcv-libfirst)
target_sources(mrcv-libfirst PRIVATE ${sources})
target_link_libraries(mrcv-libfirst PRIVATE mrcv::mrcv)

if(NOT is_top_level)
    win_copy_deps_to_target_dir(mrcv-libfirst mrcv::mrcv)
endif()
