﻿cmake_minimum_required(VERSION 3.14)
project(mrcv-class LANGUAGES CXX)

include("../../cmake/utils.cmake")
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

if(is_top_level)
    find_package( mrcv REQUIRED )    
endif()

set(sources main.cpp)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

add_executable(mrcv-class)
target_sources(mrcv-class PRIVATE ${sources})
target_link_libraries(mrcv-class PRIVATE mrcv::mrcv)

if(NOT is_top_level)
    win_copy_deps_to_target_dir(mrcv-class mrcv::mrcv)
endif()
