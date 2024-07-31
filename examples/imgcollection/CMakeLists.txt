cmake_minimum_required(VERSION 3.14)
project(mrcv-imgcollection LANGUAGES CXX)

include("../../cmake/utils.cmake")
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

if(is_top_level)
    find_package(mrcv REQUIRED)
endif()

set(sources main.cpp)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

add_executable(mrcv-imgcollection)
target_sources(mrcv-imgcollection PRIVATE ${sources})
target_link_libraries(mrcv-imgcollection PRIVATE mrcv::mrcv)

if(NOT is_top_level)
    win_copy_deps_to_target_dir(mrcv-imgcollection mrcv::mrcv)
endif()
