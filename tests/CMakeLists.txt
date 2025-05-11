cmake_minimum_required(VERSION 3.14)
project(mrcv-tests)

#----------------------------------------------------------------------------------------------------------------------
# general settings and options
#----------------------------------------------------------------------------------------------------------------------

include("../cmake/utils.cmake")
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

if(is_top_level)
    enable_testing()
endif()

#----------------------------------------------------------------------------------------------------------------------
# testing framework
#----------------------------------------------------------------------------------------------------------------------

include(FetchContent)
FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE) # do not override parent project's runtime settings on Windows
set(INSTALL_GTEST OFF)

# For simplicity, always build googletest as static library. This prevents mrcv-tests executable from
# complaining about missing googletest DLLs on Windows.
set(BUILD_SHARED_LIBS OFF)

FetchContent_MakeAvailable(googletest)

#----------------------------------------------------------------------------------------------------------------------
# tests dependencies
#----------------------------------------------------------------------------------------------------------------------

if(USE_CUDA)
    # Добавляем макрос MRCV_CUDA_ENABLED
    target_compile_definitions(mrcv PRIVATE MRCV_CUDA_ENABLED)
    set(CMAKE_PREFIX_PATH "C:/libtorch-12.4-cuda/Release;C:/yaml-cpp")
    set(OpenCV_DIR "C:/opencv-4.10.0-build-cuda/install")
    set(Torch_DIR "C:/libtorch-12.4-cuda/Release/share/cmake/Torch")
    # OpenCV DIRs
    if(WIN32)
        set(OpenCV_INCLUDE_DIRS "C:/opencv-4.10.0-build-cuda/install/include")
        set(OpenCV_LIB_DIR "C:/opencv-4.10.0-build-cuda/install/x64/vc17/lib")
    else()
        set(OpenCV_INCLUDE_DIRS "/usr/local/include/opencv4")
        set(OpenCV_LIB_DIR "/usr/local/lib")
    endif()
    # Torch DIRs
    if(WIN32)
        set(Torch_INCLUDE_DIRS "C:/libtorch-12.4-cuda/Release/include")
        set(Torch_LIB_DIR "C:/libtorch-12.4-cuda/Release/lib")
    else()
        set(Torch_INCLUDE_DIRS "/usr/local/include/torch")
        set(Torch_LIB_DIR "/usr/local/lib")
    endif()
else()
    set(CMAKE_PREFIX_PATH "C:/libtorch-12.4/Release;C:/yaml-cpp")
    set(OpenCV_DIR "C:/opencv-4.10.0-build/install")
    set(Torch_DIR "C:/libtorch-12.4/Release/share/cmake/Torch")
    # OpenCV DIRs
    if(WIN32)
        set(OpenCV_INCLUDE_DIRS "C:/opencv-4.10.0-build/install/include")
        set(OpenCV_LIB_DIR "C:/opencv-4.10.0-build/install/x64/vc17/lib")
    else()
        set(OpenCV_INCLUDE_DIRS "/usr/local/include/opencv4")
        set(OpenCV_LIB_DIR "/usr/local/lib")
    endif()
    # Torch DIRs
    if(WIN32)
        set(Torch_INCLUDE_DIRS "C:/libtorch-12.4/Release/include")
        set(Torch_LIB_DIR "C:/libtorch-12.4/Release/lib")
    else()
        set(Torch_INCLUDE_DIRS "/usr/local/include/torch")
        set(Torch_LIB_DIR "/usr/local/lib")
    endif()
endif()

# Диагностика
message(STATUS "Torch_DIR: ${Torch_DIR}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

# Поиск библиотек
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(yaml-cpp REQUIRED)

# Вывод информации о найденных библиотеках
message(STATUS "OpenCV_FOUND: ${OpenCV_FOUND}")
message(STATUS "Torch_FOUND: ${Torch_FOUND}")
message(STATUS "yaml-cpp_FOUND: ${yaml-cpp_FOUND}")

include_directories( ${OpenCV_INCLUDE_DIRS}; ${Torch_INCLUDE_DIRS}; ${YAML_CPP_INCLUDE_DIR} )
link_directories( ${OpenCV_LIB_DIR}; ${Torch_LIB_DIR}; ${YAML_CPP_LIBRARY_DIR} )

#----------------------------------------------------------------------------------------------------------------------
# tests sources
#----------------------------------------------------------------------------------------------------------------------

set(sources
    add_test.cpp
    augmentation_test.cpp
    clustering_test.cpp
    compareimages_test.cpp
    disparitymap_test.cpp
    imgpreprocessing_test.cpp
    morphologyimage_test.cpp
    objcourse_test.cpp
    sensorsfusion_test.cpp
)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

#----------------------------------------------------------------------------------------------------------------------
# tests target
#----------------------------------------------------------------------------------------------------------------------

add_executable(mrcv-tests)
target_sources(mrcv-tests PRIVATE ${sources} "augmentation_test.cpp" "clustering_test.cpp" "compareimages_test.cpp" "configuration_test.cpp" "disparitymap_test.cpp" "geometry_test.cpp" "imgpreprocessing_test.cpp" "morphologyimage_test.cpp" "objcourse_test.cpp" "sensorsfusion_test.cpp")

target_link_libraries(mrcv-tests
    PRIVATE
        mrcv::mrcv
        gtest_main)

if(NOT is_top_level)
    win_copy_deps_to_target_dir(mrcv-tests mrcv::mrcv)
endif()

include(GoogleTest)
gtest_discover_tests(mrcv-tests)
