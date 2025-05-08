cmake_minimum_required(VERSION 3.14)
project(mrcv-geometry LANGUAGES CXX)

include("../../cmake/utils.cmake")
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

if(is_top_level)
    if(USE_CUDA)
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
    find_package( mrcv REQUIRED )
    find_package(OpenCV REQUIRED)
    find_package(Torch REQUIRED)
endif()

set(sources main.cpp)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

add_executable(mrcv-geometry)
target_sources(mrcv-geometry PRIVATE ${sources})

if(USE_CUDA)
    target_compile_definitions(mrcv-geometry PRIVATE MRCV_CUDA_ENABLED)
endif()
target_include_directories(mrcv-geometry PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(mrcv-geometry PRIVATE mrcv::mrcv)

if(NOT is_top_level)
    win_copy_deps_to_target_dir(mrcv-geometry mrcv::mrcv)
endif()
