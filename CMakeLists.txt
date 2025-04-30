# Минимальная версия CMake
cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)

# Общая информация о проекте
project(mrcv
    VERSION 1.0.0
    DESCRIPTION "Marine Robotics Ccomputer Vision C++ Open Source Library"
    LANGUAGES CXX)

###############################################################################
# Общие настройки проекта
###############################################################################

# Для разработки под VS
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
endif(MSVC)

include(cmake/utils.cmake)
include(GNUInstallDirs)

string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)

# MRCV_SHARED_LIBS option (undefined by default) can be used to force shared/static build
option(MRCV_BUILD_TESTS "Build mrcv tests" OFF)
option(MRCV_BUILD_EXAMPLES "Build mrcv examples" ON)
option(MRCV_INSTALL "Generate target for installing mrcv" ${is_top_level})
set_if_undefined(MRCV_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/mrcv" CACHE STRING "Install path for mrcv package-related CMake files")

if(DEFINED MRCV_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ${MRCV_SHARED_LIBS})
endif()

set(CMAKE_BUILD_TYPE Release)

if(NOT DEFINED CMAKE_BUILD_TYPE AND NOT DEFINED CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set_if_undefined(CMAKE_CXX_VISIBILITY_PRESET hidden)
set_if_undefined(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

add_library(mrcv)
add_library(mrcv::mrcv ALIAS mrcv)

###############################################################################
# Определение зависимостей библиотеки
###############################################################################
set(CMAKE_PREFIX_PATH "C:/libtorch-12.4/Release/;C:/yaml-cpp/")

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(yaml-cpp REQUIRED)

# Диагностический вывод о наличии библиотек
#if (OpenCV_FOUND)
#    message(STATUS "OpenCV library found!")
#else ()
#    message(FATAL_ERROR "Could not locate OpenCV" \n)
#endif()

#if (Torch_FOUND)
#    message(STATUS "Torch library found!")
#else ()
#    message(FATAL_ERROR "Could not locate Torch" \n)
#endif()

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

include_directories( ${OpenCV_INCLUDE_DIRS}; ${Torch_INCLUDE_DIRS}; ${YAML_CPP_INCLUDE_DIR} )
link_directories( ${OpenCV_LIB_DIR}; ${Torch_LIB_DIR}; ${YAML_CPP_LIBRARY_DIR} )

set(MRCV_CUDA false) # Ключ сборки с/без CUDA

IF (MRCV_CUDA)
    set(ALL_LIBS
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
        yaml-cpp::yaml-cpp
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_cudart_static_LIBRARY}
    )
    target_link_libraries( mrcv ${ALL_LIBS} )
ELSE()
    target_link_libraries( mrcv ${OpenCV_LIBS}; ${TORCH_LIBRARIES}; yaml-cpp::yaml-cpp )
ENDIF()
# target_link_libraries( mrcv ${OpenCV_LIBS}; ${TORCH_LIBRARIES} )
###############################################################################
# Исходники библиотеки
###############################################################################

include(GenerateExportHeader)
set(export_file_name "export_shared.h")

if(NOT BUILD_SHARED_LIBS)
    set(export_file_name "export_static.h")
endif()

generate_export_header(mrcv EXPORT_FILE_NAME include/mrcv/${export_file_name})

set(sources
    include/mrcv/export.h
    include/mrcv/mrcv.h
    include/mrcv/mrcv-common.h
    include/mrcv/mrcv-json.hpp
    include/mrcv/mrcv-segmentation.h
    include/mrcv/mrcv-yolov5.h
    include/mrcv/mrcv-detector.h
    include/mrcv/tinystr.h
    include/mrcv/tinyxml.h
    include/mrcv/mrcv-sensorsfusion.h
    src/mrcv.cpp
    src/mrcv-utility.cpp
    src/mrcv-imgcollection.cpp
    src/mrcv-calibration.cpp
    src/mrcv-segmentation.cpp
    src/mrcv-objcourse.cpp
    src/mrcv-yolov5.cpp
    src/mrcv-detector.cpp
    src/tinystr.cpp
    src/tinyxml.cpp
    src/tinyxlmerror.cpp
    src/tinyxmlparser.cpp
    src/mrcv-vae.cpp
    src/mrcv-roi.cpp 
    src/mrcv-predictor.cpp 
    src/mrcv-optimizer.cpp
    src/mrcv-morphologyImage.cpp
    src/mrcv-compareImages.cpp
    src/mrcv-sensorsfusion.cpp
    src/mrcv-extractfeature.cpp)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${sources})

###############################################################################
# Определение таргетов библиотеки
###############################################################################

include(CMakePackageConfigHelpers)

target_sources(mrcv PRIVATE ${sources} "src/mrcv-utility.cpp" "src/mrcv-imgcollection.cpp" "src/mrcv-calibration.cpp"  "include/mrcv/mrcv-common.h" "include/mrcv/mrcv-json.hpp"  "include/mrcv/mrcv-segmentation.h" "src/mrcv-segmentation.cpp" "include/mrcv/mrcv-yolov5.h" "src/mrcv-yolov5.cpp" "src/mrcv-objcourse.cpp" "src/mrcv-imgpreprocessing.cpp" "include/mrcv/mrcv-yolov5.h" "include/mrcv/mrcv-detector.h" "src/mrcv-yolov5.cpp" "src/mrcv-clustering.cpp" "include/mrcv/mrcv-vae.h" "src/mrcv-vae.cpp" "src/mrcv-detector.cpp" "include/mrcv/tinystr.h" "include/mrcv/tinyxml.h" "src/tinystr.cpp" "src/tinyxml.cpp" "src/mrcv-augmentation.cpp" "src/mrcv-disparity.cpp"  "src/mrcv-roi.cpp" "src/mrcv-predictor.cpp" "src/mrcv-optimizer.cpp" "src/mrcv-3dscene.cpp" "src/mrcv-morphologyImage.cpp" "src/mrcv-compareImages.cpp" "include/mrcv/mrcv-sensorsfusion.h" "src/mrcv-sensorsfusion.cpp" "src/mrcv-extractfeature.cpp")

target_compile_definitions(mrcv PUBLIC "$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:MRCV_STATIC_DEFINE>")

target_include_directories(mrcv
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>")

set_target_properties(mrcv PROPERTIES
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION})

if(MRCV_INSTALL AND NOT CMAKE_SKIP_INSTALL_RULES)
    configure_package_config_file(cmake/mrcv-config.cmake.in mrcv-config.cmake
        INSTALL_DESTINATION "${MRCV_INSTALL_CMAKEDIR}")

    write_basic_package_version_file(mrcv-config-version.cmake
        COMPATIBILITY SameMajorVersion)

    install(TARGETS mrcv EXPORT mrcv_export
        RUNTIME COMPONENT mrcv
        LIBRARY COMPONENT mrcv NAMELINK_COMPONENT mrcv-dev
        ARCHIVE COMPONENT mrcv-dev
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
    install(DIRECTORY include/
        TYPE INCLUDE
        COMPONENT mrcv-dev)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/mrcv/${export_file_name}"
        COMPONENT mrcv-dev
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mrcv")

    set(targets_file "mrcv-shared-targets.cmake")

    if(NOT BUILD_SHARED_LIBS)
        set(targets_file "mrcv-static-targets.cmake")
    endif()

    install(EXPORT mrcv_export
        COMPONENT mrcv-dev
        FILE "${targets_file}"
        DESTINATION "${MRCV_INSTALL_CMAKEDIR}"
        NAMESPACE mrcv::)

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/mrcv-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/mrcv-config-version.cmake"
        COMPONENT mrcv-dev
        DESTINATION "${MRCV_INSTALL_CMAKEDIR}")

    if(MSVC)
        set(pdb_file "")
        set(pdb_file_destination "")

        if(BUILD_SHARED_LIBS)
            set(pdb_file "$<TARGET_PDB_FILE:mrcv>")
            set(pdb_file_destination "${CMAKE_INSTALL_BINDIR}")
        else()
            # TARGET_PDB_FILE does not work for pdb file generated for static library build, determining it manually
            set(pdb_file "$<TARGET_FILE_DIR:mrcv>/$<TARGET_FILE_PREFIX:mrcv>$<TARGET_FILE_BASE_NAME:mrcv>.pdb")
            set(pdb_file_destination "${CMAKE_INSTALL_LIBDIR}")
        endif()

        install(FILES "${pdb_file}"
            COMPONENT mrcv-dev
            CONFIGURATIONS Debug RelWithDebInfo
            DESTINATION "${pdb_file_destination}"
            OPTIONAL)
    endif()
endif()

###############################################################################
# Определение дополнительных таргетов
###############################################################################

if(MRCV_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(MRCV_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
