cmake_minimum_required(VERSION 3.20)
project(IsoGS-SLAM-CPP VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA Configuration
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GPU Architecture (RTX 4090D: Compute Capability 8.9)
set(CMAKE_CUDA_ARCHITECTURES "89")

# CUDA Compiler Flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# C++ Compiler Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find LibTorch
set(LIBTORCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch")
if(EXISTS "${LIBTORCH_DIR}/share/cmake/Torch")
    # Use standard Torch find_package (if LibTorch was installed properly)
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${LIBTORCH_DIR}")
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
else()
    # Try custom FindLibTorch.cmake
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    include(FindLibTorch)
    if(NOT TORCH_FOUND)
        message(FATAL_ERROR 
            "LibTorch not found!\n"
            "Please download LibTorch from https://pytorch.org/get-started/locally/\n"
            "For CUDA 12.1, download: https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip\n"
            "Extract to: ${LIBTORCH_DIR}\n"
            "Or set LIBTORCH_ROOT_DIR environment variable to LibTorch installation path."
        )
    endif()
endif()

# Find CUDA
find_package(CUDA REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# gsplat include directories (added after third_party is configured)
# This will be set by third_party/CMakeLists.txt

# Third-party libraries
add_subdirectory(third_party)

# Source files
file(GLOB_RECURSE CORE_SOURCES
    "src/core/*.cpp"
)

file(GLOB_RECURSE SLAM_SOURCES
    "src/slam/*.cpp"
)

file(GLOB_RECURSE DATASET_SOURCES
    "src/datasets/*.cpp"
)

file(GLOB_RECURSE UTILS_SOURCES
    "src/utils/*.cpp"
)

file(GLOB_RECURSE RENDERING_SOURCES
    "src/rendering/*.cpp"
)

file(GLOB_RECURSE MESHING_SOURCES
    "src/meshing/*.cpp"
)

file(GLOB_RECURSE VIZ_SOURCES
    "src/viz/*.cpp"
)

file(GLOB_RECURSE TOOLS_SOURCES
    "src/tools/*.cpp"
)

file(GLOB_RECURSE LOSSES_SOURCES
    "src/losses/*.cpp"
)

file(GLOB_RECURSE CUDA_SOURCES
    "cuda/kernels/*.cu"
)

# CUDA Kernel compilation
set(CUDA_SEPARABLE_COMPILATION ON)

# Main executable
add_executable(isogs_slam
    main.cpp
    ${CORE_SOURCES}
    ${SLAM_SOURCES}
    ${DATASET_SOURCES}
    ${UTILS_SOURCES}
    ${RENDERING_SOURCES}
    ${MESHING_SOURCES}
    ${LOSSES_SOURCES}
    ${CUDA_SOURCES}
)

target_link_libraries(isogs_slam
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    ${OpenCV_LIBS}
    gsplat_lib
    cnpy_lib
)

# Add gsplat include directories
target_include_directories(isogs_slam PRIVATE
    $<TARGET_PROPERTY:gsplat_lib,INTERFACE_INCLUDE_DIRECTORIES>
)

target_compile_features(isogs_slam PUBLIC cxx_std_17)
target_compile_definitions(isogs_slam PRIVATE _GLIBCXX_USE_CXX11_ABI=0)

# CUDA specific settings
set_property(TARGET isogs_slam PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET isogs_slam PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Gradient check tool
add_executable(grad_check
    tests/grad_check.cpp
    ${CORE_SOURCES}
    ${SLAM_SOURCES}
    ${DATASET_SOURCES}
    ${UTILS_SOURCES}
    ${RENDERING_SOURCES}
    ${MESHING_SOURCES}
    ${LOSSES_SOURCES}
    ${CUDA_SOURCES}
)

target_link_libraries(grad_check
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    ${OpenCV_LIBS}
    gsplat_lib
    cnpy_lib
)

target_include_directories(grad_check PRIVATE
    $<TARGET_PROPERTY:gsplat_lib,INTERFACE_INCLUDE_DIRECTORIES>
)

target_compile_features(grad_check PUBLIC cxx_std_17)
target_compile_definitions(grad_check PRIVATE _GLIBCXX_USE_CXX11_ABI=0)

set_property(TARGET grad_check PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET grad_check PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Export PLY tool
add_executable(export_ply
    src/tools/export_ply.cpp
    ${CORE_SOURCES}
    ${DATASET_SOURCES}
    ${UTILS_SOURCES}
    ${RENDERING_SOURCES}
    ${CUDA_SOURCES}
)

target_link_libraries(export_ply
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    ${OpenCV_LIBS}
    gsplat_lib
    cnpy_lib
)

target_include_directories(export_ply PRIVATE
    $<TARGET_PROPERTY:gsplat_lib,INTERFACE_INCLUDE_DIRECTORIES>
)

target_compile_features(export_ply PUBLIC cxx_std_17)
target_compile_definitions(export_ply PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
set_property(TARGET export_ply PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET export_ply PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Gaussian Splatting offline training tool
add_executable(gaussian_splatting
    src/tools/gaussian_splatting.cpp
    ${CORE_SOURCES}
    ${SLAM_SOURCES}
    ${DATASET_SOURCES}
    ${UTILS_SOURCES}
    ${RENDERING_SOURCES}
    ${MESHING_SOURCES}
    ${LOSSES_SOURCES}
    ${CUDA_SOURCES}
)

target_link_libraries(gaussian_splatting
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    ${OpenCV_LIBS}
    gsplat_lib
    cnpy_lib
)

target_include_directories(gaussian_splatting PRIVATE
    $<TARGET_PROPERTY:gsplat_lib,INTERFACE_INCLUDE_DIRECTORIES>
)

target_compile_features(gaussian_splatting PUBLIC cxx_std_17)
target_compile_definitions(gaussian_splatting PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
set_property(TARGET gaussian_splatting PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET gaussian_splatting PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Post-SplaTAM optimization tool
add_executable(post_splatam_opt
    src/tools/post_splatam_opt.cpp
    ${CORE_SOURCES}
    ${SLAM_SOURCES}
    ${DATASET_SOURCES}
    ${UTILS_SOURCES}
    ${RENDERING_SOURCES}
    ${MESHING_SOURCES}
    ${LOSSES_SOURCES}
    ${CUDA_SOURCES}
)

target_link_libraries(post_splatam_opt
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    ${OpenCV_LIBS}
    gsplat_lib
    cnpy_lib
)

target_include_directories(post_splatam_opt PRIVATE
    $<TARGET_PROPERTY:gsplat_lib,INTERFACE_INCLUDE_DIRECTORIES>
)

target_compile_features(post_splatam_opt PUBLIC cxx_std_17)
target_compile_definitions(post_splatam_opt PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
set_property(TARGET post_splatam_opt PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET post_splatam_opt PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Installation
install(TARGETS isogs_slam grad_check export_ply gaussian_splatting post_splatam_opt DESTINATION bin)

