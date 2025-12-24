# Third-party libraries CMakeLists.txt

# gsplat library (as git submodule)
set(GSPLAT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/gsplat")
set(GSPLAT_CSRC_DIR "${GSPLAT_DIR}/gsplat/cuda/csrc")
set(GSPLAT_INCLUDE_DIR "${GSPLAT_DIR}/gsplat/cuda/include")
set(GSPLAT_GLM_DIR "${GSPLAT_CSRC_DIR}/third_party/glm")

# Collect gsplat source files
file(GLOB GSPLAT_CUDA_SOURCES
    "${GSPLAT_CSRC_DIR}/*.cu"
)
file(GLOB GSPLAT_CPP_SOURCES
    "${GSPLAT_CSRC_DIR}/*.cpp"
)

# Remove ext.cpp as it's for Python bindings
list(REMOVE_ITEM GSPLAT_CPP_SOURCES "${GSPLAT_CSRC_DIR}/ext.cpp")

# Create gsplat library
add_library(gsplat_lib STATIC
    ${GSPLAT_CUDA_SOURCES}
    ${GSPLAT_CPP_SOURCES}
)

target_include_directories(gsplat_lib PUBLIC
    ${GSPLAT_INCLUDE_DIR}
    ${GSPLAT_CSRC_DIR}  # 添加csrc目录以包含Rasterization.h
    ${GSPLAT_GLM_DIR}
)

target_compile_features(gsplat_lib PUBLIC cxx_std_17)

# CUDA settings for gsplat
set_property(TARGET gsplat_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET gsplat_lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# CUDA compile flags (match gsplat's setup.py)
target_compile_options(gsplat_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        --expt-relaxed-constexpr
        -diag-suppress=20012,186
    >
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# Link against Torch (gsplat depends on it)
target_link_libraries(gsplat_lib PUBLIC
    ${TORCH_LIBRARIES}
)

# cnpy library (for .npz file I/O)
set(CNPY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cnpy")
if(EXISTS "${CNPY_DIR}/cnpy.h")
    add_library(cnpy_lib STATIC
        ${CNPY_DIR}/cnpy.cpp
    )
    
    target_include_directories(cnpy_lib PUBLIC
        ${CNPY_DIR}
    )
    
    target_compile_features(cnpy_lib PUBLIC cxx_std_11)
    
    # Link against zlib (required for npz compression)
    find_package(ZLIB REQUIRED)
    target_link_libraries(cnpy_lib PUBLIC ZLIB::ZLIB)
endif()
