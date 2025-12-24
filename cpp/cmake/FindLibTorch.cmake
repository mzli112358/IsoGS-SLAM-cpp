# FindLibTorch.cmake
# Helper script to find LibTorch

set(LIBTORCH_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/libtorch" CACHE PATH "Root directory of LibTorch")

if(EXISTS "${LIBTORCH_ROOT_DIR}")
    set(TORCH_FOUND TRUE)
    set(TORCH_INCLUDE_DIRS "${LIBTORCH_ROOT_DIR}/include" "${LIBTORCH_ROOT_DIR}/include/torch/csrc/api/include")
    set(TORCH_LIBRARIES 
        "${LIBTORCH_ROOT_DIR}/lib/libtorch.so"
        "${LIBTORCH_ROOT_DIR}/lib/libtorch_cuda.so"
        "${LIBTORCH_ROOT_DIR}/lib/libc10.so"
        "${LIBTORCH_ROOT_DIR}/lib/libc10_cuda.so"
    )
    message(STATUS "Found LibTorch: ${LIBTORCH_ROOT_DIR}")
else()
    message(FATAL_ERROR "LibTorch not found. Please download from https://pytorch.org/ and extract to third_party/libtorch")
endif()

