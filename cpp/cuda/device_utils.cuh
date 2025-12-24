#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>  // 包含CUDA标准类型定义

/**
 * @brief CUDA设备工具宏和函数
 */

// 检查CUDA错误
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// 计算线程块数量
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// 安全的CUDA Kernel启动配置
inline void getLaunchConfig(int numel, int& num_blocks, int& threads_per_block) {
    threads_per_block = 256;
    num_blocks = CEIL_DIV(numel, threads_per_block);
}

// float3类型已由CUDA定义，直接使用
// make_float3函数也由CUDA定义

