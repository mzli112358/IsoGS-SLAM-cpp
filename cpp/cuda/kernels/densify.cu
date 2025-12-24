#include <cuda_runtime.h>
#include <torch/torch.h>
#include "../device_utils.cuh"

/**
 * @brief CUDA Kernel: 高斯分裂
 * 
 * 将一个高斯分裂成两个高斯
 */
__global__ void split_gaussian_kernel(
    const float* src_means,
    const float* src_scales,
    const float* src_rotations,
    const float* src_sh_coeffs,
    const float* src_opacity,
    float* dst_means,
    float* dst_scales,
    float* dst_rotations,
    float* dst_sh_coeffs,
    float* dst_opacity,
    const int64_t* split_indices,
    int num_splits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_splits) return;
    
    int64_t src_idx = split_indices[idx];
    
    // 复制原始数据
    for (int i = 0; i < 3; ++i) {
        dst_means[idx * 3 + i] = src_means[src_idx * 3 + i];
        dst_scales[idx * 3 + i] = src_scales[src_idx * 3 + i];
    }
    
    for (int i = 0; i < 4; ++i) {
        dst_rotations[idx * 4 + i] = src_rotations[src_idx * 4 + i];
    }
    
    for (int i = 0; i < 48; ++i) {
        dst_sh_coeffs[idx * 48 + i] = src_sh_coeffs[src_idx * 48 + i];
    }
    
    dst_opacity[idx] = src_opacity[src_idx];
    
    // 分裂操作：缩小scale，创建两个高斯
    // 第一个高斯：缩小scale
    float scale_factor = 1.6f;  // 分裂后的scale因子
    for (int i = 0; i < 3; ++i) {
        dst_scales[idx * 3 + i] = src_scales[src_idx * 3 + i] - logf(scale_factor);
    }
    
    // 第二个高斯会在下一个位置创建
}

/**
 * @brief CUDA Kernel: 高斯克隆
 */
__global__ void clone_gaussian_kernel(
    const float* src_means,
    const float* src_scales,
    const float* src_rotations,
    const float* src_sh_coeffs,
    const float* src_opacity,
    float* dst_means,
    float* dst_scales,
    float* dst_rotations,
    float* dst_sh_coeffs,
    float* dst_opacity,
    const int64_t* clone_indices,
    int num_clones
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_clones) return;
    
    int64_t src_idx = clone_indices[idx];
    
    // 完全复制
    for (int i = 0; i < 3; ++i) {
        dst_means[idx * 3 + i] = src_means[src_idx * 3 + i];
        dst_scales[idx * 3 + i] = src_scales[src_idx * 3 + i];
    }
    
    for (int i = 0; i < 4; ++i) {
        dst_rotations[idx * 4 + i] = src_rotations[src_idx * 4 + i];
    }
    
    for (int i = 0; i < 48; ++i) {
        dst_sh_coeffs[idx * 48 + i] = src_sh_coeffs[src_idx * 48 + i];
    }
    
    dst_opacity[idx] = src_opacity[src_idx];
}

// 包装函数
extern "C" {
void launch_split_gaussians(
    torch::Tensor src_means,
    torch::Tensor src_scales,
    torch::Tensor src_rotations,
    torch::Tensor src_sh_coeffs,
    torch::Tensor src_opacity,
    torch::Tensor dst_means,
    torch::Tensor dst_scales,
    torch::Tensor dst_rotations,
    torch::Tensor dst_sh_coeffs,
    torch::Tensor dst_opacity,
    torch::Tensor split_indices
) {
    int num_splits = split_indices.size(0);
    if (num_splits == 0) return;
    
    int threads = 256;
    int blocks = (num_splits + threads - 1) / threads;
    
    split_gaussian_kernel<<<blocks, threads>>>(
        src_means.data_ptr<float>(),
        src_scales.data_ptr<float>(),
        src_rotations.data_ptr<float>(),
        src_sh_coeffs.data_ptr<float>(),
        src_opacity.data_ptr<float>(),
        dst_means.data_ptr<float>(),
        dst_scales.data_ptr<float>(),
        dst_rotations.data_ptr<float>(),
        dst_sh_coeffs.data_ptr<float>(),
        dst_opacity.data_ptr<float>(),
        split_indices.data_ptr<int64_t>(),
        num_splits
    );
}

void launch_clone_gaussians(
    torch::Tensor src_means,
    torch::Tensor src_scales,
    torch::Tensor src_rotations,
    torch::Tensor src_sh_coeffs,
    torch::Tensor src_opacity,
    torch::Tensor dst_means,
    torch::Tensor dst_scales,
    torch::Tensor dst_rotations,
    torch::Tensor dst_sh_coeffs,
    torch::Tensor dst_opacity,
    torch::Tensor clone_indices
) {
    int num_clones = clone_indices.size(0);
    if (num_clones == 0) return;
    
    int threads = 256;
    int blocks = (num_clones + threads - 1) / threads;
    
    clone_gaussian_kernel<<<blocks, threads>>>(
        src_means.data_ptr<float>(),
        src_scales.data_ptr<float>(),
        src_rotations.data_ptr<float>(),
        src_sh_coeffs.data_ptr<float>(),
        src_opacity.data_ptr<float>(),
        dst_means.data_ptr<float>(),
        dst_scales.data_ptr<float>(),
        dst_rotations.data_ptr<float>(),
        dst_sh_coeffs.data_ptr<float>(),
        dst_opacity.data_ptr<float>(),
        clone_indices.data_ptr<int64_t>(),
        num_clones
    );
}
} // extern "C"

