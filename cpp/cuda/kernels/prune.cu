#include <cuda_runtime.h>
#include <torch/torch.h>
#include "../device_utils.cuh"

/**
 * @brief CUDA Kernel: 基于不透明度和梯度的剪枝
 */
__global__ void prune_gaussians_kernel(
    const float* opacity,
    const float* opacity_grad,
    bool* prune_mask,
    int num_gaussians,
    float opacity_threshold,
    float grad_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;
    
    float op = opacity[idx];
    float grad = fabsf(opacity_grad[idx]);
    
    // 剪枝条件：不透明度低且梯度小
    bool should_prune = (op < opacity_threshold) && (grad < grad_threshold);
    prune_mask[idx] = should_prune;
}

