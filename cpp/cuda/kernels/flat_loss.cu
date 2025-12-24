#include <cuda_runtime.h>
#include <torch/torch.h>
#include "../device_utils.cuh"

/**
 * @brief CUDA Kernel: 计算Flat Loss和梯度
 * 
 * Loss = mean(min(s_x, s_y, s_z))
 * 
 * 梯度：
 * - 如果s_x是最小值：dL/ds_x = 1/N, dL/ds_y = 0, dL/ds_z = 0
 * - 如果s_y是最小值：dL/ds_x = 0, dL/ds_y = 1/N, dL/ds_z = 0
 * - 如果s_z是最小值：dL/ds_x = 0, dL/ds_y = 0, dL/ds_z = 1/N
 */
__global__ void flat_loss_kernel(
    const float* scales,      // [N, 3]
    float* grad_scales,       // [N, 3] (输出，会被累加)
    float* loss_buffer,       // [N] (每个高斯的loss贡献)
    int num_gaussians,
    float inv_N               // 1.0f / N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;
    
    float sx = scales[idx * 3 + 0];
    float sy = scales[idx * 3 + 1];
    float sz = scales[idx * 3 + 2];
    
    // 找到最小轴
    float s_min = fminf(fminf(sx, sy), sz);
    
    // 计算loss贡献
    loss_buffer[idx] = s_min;
    
    // 计算梯度
    float grad_x = 0.0f;
    float grad_y = 0.0f;
    float grad_z = 0.0f;
    
    if (s_min == sx) {
        grad_x = inv_N;
    } else if (s_min == sy) {
        grad_y = inv_N;
    } else {
        grad_z = inv_N;
    }
    
    // 写入梯度（累加）
    grad_scales[idx * 3 + 0] += grad_x;
    grad_scales[idx * 3 + 1] += grad_y;
    grad_scales[idx * 3 + 2] += grad_z;
}

/**
 * @brief 仅计算Loss的Kernel（不计算梯度）
 */
__global__ void flat_loss_only_kernel(
    const float* scales,
    float* loss_buffer,
    int num_gaussians
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;
    
    float sx = scales[idx * 3 + 0];
    float sy = scales[idx * 3 + 1];
    float sz = scales[idx * 3 + 2];
    
    float s_min = fminf(fminf(sx, sy), sz);
    loss_buffer[idx] = s_min;
}

// 包装函数
extern "C" {
void launch_flat_loss(
    torch::Tensor scales,
    torch::Tensor grad_scales,
    torch::Tensor loss_buffer
) {
    int num_gaussians = scales.size(0);
    float inv_N = 1.0f / static_cast<float>(num_gaussians);
    
    int threads = 256;
    int blocks = (num_gaussians + threads - 1) / threads;
    
    flat_loss_kernel<<<blocks, threads>>>(
        scales.data_ptr<float>(),
        grad_scales.data_ptr<float>(),
        loss_buffer.data_ptr<float>(),
        num_gaussians,
        inv_N
    );
}

void launch_flat_loss_only(
    torch::Tensor scales,
    torch::Tensor loss_buffer
) {
    int num_gaussians = scales.size(0);
    
    int threads = 256;
    int blocks = (num_gaussians + threads - 1) / threads;
    
    flat_loss_only_kernel<<<blocks, threads>>>(
        scales.data_ptr<float>(),
        loss_buffer.data_ptr<float>(),
        num_gaussians
    );
}
} // extern "C"

