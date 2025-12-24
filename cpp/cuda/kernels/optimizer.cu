#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>

/**
 * @brief CUDA Kernel: Adam优化器更新
 * 
 * 实现Adam算法的参数更新：
 * m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
 * v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
 * m_hat = m_t / (1 - beta1^t)
 * v_hat = v_t / (1 - beta2^t)
 * param = param - lr * m_hat / (sqrt(v_hat) + eps)
 */
__global__ void adam_update_kernel(
    float* params,
    const float* grads,
    float* m,
    float* v,
    int numel,
    float lr,
    float beta1,
    float beta2,
    float beta1_pow_t,
    float beta2_pow_t,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    
    // 更新一阶矩估计
    float grad_val = grads[idx];
    m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad_val;
    
    // 更新二阶矩估计
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad_val * grad_val;
    
    // 计算偏差修正后的估计
    float m_hat = m[idx] / (1.0f - beta1_pow_t);
    float v_hat = v[idx] / (1.0f - beta2_pow_t);
    
    // 更新参数
    params[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

// 前向声明
extern "C" {
/**
 * @brief 调用Adam更新Kernel的包装函数
 */
void launch_adam_update(
    torch::Tensor params,
    torch::Tensor grads,
    torch::Tensor m,
    torch::Tensor v,
    float lr,
    float beta1,
    float beta2,
    int step_count,
    float eps = 1e-8f
) {
    // 检查输入
    TORCH_CHECK(params.is_cuda(), "Params must be on CUDA");
    TORCH_CHECK(grads.is_cuda(), "Grads must be on CUDA");
    TORCH_CHECK(m.is_cuda(), "M must be on CUDA");
    TORCH_CHECK(v.is_cuda(), "V must be on CUDA");
    TORCH_CHECK(params.sizes() == grads.sizes(), "Params and grads must have same shape");
    
    int numel = params.numel();
    
    // 计算beta的t次方
    float beta1_pow_t = powf(beta1, static_cast<float>(step_count + 1));
    float beta2_pow_t = powf(beta2, static_cast<float>(step_count + 1));
    
    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (numel + threads_per_block - 1) / threads_per_block;
    
    adam_update_kernel<<<num_blocks, threads_per_block>>>(
        params.data_ptr<float>(),
        grads.data_ptr<float>(),
        m.data_ptr<float>(),
        v.data_ptr<float>(),
        numel,
        lr,
        beta1,
        beta2,
        beta1_pow_t,
        beta2_pow_t,
        eps
    );
    
    // 检查CUDA错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));
    }
}
} // extern "C"

