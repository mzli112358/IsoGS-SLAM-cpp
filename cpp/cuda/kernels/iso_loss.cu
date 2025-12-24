#include <cuda_runtime.h>
#include <torch/torch.h>
#include "../device_utils.cuh"

/**
 * @brief CUDA Kernel: 计算密度值
 * 
 * D(p) = sum(alpha_j * exp(-0.5 * delta^T * Sigma_j^{-1} * delta))
 */
__device__ float compute_density(
    float px, float py, float pz,
    const float* means,
    const float* inv_covs,
    const float* opacities,
    const int64_t* neighbor_indices,
    int K
) {
    float density = 0.0f;
    
    for (int k = 0; k < K; ++k) {
        int64_t idx = neighbor_indices[k];
        if (idx < 0) break;
        
        float alpha = opacities[idx];
        if (alpha < 1e-6f) continue;
        
        // 计算delta = p - mu_j
        float dx = px - means[idx * 3 + 0];
        float dy = py - means[idx * 3 + 1];
        float dz = pz - means[idx * 3 + 2];
        
        // 计算delta^T * inv_cov * delta
        const float* inv_cov = inv_covs + idx * 9;  // [3, 3]矩阵
        
        float quad_form = 
            dx * (inv_cov[0] * dx + inv_cov[1] * dy + inv_cov[2] * dz) +
            dy * (inv_cov[3] * dx + inv_cov[4] * dy + inv_cov[5] * dz) +
            dz * (inv_cov[6] * dx + inv_cov[7] * dy + inv_cov[8] * dz);
        
        // 计算exp项
        float exp_term = expf(-0.5f * quad_form);
        
        density += alpha * exp_term;
    }
    
    return density;
}

/**
 * @brief CUDA Kernel: 计算Iso-Surface Loss和梯度
 */
__global__ void iso_loss_kernel(
    const float* query_points,      // [M, 3]
    const float* means,             // [N, 3]
    const float* inv_covariances,   // [N, 3, 3]
    const float* opacities,         // [N, 1]
    const int64_t* knn_indices,     // [M, K]
    float* grad_means,              // [N, 3] (输出)
    float* grad_opacities,          // [N, 1] (输出)
    float* loss_buffer,             // [M] (输出)
    int num_queries,
    int K,
    float target_saturation,
    float inv_M                     // 1.0f / M
) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= num_queries) return;
    
    float px = query_points[query_idx * 3 + 0];
    float py = query_points[query_idx * 3 + 1];
    float pz = query_points[query_idx * 3 + 2];
    
    // 获取KNN索引
    const int64_t* neighbors = knn_indices + query_idx * K;
    
    // 计算密度
    float density = compute_density(
        px, py, pz,
        means,
        inv_covariances,
        opacities,
        neighbors,
        K
    );
    
    // 计算loss
    float diff = density - target_saturation;
    float loss = diff * diff;
    loss_buffer[query_idx] = loss;
    
    // 计算梯度
    float loss_grad = 2.0f * diff * inv_M;
    
    // 对每个邻居高斯计算梯度
    for (int k = 0; k < K; ++k) {
        int64_t gaussian_idx = neighbors[k];
        if (gaussian_idx < 0) break;
        
        float alpha = opacities[gaussian_idx];
        if (alpha < 1e-6f) continue;
        
        // 计算delta
        float dx = px - means[gaussian_idx * 3 + 0];
        float dy = py - means[gaussian_idx * 3 + 1];
        float dz = pz - means[gaussian_idx * 3 + 2];
        
        // 计算exp项
        const float* inv_cov = inv_covariances + gaussian_idx * 9;
        float quad_form = 
            dx * (inv_cov[0] * dx + inv_cov[1] * dy + inv_cov[2] * dz) +
            dy * (inv_cov[3] * dx + inv_cov[4] * dy + inv_cov[5] * dz) +
            dz * (inv_cov[6] * dx + inv_cov[7] * dy + inv_cov[8] * dz);
        float exp_term = expf(-0.5f * quad_form);
        
        // 梯度w.r.t. opacity
        float grad_alpha = loss_grad * exp_term;
        atomicAdd(&grad_opacities[gaussian_idx], grad_alpha);
        
        // 梯度w.r.t. means
        // 推导: dL/dmu_j = loss_grad * alpha * exp_term * inv_cov * delta
        // 其中 loss_grad = 2 * (D - target) / M
        // delta = p - mu_j
        // inv_cov是逆协方差矩阵
        float grad_coeff = loss_grad * alpha * exp_term;
        
        // 计算 inv_cov * delta
        float inv_cov_delta_x = 
            inv_cov[0] * dx + inv_cov[1] * dy + inv_cov[2] * dz;
        float inv_cov_delta_y = 
            inv_cov[3] * dx + inv_cov[4] * dy + inv_cov[5] * dz;
        float inv_cov_delta_z = 
            inv_cov[6] * dx + inv_cov[7] * dy + inv_cov[8] * dz;
        
        // 计算梯度并累加
        // dL/dmu_j = loss_grad * alpha * exp_term * inv_cov * delta_j
        float grad_mu_x = grad_coeff * inv_cov_delta_x;
        float grad_mu_y = grad_coeff * inv_cov_delta_y;
        float grad_mu_z = grad_coeff * inv_cov_delta_z;
        
        atomicAdd(&grad_means[gaussian_idx * 3 + 0], grad_mu_x);
        atomicAdd(&grad_means[gaussian_idx * 3 + 1], grad_mu_y);
        atomicAdd(&grad_means[gaussian_idx * 3 + 2], grad_mu_z);
    }
}

/**
 * @brief CUDA Kernel: 仅计算Iso-Surface Loss（不计算梯度）
 */
__global__ void iso_loss_only_kernel(
    const float* query_points,      // [M, 3]
    const float* means,             // [N, 3]
    const float* inv_covariances,   // [N, 3, 3]
    const float* opacities,         // [N, 1]
    const int64_t* knn_indices,     // [M, K]
    float* loss_buffer,             // [M] (输出)
    int num_queries,
    int K,
    float target_saturation
) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= num_queries) return;
    
    float px = query_points[query_idx * 3 + 0];
    float py = query_points[query_idx * 3 + 1];
    float pz = query_points[query_idx * 3 + 2];
    
    // 获取KNN索引
    const int64_t* neighbors = knn_indices + query_idx * K;
    
    // 计算密度
    float density = compute_density(
        px, py, pz,
        means,
        inv_covariances,
        opacities,
        neighbors,
        K
    );
    
    // 计算loss（不计算梯度）
    float diff = density - target_saturation;
    float loss = diff * diff;
    loss_buffer[query_idx] = loss;
}

// 包装函数
extern "C" {
void launch_iso_loss(
    torch::Tensor query_points,
    torch::Tensor means,
    torch::Tensor inv_covariances,
    torch::Tensor opacities,
    torch::Tensor knn_indices,
    torch::Tensor grad_means,
    torch::Tensor grad_opacities,
    torch::Tensor loss_buffer,
    float target_saturation
) {
    int num_queries = query_points.size(0);
    int K = knn_indices.size(1);
    float inv_M = 1.0f / static_cast<float>(num_queries);
    
    int threads = 256;
    int blocks = (num_queries + threads - 1) / threads;
    
    iso_loss_kernel<<<blocks, threads>>>(
        query_points.data_ptr<float>(),
        means.data_ptr<float>(),
        inv_covariances.data_ptr<float>(),
        opacities.data_ptr<float>(),
        knn_indices.data_ptr<int64_t>(),
        grad_means.data_ptr<float>(),
        grad_opacities.data_ptr<float>(),
        loss_buffer.data_ptr<float>(),
        num_queries,
        K,
        target_saturation,
        inv_M
    );
}

void launch_iso_loss_only(
    torch::Tensor query_points,
    torch::Tensor means,
    torch::Tensor inv_covariances,
    torch::Tensor opacities,
    torch::Tensor knn_indices,
    torch::Tensor loss_buffer,
    float target_saturation
) {
    int num_queries = query_points.size(0);
    int K = knn_indices.size(1);
    
    int threads = 256;
    int blocks = (num_queries + threads - 1) / threads;
    
    iso_loss_only_kernel<<<blocks, threads>>>(
        query_points.data_ptr<float>(),
        means.data_ptr<float>(),
        inv_covariances.data_ptr<float>(),
        opacities.data_ptr<float>(),
        knn_indices.data_ptr<int64_t>(),
        loss_buffer.data_ptr<float>(),
        num_queries,
        K,
        target_saturation
    );
}
} // extern "C"
