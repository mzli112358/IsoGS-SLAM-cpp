#include "losses/iso_loss.hpp"
#include <torch/torch.h>

// CUDA Kernel声明
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
);

void launch_iso_loss_only(
    torch::Tensor query_points,
    torch::Tensor means,
    torch::Tensor inv_covariances,
    torch::Tensor opacities,
    torch::Tensor knn_indices,
    torch::Tensor loss_buffer,
    float target_saturation
);
}

namespace isogs {

IsoSurfaceLoss::IsoSurfaceLoss(const Config& config)
    : config_(config)
{
}

float IsoSurfaceLoss::compute(
    const Tensor& means,
    const Tensor& inverse_covariances,
    const Tensor& opacities,
    Tensor& grad_means,
    Tensor& grad_opacities,
    const SpatialHash& spatial_hash)
{
    TORCH_CHECK(means.is_cuda(), "Means must be on CUDA");
    
    int num_gaussians = means.size(0);
    
    // 采样查询点（随机选择高斯中心作为查询点）
    int sample_size = std::min(config_.sample_size, num_gaussians);
    auto sample_indices = torch::randperm(num_gaussians, torch::kInt64).to(means.device()).slice(0, 0, sample_size);
    auto query_points = means.index_select(0, sample_indices);
    
    // 查询KNN
    auto knn_indices = spatial_hash.queryKNN(query_points, config_.K);
    
    // 初始化梯度
    if (!grad_means.defined() || grad_means.sizes() != means.sizes()) {
        grad_means = torch::zeros_like(means);
    } else {
        grad_means.zero_();
    }
    
    if (!grad_opacities.defined() || grad_opacities.sizes() != opacities.sizes()) {
        grad_opacities = torch::zeros_like(opacities);
    } else {
        grad_opacities.zero_();
    }
    
    // 创建loss buffer
    auto loss_buffer = torch::zeros({sample_size}, torch::kFloat32).to(means.device());
    
    // 调用CUDA Kernel
    launch_iso_loss(
        query_points,
        means,
        inverse_covariances,
        opacities,
        knn_indices,
        grad_means,
        grad_opacities,
        loss_buffer,
        config_.target_saturation
    );
    
    // 计算平均loss（安全地从CUDA tensor提取，先移到CPU）
    return loss_buffer.mean().detach().cpu().item<float>();
}

float IsoSurfaceLoss::computeLoss(
    const Tensor& means,
    const Tensor& inverse_covariances,
    const Tensor& opacities,
    const SpatialHash& spatial_hash)
{
    TORCH_CHECK(means.is_cuda(), "Means must be on CUDA");
    
    int num_gaussians = means.size(0);
    
    // 采样查询点（随机选择高斯中心作为查询点）
    int sample_size = std::min(config_.sample_size, num_gaussians);
    auto sample_indices = torch::randperm(num_gaussians, torch::kInt64).to(means.device()).slice(0, 0, sample_size);
    auto query_points = means.index_select(0, sample_indices);
    
    // 查询KNN
    auto knn_indices = spatial_hash.queryKNN(query_points, config_.K);
    
    // 创建loss buffer
    auto loss_buffer = torch::zeros({sample_size}, torch::kFloat32).to(means.device());
    
    // 调用仅计算Loss的CUDA Kernel
    launch_iso_loss_only(
        query_points,
        means,
        inverse_covariances,
        opacities,
        knn_indices,
        loss_buffer,
        config_.target_saturation
    );
    
    // 计算平均loss（安全地从CUDA tensor提取，先移到CPU）
    return loss_buffer.mean().detach().cpu().item<float>();
}

} // namespace isogs

