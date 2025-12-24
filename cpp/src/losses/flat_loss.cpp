#include "losses/flat_loss.hpp"
#include <torch/torch.h>

// CUDA Kernel声明
extern "C" {
void launch_flat_loss(
    torch::Tensor scales,
    torch::Tensor grad_scales,
    torch::Tensor loss_buffer
);

void launch_flat_loss_only(
    torch::Tensor scales,
    torch::Tensor loss_buffer
);
}

namespace isogs {

float FlatLoss::compute(
    const Tensor& scales,
    Tensor& grad_scales)
{
    TORCH_CHECK(scales.is_cuda(), "Scales must be on CUDA");
    TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 3, "Scales must be [N, 3]");
    
    int num_gaussians = scales.size(0);
    
    // 初始化梯度（如果需要）
    if (!grad_scales.defined() || grad_scales.sizes() != scales.sizes()) {
        grad_scales = torch::zeros_like(scales);
    } else {
        grad_scales.zero_();
    }
    
    // 创建loss buffer
    auto loss_buffer = torch::zeros({num_gaussians}, torch::kFloat32).to(scales.device());
    
    // 调用CUDA Kernel
    launch_flat_loss(scales, grad_scales, loss_buffer);
    
    // 计算平均loss（安全地从CUDA tensor提取，先移到CPU）
    float loss = loss_buffer.mean().detach().cpu().item<float>();
    
    return loss;
}

float FlatLoss::computeLoss(const Tensor& scales)
{
    TORCH_CHECK(scales.is_cuda(), "Scales must be on CUDA");
    TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 3, "Scales must be [N, 3]");
    
    int num_gaussians = scales.size(0);
    auto loss_buffer = torch::zeros({num_gaussians}, torch::kFloat32).to(scales.device());
    
    launch_flat_loss_only(scales, loss_buffer);
    
    // 安全地从CUDA tensor提取，先移到CPU
    return loss_buffer.mean().detach().cpu().item<float>();
}

} // namespace isogs

