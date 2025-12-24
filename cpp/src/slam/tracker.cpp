#include "slam/tracker.hpp"
#include "utils/losses.hpp"
#include "utils/eval.hpp"
#include <torch/torch.h>
#include <cmath>

namespace isogs {

Tracker::Tracker(const Config& config)
    : config_(config)
    , renderer_(std::make_unique<Renderer>())
{
}

float Tracker::optimizePose(
    GaussianModel& model,
    Camera& camera,
    const Tensor& gt_color,
    const Tensor& gt_depth)
{
    // 创建可优化的位姿参数（w2c矩阵）
    auto w2c = camera.getW2C().clone().requires_grad_(true);
    
    // 创建优化器（使用Adam优化器）
    std::vector<torch::optim::OptimizerParamGroup> param_groups;
    std::vector<torch::Tensor> params = {w2c};
    auto adam_opts = std::make_unique<torch::optim::AdamOptions>(config_.learning_rates.at("cam_trans"));
    param_groups.emplace_back(params, std::move(adam_opts));
    
    torch::optim::Adam optimizer(param_groups);
    
    float best_loss = 1e20f;
    Tensor best_w2c = w2c.clone();
    
    // 优化循环
    for (int iter = 0; iter < config_.num_iters; ++iter) {
        optimizer.zero_grad();
        
        // 创建临时相机（使用当前w2c，支持梯度）
        Camera temp_camera(camera.getIntrinsics(), w2c, 
                          camera.getWidth(), camera.getHeight());
        
        // 可微渲染
        auto [rendered_color, rendered_depth] = renderer_->renderRGBD(model, temp_camera);
        
        // 检查渲染结果是否在计算图中
        // 注意：gsplat的CUDA kernel可能不支持自动微分
        // 如果rendered_color没有梯度，我们需要跳过tracking或使用其他方法
        if (!rendered_color.requires_grad() && w2c.requires_grad()) {
            // gsplat的CUDA kernel可能不支持自动微分
            // 在这种情况下，我们暂时跳过tracking，使用GT poses
            std::cerr << "Warning: rendered_color does not require grad, gsplat may not support autograd. Skipping tracking." << std::endl;
            // 直接使用初始w2c，不进行优化
            break;
        }
        
        // 计算RGB Loss（L1）
        auto rgb_diff = rendered_color - gt_color;
        auto rgb_loss_tensor = torch::abs(rgb_diff).mean();
        
        // 计算Depth Loss（L1，仅在有效像素上）
        auto depth_mask = (gt_depth > 0.0f);
        auto depth_diff = rendered_depth - gt_depth;
        auto masked_depth_diff = torch::abs(depth_diff) * depth_mask;
        auto valid_depth_count = depth_mask.sum();
        Tensor depth_loss_tensor;
        
        // 安全地从CUDA tensor提取标量值（先移到CPU）
        float valid_count_val = valid_depth_count.cpu().item<float>();
        if (valid_count_val > 0) {
            depth_loss_tensor = masked_depth_diff.sum() / valid_depth_count;
        } else {
            // 如果没有有效深度，创建一个零tensor
            depth_loss_tensor = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(gt_depth.device()));
        }
        
        // 总Loss
        auto total_loss_tensor = rgb_loss_tensor * config_.loss_weights.at("im") +
                                depth_loss_tensor * config_.loss_weights.at("depth");
        
        // 检查total_loss_tensor是否有梯度
        if (!total_loss_tensor.requires_grad()) {
            // 如果total_loss_tensor没有梯度，可能是因为所有输入都没有梯度
            // 这种情况下，我们跳过反向传播
            std::cerr << "Warning: total_loss_tensor does not require grad, skipping backward pass" << std::endl;
            continue;
        }
        
        // 安全地从CUDA tensor提取标量值（先移到CPU，并检查有效性）
        float total_loss_val = total_loss_tensor.detach().cpu().item<float>();
        
        // 检查loss是否有效（非NaN，非inf）
        if (std::isnan(total_loss_val) || std::isinf(total_loss_val)) {
            std::cerr << "Warning: Invalid loss value (NaN or Inf), skipping iteration" << std::endl;
            continue;
        }
        
        // 反向传播
        total_loss_tensor.backward();
        
        // 优化器更新
        optimizer.step();
        
        // 保存最佳位姿
        if (total_loss_val < best_loss) {
            best_loss = total_loss_val;
            best_w2c = w2c.clone().detach();
        }
    }
    
    // 更新相机位姿（使用最佳位姿）
    camera.setW2C(best_w2c);
    
    return best_loss;
}

std::map<std::string, float> Tracker::computeLoss(
    const GaussianModel& model,
    const Camera& camera,
    const Tensor& gt_color,
    const Tensor& gt_depth)
{
    std::map<std::string, float> losses;
    
    // 渲染
    auto [rendered_color, rendered_depth] = renderer_->renderRGBD(model, camera);
    
    // RGB Loss
    auto rgb_loss = computeRGBLoss(rendered_color, gt_color);
    losses["im"] = rgb_loss.detach().cpu().item<float>();
    
    // Depth Loss
    auto depth_mask = (gt_depth > 0.0f);
    auto depth_loss = computeDepthLoss(rendered_depth, gt_depth, depth_mask);
    losses["depth"] = depth_loss.detach().cpu().item<float>();
    
    return losses;
}

Tensor Tracker::computeRGBLoss(const Tensor& rendered, const Tensor& gt)
{
    // L1 Loss
    auto l1_loss = torch::abs(rendered - gt).mean();
    
    // SSIM Loss (1 - SSIM，因为SSIM越大越好，Loss越小越好)
    // 注意：SSIM计算可能不支持梯度，所以ssim_loss是一个标量tensor
    float ssim_val = Evaluator::computeSSIM(rendered, gt);
    auto ssim_loss = torch::tensor(1.0f - ssim_val, torch::kFloat32).to(rendered.device());
    // 不需要手动设置requires_grad_，因为这是新创建的tensor（叶子变量）
    // 如果rendered支持梯度，ssim_loss会通过l1_loss在计算图中
    
    // 组合Loss: L1 + SSIM
    // 权重可以根据需要调整
    float ssim_weight = 0.2f;  // SSIM权重
    return l1_loss + ssim_weight * ssim_loss;
}

Tensor Tracker::computeDepthLoss(const Tensor& rendered, const Tensor& gt, const Tensor& mask)
{
    // L1 Loss on valid pixels
    auto masked_diff = torch::abs(rendered - gt) * mask;
    auto valid_count = mask.sum();
    
    // 安全地从CUDA tensor提取标量值（先移到CPU）
    float valid_count_val = valid_count.cpu().item<float>();
    if (valid_count_val > 0) {
        return masked_diff.sum() / valid_count;
    } else {
        return torch::tensor(0.0f, torch::TensorOptions().dtype(torch::kFloat32).device(rendered.device()));
    }
}

} // namespace isogs

