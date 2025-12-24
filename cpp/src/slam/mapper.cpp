#include "slam/mapper.hpp"
#include "utils/losses.hpp"
#include "utils/pointcloud.hpp"
#include "utils/keyframe_selection.hpp"
#include "losses/flat_loss.hpp"
#include "losses/iso_loss.hpp"
#include "utils/spatial_hash.hpp"
#include <torch/torch.h>
#include <algorithm>
#include <random>
#include <tuple>
#include <iostream>

namespace isogs {

Mapper::Mapper(const Config& config)
    : config_(config)
    , renderer_(std::make_unique<Renderer>())
    , optimizer_(std::make_unique<CudaAdamOptimizer>(config_.learning_rates))
    , optimizer_initialized_(false)
{
}

void Mapper::addKeyframe(const Keyframe& keyframe)
{
    keyframes_.push_back(keyframe);
    
    // 保持窗口大小
    if (static_cast<int>(keyframes_.size()) > config_.mapping_window_size) {
        keyframes_.erase(keyframes_.begin());
    }
}

void Mapper::optimize(
    GaussianModel& model,
    const Keyframe& current_frame)
{
    // 初始化优化器（首次调用时）
    if (!optimizer_initialized_) {
        initializeOptimizer(model);
        optimizer_initialized_ = true;
    }
    
    // 添加新高斯（如果需要）
    if (config_.add_new_gaussians) {
        addNewGaussians(model, current_frame);
        // 如果添加了新高斯，需要重新初始化优化器
        if (model.getActiveCount() > 0) {
            initializeOptimizer(model);
        }
    }
    
    // 优化循环
    for (int iter = 0; iter < config_.num_iters; ++iter) {
        // 使用基于重叠度的关键帧选择算法
        std::vector<int> selected_indices = selectKeyframes(1, current_frame);
        if (selected_indices.empty()) {
            // 如果没有选中任何关键帧，使用最后一个关键帧
            if (!keyframes_.empty()) {
                selected_indices.push_back(static_cast<int>(keyframes_.size()) - 1);
            }
        }
        
        int frame_idx = selected_indices[0];
        const auto& frame = (frame_idx == -1) ? current_frame : keyframes_[frame_idx];
        
        // 清零梯度
        optimizer_->zeroGrad();
        
        // 计算Loss和梯度
        std::map<std::string, Tensor> grads;
        auto losses = computeLoss(model, frame, grads);
        
        // 计算总Loss
        float total_loss = losses.at("im") * config_.loss_weights.at("im") +
                          losses.at("depth") * config_.loss_weights.at("depth") +
                          losses.at("flat") * config_.loss_weights.at("flat") +
                          losses.at("iso") * config_.loss_weights.at("iso");
        
        // 将梯度写入优化器
        if (grads.find("means3D") != grads.end() && optimizer_->hasParamGroup("means3D")) {
            auto& group = optimizer_->getParamGroup("means3D");
            group.grad.copy_(grads["means3D"]);
        }
        if (grads.find("log_scales") != grads.end() && optimizer_->hasParamGroup("log_scales")) {
            auto& group = optimizer_->getParamGroup("log_scales");
            group.grad.copy_(grads["log_scales"]);
        }
        if (grads.find("unnorm_rotations") != grads.end() && optimizer_->hasParamGroup("unnorm_rotations")) {
            auto& group = optimizer_->getParamGroup("unnorm_rotations");
            group.grad.copy_(grads["unnorm_rotations"]);
        }
        if (grads.find("logit_opacities") != grads.end() && optimizer_->hasParamGroup("logit_opacities")) {
            auto& group = optimizer_->getParamGroup("logit_opacities");
            group.grad.copy_(grads["logit_opacities"]);
        }
        if (grads.find("rgb_colors") != grads.end() && optimizer_->hasParamGroup("rgb_colors")) {
            auto& group = optimizer_->getParamGroup("rgb_colors");
            group.grad.copy_(grads["rgb_colors"]);
        }
        
        // 优化器更新
        optimizer_->step();
        
        // 将更新后的参数写回model
        Tensor updated_means = optimizer_->getParamGroup("means3D").params;
        Tensor updated_scales = optimizer_->getParamGroup("log_scales").params;
        Tensor updated_rotations = optimizer_->getParamGroup("unnorm_rotations").params;
        Tensor updated_opacity = optimizer_->getParamGroup("logit_opacities").params;
        Tensor updated_rgb = optimizer_->getParamGroup("rgb_colors").params;
        
        // 更新SH系数：只更新前3维（RGB），其余保持原值
        Tensor updated_sh = model.getSHCoeffs().clone();
        updated_sh.slice(1, 0, 3).copy_(updated_rgb);
        
        model.updateParams(updated_means, updated_scales, updated_rotations, updated_sh, updated_opacity);
        
        // Pruning（如果需要）
        if (config_.prune_gaussians && 
            iter >= config_.prune_start_after && 
            iter <= config_.prune_stop_after &&
            iter % config_.prune_every == 0) {
            Indices prune_indices;
            detectPruning(model, iter, prune_indices);
            
            if (!prune_indices.empty()) {
                model.prune(prune_indices);
                
                // 重新初始化优化器（因为高斯数量改变了）
                initializeOptimizer(model);
            }
        }
        
        if (config_.use_gaussian_splatting_densification && 
            iter >= config_.densify_start_after && 
            iter <= config_.densify_stop_after &&
            iter % config_.densify_every == 0) {
            // 检测需要densify的高斯
            Indices split_indices, clone_indices;
            detectDensification(model, grads, split_indices, clone_indices);
            
            // 执行densification
            if (!split_indices.empty() || !clone_indices.empty()) {
                model.densify(split_indices, clone_indices);
                
                // 重新初始化优化器（因为高斯数量改变了）
                initializeOptimizer(model);
            }
        }
    }
}

std::vector<int> Mapper::selectKeyframes(int num_keyframes, const Keyframe& current_frame) const
{
    if (keyframes_.empty()) {
        return {};
    }
    
    // 使用基于重叠度的关键帧选择算法
    auto intrinsics = current_frame.camera.getIntrinsics();
    auto current_w2c = current_frame.camera.getW2C();
    auto current_depth = current_frame.depth;  // [H, W]
    
    // 调用关键帧选择函数
    return keyframeSelectionOverlap(
        current_depth,
        current_w2c,
        intrinsics,
        keyframes_,
        num_keyframes,
        1600  // 默认采样1600个像素
    );
}

void Mapper::addNewGaussians(
    GaussianModel& model,
    const Keyframe& frame)
{
    // 基于silhouette的新高斯添加
    // 1. 渲染当前帧的silhouette
    // 2. 找到silhouette < threshold的区域
    // 3. 从这些区域提取点云
    // 4. 添加到模型中
    
    // TODO: 实现完整的silhouette渲染和点云提取
    // 目前简化实现：从深度图提取新点
    auto depth_mask = (frame.depth > 0.0f);
    
    // 提取新点云（简化版本）
    try {
        auto [points, colors, mean_sq_dist] = PointCloudUtils::extractPointCloud(
            frame.color.permute({1, 2, 0}),  // [3, H, W] -> [H, W, 3]
            frame.depth,
            frame.camera.getIntrinsics(),
            frame.camera.getW2C(),
            depth_mask,
            true
        );
        
        // 添加新高斯（使用当前帧ID作为timestep）
        model.addGaussians(points, colors, mean_sq_dist, static_cast<int>(frame.frame_id), config_.gaussian_distribution);
    } catch (const std::exception& e) {
        // 如果提取失败，跳过
        std::cerr << "Warning: Failed to add new Gaussians: " << e.what() << std::endl;
    }
}

void Mapper::initializeOptimizer(GaussianModel& model)
{
    int64_t N = model.getActiveCount();
    if (N == 0) return;
    
    // 获取参数（需要可写版本）
    // 使用 detach() 确保是叶子变量
    auto means = model.getMeans3D().detach().clone().requires_grad_(false);
    auto scales = model.getScales().detach().clone().requires_grad_(false);
    auto rotations = model.getRotations().detach().clone().requires_grad_(false);
    auto opacity = model.getOpacity().detach().clone().requires_grad_(false);
    auto sh_coeffs = model.getSHCoeffs().detach().clone().requires_grad_(false);
    
    // 只取RGB颜色（前3个SH系数）
    auto rgb_colors = sh_coeffs.slice(1, 0, 3);
    
    // 清空旧的参数组
    optimizer_ = std::make_unique<CudaAdamOptimizer>(config_.learning_rates);
    
    // 添加参数组
    optimizer_->addParamGroup("means3D", means, config_.learning_rates.at("means3D"));
    optimizer_->addParamGroup("log_scales", scales, config_.learning_rates.at("log_scales"));
    optimizer_->addParamGroup("unnorm_rotations", rotations, config_.learning_rates.at("unnorm_rotations"));
    optimizer_->addParamGroup("logit_opacities", opacity, config_.learning_rates.at("logit_opacities"));
    optimizer_->addParamGroup("rgb_colors", rgb_colors, config_.learning_rates.at("rgb_colors"));
}

std::map<std::string, float> Mapper::computeLoss(
    GaussianModel& model,
    const Keyframe& frame,
    std::map<std::string, Tensor>& grads)
{
    std::map<std::string, float> losses;
    
    // 获取参数（需要支持梯度）
    // 注意：clone() 会创建新的叶子变量，可以安全地设置 requires_grad_
    // 但为了确保是叶子变量，我们使用 detach() 然后 clone()
    auto means = model.getMeans3D().detach().clone().requires_grad_(true);
    auto scales = model.getScales().detach().clone().requires_grad_(true);
    auto rotations = model.getRotations().detach().clone().requires_grad_(true);
    auto opacity = model.getOpacity().detach().clone().requires_grad_(true);
    auto sh_coeffs = model.getSHCoeffs().detach().clone().requires_grad_(true);
    
    // 创建临时model用于渲染（使用可微参数）
    // 注意：这里我们需要直接使用可微参数进行渲染
    // 但由于Renderer需要GaussianModel，我们需要另一种方式
    
    // 方案：直接使用可微参数渲染
    // 但Renderer::renderRGBD需要GaussianModel，所以我们需要修改渲染方式
    // 或者创建一个临时的可微model
    
    // 简化方案：使用LibTorch的autograd计算RGB/Depth Loss的梯度
    // 对于Flat和Iso Loss，使用手动梯度
    
    // 渲染（使用原始model，但我们需要可微版本）
    Tensor rendered_color, rendered_depth;
    
    // 为了支持梯度，我们需要创建一个临时的可微model
    // 但这样会很复杂，所以我们采用另一种方式：
    // 1. RGB/Depth Loss使用LibTorch autograd（需要可微渲染）
    // 2. Flat/Iso Loss使用手动梯度
    
    // 先计算RGB/Depth Loss（使用原始model）
    std::tie(rendered_color, rendered_depth) = renderer_->renderRGBD(model, frame.camera);
    
    // RGB Loss - 需要可微
    auto color_diff = rendered_color - frame.color;
    Tensor rgb_loss_tensor = torch::abs(color_diff).mean();
    // 安全地从CUDA tensor提取标量值（先移到CPU）
    losses["im"] = rgb_loss_tensor.detach().cpu().item<float>();
    
    // Depth Loss - 需要可微
    auto depth_mask = (frame.depth > 0.0f);
    auto depth_diff = rendered_depth - frame.depth;
    auto masked_depth_diff = torch::abs(depth_diff) * depth_mask;
    auto valid_depth_count = depth_mask.sum();
    Tensor depth_loss;
    // 安全地从CUDA tensor提取标量值（先移到CPU）
    float valid_count_val = valid_depth_count.cpu().item<float>();
    if (valid_count_val > 0) {
        depth_loss = masked_depth_diff.sum() / valid_depth_count;
    } else {
        depth_loss = torch::tensor(0.0f, torch::kFloat32).to(frame.depth.device());
    }
    losses["depth"] = depth_loss.detach().cpu().item<float>();
    
    // 计算RGB/Depth Loss的总梯度
    Tensor total_rgbd_loss = rgb_loss_tensor * config_.loss_weights.at("im") +
                            depth_loss * config_.loss_weights.at("depth");
    
    // 注意：由于渲染器不支持可微，我们需要手动计算梯度
    // 这里先设置梯度为0，后续可以通过其他方式计算
    // 或者修改渲染器支持可微
    
    // Flat Loss和Iso Loss（使用手动梯度）
    try {
        // 验证scales tensor有效性
        if (!scales.defined() || scales.numel() == 0 || !scales.is_cuda()) {
            // 如果scales无效，跳过Flat Loss和Iso Loss
            losses["flat"] = 0.0f;
            losses["iso"] = 0.0f;
        } else {
            // 获取scales（从log空间转换），添加clamp避免exp溢出
            // 注意：scales可能包含NaN或Inf，需要先处理
            auto scales_valid = torch::nan_to_num(scales, 0.0f, 0.0f, 0.0f);
            torch::cuda::synchronize();  // 确保nan_to_num完成
            auto scales_clamped = torch::clamp(scales_valid, -10.0f, 10.0f);
            torch::cuda::synchronize();  // 确保clamp完成
            auto scales_exp = torch::exp(scales_clamped);
            torch::cuda::synchronize();  // 确保exp完成
            scales_exp = torch::clamp(scales_exp, 1e-5f, 100.0f);  // 添加上限避免Inf
            torch::cuda::synchronize();  // 确保clamp完成
            
            // 计算Flat Loss和梯度
            if (scales_exp.size(1) == 3) {
            Tensor grad_scales_flat;
            losses["flat"] = FlatLoss::compute(scales_exp, grad_scales_flat);
            
            // 将梯度从exp空间转换回log空间
            // dL/dlog_s = dL/ds * ds/dlog_s = dL/ds * s
            auto grad_scales_log = grad_scales_flat * scales_exp * config_.loss_weights.at("flat");
            grads["log_scales"] = grad_scales_log;
            
            // 计算Iso-Surface Loss（需要构建空间哈希表）
            auto opacity_sigmoid = torch::sigmoid(opacity);
            
            // 构建空间哈希表
            SpatialHash::Config hash_config;
            hash_config.cell_size = 0.1f;
            SpatialHash spatial_hash(hash_config);
            spatial_hash.build(means);
            
            // 构建逆协方差矩阵（简化版本）
            // TODO: 从rotations和scales构建完整的逆协方差矩阵
            auto inv_covariances = torch::eye(3, torch::kFloat32)
                .unsqueeze(0).expand({means.size(0), -1, -1}).to(means.device());
            
            // 计算Iso Loss和梯度
            Tensor grad_means_iso, grad_opacities_iso;
            IsoSurfaceLoss iso_loss;
            losses["iso"] = iso_loss.compute(means, inv_covariances, opacity_sigmoid, 
                                             grad_means_iso, grad_opacities_iso, spatial_hash);
            
            // 应用loss权重
            grad_means_iso = grad_means_iso * config_.loss_weights.at("iso");
            grad_opacities_iso = grad_opacities_iso * config_.loss_weights.at("iso");
            
            // 将opacity梯度从sigmoid空间转换回logit空间
            // dL/dlogit_alpha = dL/dalpha * dalpha/dlogit_alpha
            // dalpha/dlogit_alpha = alpha * (1 - alpha)
            auto alpha = opacity_sigmoid;
            auto grad_opacity_logit = grad_opacities_iso * alpha * (1.0f - alpha);
            
            // 累加梯度
            if (grads.find("means3D") == grads.end()) {
                grads["means3D"] = grad_means_iso;
            } else {
                grads["means3D"] = grads["means3D"] + grad_means_iso;
            }
            
            if (grads.find("logit_opacities") == grads.end()) {
                grads["logit_opacities"] = grad_opacity_logit;
            } else {
                grads["logit_opacities"] = grads["logit_opacities"] + grad_opacity_logit;
            }
        } else {
            losses["flat"] = 0.0f;
            losses["iso"] = 0.0f;
        }
        }  // 闭合 else 块
    } catch (const std::exception& e) {
        // 如果计算失败，设为0
        losses["flat"] = 0.0f;
        losses["iso"] = 0.0f;
        std::cerr << "Warning: Failed to compute Flat/Iso Loss: " << e.what() << std::endl;
    }
    
    // 初始化RGB/Depth Loss的梯度（目前设为0，因为渲染器不支持可微）
    // TODO: 后续需要修改渲染器支持可微，或者手动计算梯度
    // 注意：RGB/Depth Loss的梯度需要通过可微渲染计算，这里暂时设为0
    // 实际应用中，gsplat渲染器应该支持可微，或者我们需要手动计算梯度
    
    if (grads.find("means3D") == grads.end()) {
        grads["means3D"] = torch::zeros_like(means);
    }
    if (grads.find("log_scales") == grads.end()) {
        grads["log_scales"] = torch::zeros_like(scales);
    }
    if (grads.find("unnorm_rotations") == grads.end()) {
        grads["unnorm_rotations"] = torch::zeros_like(rotations);
    }
    if (grads.find("logit_opacities") == grads.end()) {
        grads["logit_opacities"] = torch::zeros_like(opacity);
    }
    // RGB颜色梯度（只针对前3个SH系数）
    if (grads.find("rgb_colors") == grads.end()) {
        grads["rgb_colors"] = torch::zeros({sh_coeffs.size(0), 3}, torch::kFloat32).to(sh_coeffs.device());
    }
    
    return losses;
}

void Mapper::detectDensification(
    GaussianModel& model,
    const std::map<std::string, Tensor>& grads,
    Indices& split_indices,
    Indices& clone_indices)
{
    split_indices.clear();
    clone_indices.clear();
    
    int64_t N = model.getActiveCount();
    if (N == 0) return;
    
    // 获取梯度（使用means3D的梯度作为means2D梯度的近似）
    Tensor grad_means;
    if (grads.find("means3D") != grads.end()) {
        grad_means = grads.at("means3D");
    } else {
        return;  // 没有梯度，无法densify
    }
    
    // 计算梯度大小（L2 norm）
    auto grad_norms = torch::norm(grad_means, 2, 1);  // [N]
    grad_norms = torch::nan_to_num(grad_norms, 0.0f, 0.0f, 0.0f);
    
    // 获取scales（log空间），安全地转换为exp空间，添加clamp避免数值溢出
    auto scales_log = model.getScales();  // [N, 3]
    
    // 验证tensor有效性（先检查基本属性，避免访问可能损坏的tensor）
    if (!scales_log.defined()) {
        return;  // 无效tensor，无法densify
    }
    
    // 安全地检查tensor属性（先移到CPU检查，避免CUDA访问错误）
    try {
        auto scales_log_cpu = scales_log.cpu();
        if (scales_log_cpu.numel() == 0) {
            return;  // 空tensor
        }
        
        // 检查维度（在CPU上检查是安全的）
        if (scales_log_cpu.dim() != 2 || scales_log_cpu.size(1) != 3) {
            return;  // tensor形状不正确
        }
    } catch (const std::exception& e) {
        // 如果访问tensor属性时出错，说明tensor已损坏
        std::cerr << "Warning: Error accessing scales_log tensor in detectDensification: " << e.what() << std::endl;
        return;
    } catch (...) {
        // 如果访问tensor属性时出错，说明tensor已损坏
        std::cerr << "Warning: Unknown error accessing scales_log tensor in detectDensification" << std::endl;
        return;
    }
    
    // 确保tensor在CUDA上
    if (!scales_log.is_cuda()) {
        return;
    }
    
    // 先处理NaN和Inf，然后clamp（确保在正确的设备上）
    try {
        auto scales_valid = torch::nan_to_num(scales_log, 0.0f, 0.0f, 0.0f);
        // 同步CUDA操作，确保之前的操作完成
        torch::cuda::synchronize();
        auto scales_clamped = torch::clamp(scales_valid, -10.0f, 10.0f);
        torch::cuda::synchronize();
        auto scales_exp = torch::exp(scales_clamped);  // [N, 3]
        torch::cuda::synchronize();
        auto max_scales_result = torch::max(scales_exp, 1);
        auto max_scales = std::get<0>(max_scales_result);  // [N]
        torch::cuda::synchronize();
    
        // 计算阈值
        float grad_thresh = config_.densify_grad_thresh;
        float scale_thresh = 0.01f * config_.scene_radius;
    
        // 克隆条件：梯度 >= 阈值 且 scale <= 阈值
        auto to_clone_mask = (grad_norms >= grad_thresh) & (max_scales <= scale_thresh);
        auto to_clone_indices_tensor = torch::nonzero(to_clone_mask);
        
        // 检查是否有需要克隆的高斯（安全地检查numel）
        int64_t num_clone_elements = 0;
        try {
            num_clone_elements = to_clone_indices_tensor.numel();
        } catch (...) {
            return;  // 如果访问tensor属性失败，说明tensor已损坏
        }
        
        if (num_clone_elements > 0) {
            // 安全地检查维度（先移到CPU）
            auto to_clone_indices_cpu_check = to_clone_indices_tensor.cpu();
            if (to_clone_indices_cpu_check.dim() == 2 && to_clone_indices_cpu_check.size(1) == 1) {
                to_clone_indices_tensor = to_clone_indices_tensor.squeeze(1);
            }
            
            // 安全地从CUDA tensor提取索引（先移到CPU）
            auto to_clone_indices_cpu = to_clone_indices_tensor.cpu();
            int64_t num_clone = to_clone_indices_cpu.size(0);
            for (int64_t i = 0; i < num_clone; ++i) {
                clone_indices.push_back(to_clone_indices_cpu[i].item<int64_t>());
            }
        }
    
        // 分裂条件：梯度 >= 阈值 且 scale > 阈值
        auto to_split_mask = (grad_norms >= grad_thresh) & (max_scales > scale_thresh);
        auto to_split_indices_tensor = torch::nonzero(to_split_mask);
        
        // 检查是否有需要分裂的高斯（安全地检查numel）
        int64_t num_split_elements = 0;
        try {
            num_split_elements = to_split_indices_tensor.numel();
        } catch (...) {
            return;  // 如果访问tensor属性失败，说明tensor已损坏
        }
        
        if (num_split_elements > 0) {
            // 如果tensor是[N, 1]形状，需要squeeze
            if (to_split_indices_tensor.dim() == 2 && to_split_indices_tensor.size(1) == 1) {
                to_split_indices_tensor = to_split_indices_tensor.squeeze(1);
            }
            
            // 安全地从CUDA tensor提取索引（先移到CPU）
            auto to_split_indices_cpu = to_split_indices_tensor.cpu();
            int64_t num_split = to_split_indices_cpu.size(0);
            for (int64_t i = 0; i < num_split; ++i) {
                split_indices.push_back(to_split_indices_cpu[i].item<int64_t>());
            }
        }
    } catch (const std::exception& e) {
        // 如果处理过程中出错，记录错误并返回
        std::cerr << "Warning: Error in detectDensification: " << e.what() << std::endl;
        return;
    } catch (...) {
        // 如果处理过程中出错，记录错误并返回
        std::cerr << "Warning: Unknown error in detectDensification" << std::endl;
        return;
    }
}

void Mapper::detectPruning(
    GaussianModel& model,
    int iter,
    Indices& prune_indices)
{
    prune_indices.clear();
    
    int64_t N = model.getActiveCount();
    if (N == 0) return;
    
    // 获取opacity（sigmoid空间）
    auto opacity_sigmoid = torch::sigmoid(model.getOpacity()).squeeze(1);  // [N]
    
    // 确定opacity阈值
    float opacity_thresh;
    if (iter == config_.prune_stop_after) {
        opacity_thresh = config_.final_removal_opacity_threshold;
    } else {
        opacity_thresh = config_.removal_opacity_threshold;
    }
    
    // 低opacity的高斯需要prune
    auto low_opacity_mask = opacity_sigmoid < opacity_thresh;
    
    // 太大的高斯也需要prune（在特定迭代后）
    Tensor too_big_mask;
    if (iter >= config_.remove_big_after) {
        auto scales_log = model.getScales();  // [N, 3]
        // 安全地计算exp，添加clamp避免数值溢出
        // 先处理NaN和Inf，然后clamp
        auto scales_valid = torch::nan_to_num(scales_log, 0.0f, 0.0f, 0.0f);
        auto scales_clamped = torch::clamp(scales_valid, -10.0f, 10.0f);
        auto scales_exp = torch::exp(scales_clamped);  // [N, 3]
        auto max_scales_result = torch::max(scales_exp, 1);
        auto max_scales = std::get<0>(max_scales_result);  // [N]
        float big_thresh = 0.1f * config_.scene_radius;
        too_big_mask = max_scales > big_thresh;
    } else {
        too_big_mask = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(model.getDevice()));
    }
    
    // 合并mask
    auto to_remove_mask = low_opacity_mask | too_big_mask;
    auto to_remove_indices_tensor = torch::nonzero(to_remove_mask);
    
    // 检查是否有需要prune的高斯（安全地检查numel）
    int64_t num_remove_elements = 0;
    try {
        num_remove_elements = to_remove_indices_tensor.numel();
    } catch (...) {
        return;  // 如果访问tensor属性失败，说明tensor已损坏
    }
    
    if (num_remove_elements > 0) {
        // 安全地检查维度（先移到CPU）
        auto to_remove_indices_cpu_check = to_remove_indices_tensor.cpu();
        if (to_remove_indices_cpu_check.dim() == 2 && to_remove_indices_cpu_check.size(1) == 1) {
            to_remove_indices_tensor = to_remove_indices_tensor.squeeze(1);
        }
        
        // 安全地从CUDA tensor提取索引（先移到CPU）
        auto to_remove_indices_cpu = to_remove_indices_tensor.cpu();
        int64_t num_remove = to_remove_indices_cpu.size(0);
        for (int64_t i = 0; i < num_remove; ++i) {
            prune_indices.push_back(to_remove_indices_cpu[i].item<int64_t>());
        }
    }
}

} // namespace isogs

