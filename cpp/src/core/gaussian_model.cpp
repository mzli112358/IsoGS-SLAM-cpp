#include "core/gaussian_model.hpp"
#include <torch/torch.h>
#include <cassert>
#include <cmath>

// CUDA Kernel声明
extern "C" {
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
);
}

namespace isogs {

GaussianModel::GaussianModel(int64_t max_gaussians, Device device)
    : max_gaussians_(max_gaussians)
    , active_count_(0)
    , device_(device)
{
    // 预分配所有显存
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device_);
    
    means3D_ = torch::zeros({max_gaussians_, 3}, options);
    scales_ = torch::zeros({max_gaussians_, 3}, options);
    rotations_ = torch::zeros({max_gaussians_, 4}, options);
    sh_coeffs_ = torch::zeros({max_gaussians_, 48}, options);
    opacity_ = torch::zeros({max_gaussians_, 1}, options);
    timestep_ = torch::zeros({max_gaussians_}, options);
    alive_mask_ = torch::zeros({max_gaussians_}, options.dtype(torch::kBool));
    
    // 初始化旋转为单位四元数 [1, 0, 0, 0]
    rotations_.slice(1, 0, 1).fill_(1.0f);
}

void GaussianModel::initializeFromPointCloud(
    const Tensor& points,
    const Tensor& colors,
    const Tensor& mean_sq_dist,
    const std::string& gaussian_distribution)
{
    int64_t num_points = points.size(0);
    assert(num_points <= max_gaussians_);
    assert(points.size(1) == 3);
    assert(colors.size(0) == num_points && colors.size(1) == 3);
    assert(mean_sq_dist.size(0) == num_points);
    
    // 复制位置
    means3D_.slice(0, 0, num_points).copy_(points);
    
    // 初始化scale (log空间)
    // base_log_scale = log(sqrt(mean_sq_dist)) for each point
    auto base_log_scale = torch::log(torch::sqrt(mean_sq_dist) + 1e-8f).unsqueeze(1); // [N, 1]
    
    // [IsoGS] Force 3D (anisotropic) initialization for flatness regularization
    // Even if config says "isotropic", we initialize as 3D with small random perturbation
    Tensor log_scales;
    if (gaussian_distribution == "isotropic") {
        // Convert to 3D by tiling and adding small random perturbation to break symmetry
        // This allows flatness regularization to work even with "isotropic" config
        auto log_scales_base = base_log_scale.expand({num_points, 3}); // [N, 3]
        // Add small random perturbation (std=0.01) to prevent identical gradients
        auto perturbation = torch::randn_like(log_scales_base) * 0.01f;
        log_scales = log_scales_base + perturbation;
        std::cout << "[IsoGS] Forced 3D initialization: Converting isotropic to anisotropic for flatness regularization." << std::endl;
    } else if (gaussian_distribution == "anisotropic") {
        log_scales = base_log_scale.expand({num_points, 3}); // [N, 3]
    } else {
        throw std::runtime_error("Unknown gaussian_distribution: " + gaussian_distribution);
    }
    scales_.slice(0, 0, num_points).copy_(log_scales);
    
    // 初始化旋转为单位四元数（已经在构造函数中设置）
    
    // 初始化SH系数（RGB用前3个系数，其余为0）
    // SH系数格式: [DC, Y, Z, X, ...] for RGB
    auto sh_rgb = colors.unsqueeze(1); // [N, 1, 3]
    sh_coeffs_.slice(0, 0, num_points).slice(1, 0, 3).copy_(sh_rgb.squeeze(1));
    
    // 初始化opacity (logit空间，初始化为0，sigmoid(0) = 0.5)
    opacity_.slice(0, 0, num_points).fill_(0.0f);
    
    // 初始化timestep为0（第一帧）
    timestep_.slice(0, 0, num_points).fill_(0.0f);
    
    // 更新活跃掩码
    active_count_ = num_points;
    alive_mask_.slice(0, 0, num_points).fill_(true);
    alive_mask_.slice(0, num_points, max_gaussians_).fill_(false);
}

void GaussianModel::addGaussians(
    const Tensor& points,
    const Tensor& colors,
    const Tensor& mean_sq_dist,
    int time_idx,
    const std::string& gaussian_distribution)
{
    int64_t num_new = points.size(0);
    ensureCapacity(num_new);
    
    int64_t start_idx = active_count_;
    int64_t end_idx = active_count_ + num_new;
    
    // 复制位置
    means3D_.slice(0, start_idx, end_idx).copy_(points);
    
    // 初始化scale (log空间)
    // base_log_scale = log(sqrt(mean_sq_dist)) for each point
    auto base_log_scale = torch::log(torch::sqrt(mean_sq_dist) + 1e-8f).unsqueeze(1); // [N, 1]
    
    // [IsoGS] Force 3D (anisotropic) initialization for flatness regularization
    Tensor log_scales;
    if (gaussian_distribution == "isotropic") {
        // Convert to 3D by tiling and adding small random perturbation to break symmetry
        auto log_scales_base = base_log_scale.expand({num_new, 3}); // [N, 3]
        // Add small random perturbation (std=0.01) to prevent identical gradients
        auto perturbation = torch::randn_like(log_scales_base) * 0.01f;
        log_scales = log_scales_base + perturbation;
    } else if (gaussian_distribution == "anisotropic") {
        log_scales = base_log_scale.expand({num_new, 3}); // [N, 3]
    } else {
        throw std::runtime_error("Unknown gaussian_distribution: " + gaussian_distribution);
    }
    scales_.slice(0, start_idx, end_idx).copy_(log_scales);
    
    // 初始化旋转为单位四元数
    rotations_.slice(0, start_idx, end_idx).slice(1, 0, 1).fill_(1.0f);
    rotations_.slice(0, start_idx, end_idx).slice(1, 1, 4).fill_(0.0f);
    
    // 初始化SH系数
    sh_coeffs_.slice(0, start_idx, end_idx).slice(1, 0, 3).copy_(colors);
    sh_coeffs_.slice(0, start_idx, end_idx).slice(1, 3, 48).fill_(0.0f);
    
    // 初始化opacity
    opacity_.slice(0, start_idx, end_idx).fill_(0.0f);
    
    // 初始化timestep为当前帧索引
    timestep_.slice(0, start_idx, end_idx).fill_(static_cast<float>(time_idx));
    
    // 更新活跃掩码
    active_count_ = end_idx;
    alive_mask_.slice(0, start_idx, end_idx).fill_(true);
}

void GaussianModel::densify(const Indices& split_indices, const Indices& clone_indices)
{
    int64_t num_splits = static_cast<int64_t>(split_indices.size());
    int64_t num_clones = static_cast<int64_t>(clone_indices.size());
    int64_t num_new = num_splits * 2 + num_clones;  // 每个split产生2个新高斯
    
    if (num_new == 0) return;
    
    // 确保有足够空间
    ensureCapacity(num_new);
    
    int64_t start_idx = active_count_;
    int64_t end_idx = active_count_ + num_new;
    
    // 准备目标张量（新高斯的位置）
    auto new_means = means3D_.slice(0, start_idx, end_idx);
    auto new_scales = scales_.slice(0, start_idx, end_idx);
    auto new_rotations = rotations_.slice(0, start_idx, end_idx);
    auto new_sh_coeffs = sh_coeffs_.slice(0, start_idx, end_idx);
    auto new_opacity = opacity_.slice(0, start_idx, end_idx);
    auto new_timestep = timestep_.slice(0, start_idx, end_idx);
    
    // 获取源张量（当前活跃高斯）
    auto src_means = means3D_.slice(0, 0, active_count_);
    auto src_scales = scales_.slice(0, 0, active_count_);
    auto src_rotations = rotations_.slice(0, 0, active_count_);
    auto src_sh_coeffs = sh_coeffs_.slice(0, 0, active_count_);
    auto src_opacity = opacity_.slice(0, 0, active_count_);
    auto src_timestep = timestep_.slice(0, 0, active_count_);
    
    // 处理克隆（先处理，因为位置连续）
    if (num_clones > 0) {
        auto clone_indices_tensor = torch::zeros({num_clones}, torch::kInt64).to(device_);
        for (int64_t i = 0; i < num_clones; ++i) {
            clone_indices_tensor[i] = clone_indices[i];
        }
        
        auto clone_means = new_means.slice(0, 0, num_clones);
        auto clone_scales = new_scales.slice(0, 0, num_clones);
        auto clone_rotations = new_rotations.slice(0, 0, num_clones);
        auto clone_sh_coeffs = new_sh_coeffs.slice(0, 0, num_clones);
        auto clone_opacity = new_opacity.slice(0, 0, num_clones);
        auto clone_timestep = new_timestep.slice(0, 0, num_clones);
        
        // 调用CUDA Kernel
        launch_clone_gaussians(
            src_means, src_scales, src_rotations, src_sh_coeffs, src_opacity,
            clone_means, clone_scales, clone_rotations, clone_sh_coeffs, clone_opacity,
            clone_indices_tensor
        );
        
        // 复制timestep（CUDA kernel不支持timestep，手动复制）
        for (int64_t i = 0; i < num_clones; ++i) {
            int64_t src_idx = clone_indices[i];
            clone_timestep[i] = src_timestep[src_idx];
        }
    }
    
    // 处理分裂（每个split产生2个新高斯）
    if (num_splits > 0) {
        auto split_indices_tensor = torch::zeros({num_splits}, torch::kInt64).to(device_);
        for (int64_t i = 0; i < num_splits; ++i) {
            split_indices_tensor[i] = split_indices[i];
        }
        
        // 分裂产生2倍的高斯
        int64_t split_start = num_clones;
        int64_t split_end = split_start + num_splits * 2;
        
        // 先复制原始高斯到新位置（两次）
        for (int64_t i = 0; i < num_splits; ++i) {
            int64_t src_idx = split_indices[i];
            int64_t dst_idx1 = split_start + i * 2;
            int64_t dst_idx2 = split_start + i * 2 + 1;
            
            // 复制到第一个位置
            new_means.slice(0, dst_idx1, dst_idx1 + 1).copy_(src_means.slice(0, src_idx, src_idx + 1));
            new_scales.slice(0, dst_idx1, dst_idx1 + 1).copy_(src_scales.slice(0, src_idx, src_idx + 1));
            new_rotations.slice(0, dst_idx1, dst_idx1 + 1).copy_(src_rotations.slice(0, src_idx, src_idx + 1));
            new_sh_coeffs.slice(0, dst_idx1, dst_idx1 + 1).copy_(src_sh_coeffs.slice(0, src_idx, src_idx + 1));
            new_opacity.slice(0, dst_idx1, dst_idx1 + 1).copy_(src_opacity.slice(0, src_idx, src_idx + 1));
            new_timestep[dst_idx1 - start_idx] = src_timestep[src_idx];
            
            // 复制到第二个位置
            new_means.slice(0, dst_idx2, dst_idx2 + 1).copy_(src_means.slice(0, src_idx, src_idx + 1));
            new_scales.slice(0, dst_idx2, dst_idx2 + 1).copy_(src_scales.slice(0, src_idx, src_idx + 1));
            new_rotations.slice(0, dst_idx2, dst_idx2 + 1).copy_(src_rotations.slice(0, src_idx, src_idx + 1));
            new_sh_coeffs.slice(0, dst_idx2, dst_idx2 + 1).copy_(src_sh_coeffs.slice(0, src_idx, src_idx + 1));
            new_opacity.slice(0, dst_idx2, dst_idx2 + 1).copy_(src_opacity.slice(0, src_idx, src_idx + 1));
            new_timestep[dst_idx2 - start_idx] = src_timestep[src_idx];
        }
        
        // 缩小scale（分裂后的scale = 原scale / (0.8 * 2)）
        float scale_factor = 0.8f * 2.0f;  // 0.8 * num_to_split_into
        auto split_scales = new_scales.slice(0, split_start, split_end);
        split_scales.copy_(split_scales - std::log(scale_factor));
        
        // 在局部坐标系中采样新位置
        // 使用scale作为std，在旋转后的坐标系中采样
        auto split_means = new_means.slice(0, split_start, split_end);
        auto split_rotations = new_rotations.slice(0, split_start, split_end);
        
        // 获取原始scales（exp空间），添加clamp避免数值溢出
        auto src_scales_clamped = torch::clamp(src_scales, -10.0f, 10.0f);
        auto original_scales_exp = torch::exp(src_scales_clamped);
        
        // 为每个分裂的高斯采样新位置
        for (int64_t i = 0; i < num_splits; ++i) {
            int64_t src_idx = split_indices[i];
            int64_t dst_idx1 = split_start + i * 2;
            int64_t dst_idx2 = split_start + i * 2 + 1;
            
            // 获取原始scale和rotation
            auto scale_exp = original_scales_exp.slice(0, src_idx, src_idx + 1);  // [1, 3]
            auto rotation = src_rotations.slice(0, src_idx, src_idx + 1);  // [1, 4]
            
            // 采样偏移（在局部坐标系中，使用scale作为std）
            auto offset1 = torch::randn({1, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_)) * scale_exp;
            auto offset2 = torch::randn({1, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(device_)) * scale_exp;
            
            // 将偏移转换到世界坐标系（需要旋转矩阵）
            // 简化：直接添加到means（实际应该用旋转矩阵）
            auto mean1 = new_means.slice(0, dst_idx1, dst_idx1 + 1);
            auto mean2 = new_means.slice(0, dst_idx2, dst_idx2 + 1);
            mean1.copy_(mean1 + offset1);
            mean2.copy_(mean2 + offset2);
        }
    }
    
    // 更新活跃掩码和计数
    active_count_ = end_idx;
    alive_mask_.slice(0, start_idx, end_idx).fill_(true);
}

void GaussianModel::prune(const Indices& prune_indices)
{
    if (prune_indices.empty()) return;
    
    // 创建prune掩码
    auto prune_mask = torch::zeros({max_gaussians_}, 
        torch::TensorOptions().dtype(torch::kBool).device(device_));
    
    for (auto idx : prune_indices) {
        if (idx >= 0 && idx < active_count_) {
            prune_mask[idx] = true;
        }
    }
    
    // 更新活跃掩码
    alive_mask_ = alive_mask_ & (~prune_mask);
    
    // 压缩：将活跃高斯移动到前面
    // 使用torch::masked_select和重新排列
    auto active_indices = torch::nonzero(alive_mask_).squeeze(1);
    
    if (active_indices.size(0) > 0) {
        // 重新排列数据
        means3D_.slice(0, 0, active_indices.size(0)) = means3D_.index_select(0, active_indices);
        scales_.slice(0, 0, active_indices.size(0)) = scales_.index_select(0, active_indices);
        rotations_.slice(0, 0, active_indices.size(0)) = rotations_.index_select(0, active_indices);
        sh_coeffs_.slice(0, 0, active_indices.size(0)) = sh_coeffs_.index_select(0, active_indices);
        opacity_.slice(0, 0, active_indices.size(0)) = opacity_.index_select(0, active_indices);
        timestep_.slice(0, 0, active_indices.size(0)) = timestep_.index_select(0, active_indices);
        
        // 更新活跃数量
        active_count_ = active_indices.size(0);
        
        // 重置后面的掩码
        alive_mask_.slice(0, 0, active_count_).fill_(true);
        alive_mask_.slice(0, active_count_, max_gaussians_).fill_(false);
    } else {
        active_count_ = 0;
        alive_mask_.fill_(false);
    }
}

void GaussianModel::ensureCapacity(int64_t additional_count)
{
    if (active_count_ + additional_count > max_gaussians_) {
        throw std::runtime_error(
            "GaussianModel: Exceeded maximum capacity. "
            "Requested: " + std::to_string(active_count_ + additional_count) +
            ", Max: " + std::to_string(max_gaussians_)
        );
    }
}

void GaussianModel::loadFromTensors(
    const Tensor& means3D,
    const Tensor& scales,
    const Tensor& rotations,
    const Tensor& sh_coeffs,
    const Tensor& opacity,
    const Tensor& timestep)
{
    int64_t N = means3D.size(0);
    assert(N <= max_gaussians_);
    assert(means3D.size(1) == 3);
    assert(scales.size(0) == N && scales.size(1) == 3);
    assert(rotations.size(0) == N && rotations.size(1) == 4);
    assert(sh_coeffs.size(0) == N && sh_coeffs.size(1) == 48);
    assert(opacity.size(0) == N && opacity.size(1) == 1);
    
    // 复制所有参数到预分配的内存
    means3D_.slice(0, 0, N).copy_(means3D.to(device_));
    scales_.slice(0, 0, N).copy_(scales.to(device_));
    rotations_.slice(0, 0, N).copy_(rotations.to(device_));
    sh_coeffs_.slice(0, 0, N).copy_(sh_coeffs.to(device_));
    opacity_.slice(0, 0, N).copy_(opacity.to(device_));
    
    // 加载timestep，如果没有提供则初始化为0
    if (timestep.defined() && timestep.size(0) == N) {
        timestep_.slice(0, 0, N).copy_(timestep.to(device_));
    } else {
        timestep_.slice(0, 0, N).fill_(0.0f);
    }
    
    // 更新活跃掩码
    active_count_ = N;
    alive_mask_.slice(0, 0, N).fill_(true);
    if (N < max_gaussians_) {
        alive_mask_.slice(0, N, max_gaussians_).fill_(false);
    }
}

void GaussianModel::updateParams(
    const Tensor& means3D,
    const Tensor& scales,
    const Tensor& rotations,
    const Tensor& sh_coeffs,
    const Tensor& opacity)
{
    int64_t N = means3D.size(0);
    assert(N == active_count_);
    assert(means3D.size(1) == 3);
    assert(scales.size(0) == N && scales.size(1) == 3);
    assert(rotations.size(0) == N && rotations.size(1) == 4);
    assert(sh_coeffs.size(0) == N && sh_coeffs.size(1) == 48);
    assert(opacity.size(0) == N && opacity.size(1) == 1);
    
    // 直接更新活跃部分的参数
    means3D_.slice(0, 0, N).copy_(means3D.to(device_));
    scales_.slice(0, 0, N).copy_(scales.to(device_));
    rotations_.slice(0, 0, N).copy_(rotations.to(device_));
    sh_coeffs_.slice(0, 0, N).copy_(sh_coeffs.to(device_));
    opacity_.slice(0, 0, N).copy_(opacity.to(device_));
}

void GaussianModel::updateAliveMask()
{
    // 更新活跃掩码，确保与active_count_一致
    if (active_count_ > 0) {
        alive_mask_.slice(0, 0, active_count_).fill_(true);
    }
    if (active_count_ < max_gaussians_) {
        alive_mask_.slice(0, active_count_, max_gaussians_).fill_(false);
    }
}

} // namespace isogs

