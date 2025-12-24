#pragma once

#include "types.hpp"
#include <torch/torch.h>
#include <cstdint>
#include <memory>

namespace isogs {

/**
 * @brief 高斯内存池 - 预分配显存，支持零拷贝操作
 * 
 * 设计目标：
 * - 预分配固定大小显存（支持2000万高斯）
 * - 使用位掩码管理活跃高斯
 * - 支持零拷贝的densification和pruning
 */
class GaussianModel {
public:
    // 构造函数
    explicit GaussianModel(int64_t max_gaussians = 20000000, Device device = kCUDA);
    
    // 析构函数
    ~GaussianModel() = default;
    
    // 禁用拷贝构造和赋值
    GaussianModel(const GaussianModel&) = delete;
    GaussianModel& operator=(const GaussianModel&) = delete;
    
    // 移动构造和赋值
    GaussianModel(GaussianModel&&) = default;
    GaussianModel& operator=(GaussianModel&&) = default;
    
    // 初始化：从点云数据初始化高斯
    void initializeFromPointCloud(
        const Tensor& points,      // [N, 3]
        const Tensor& colors,      // [N, 3]
        const Tensor& mean_sq_dist,  // [N] 用于初始化scale
        const std::string& gaussian_distribution = "anisotropic"  // "isotropic" or "anisotropic"
    );
    
    // 获取活跃高斯数量
    int64_t getActiveCount() const { return active_count_; }
    
    // 获取最大高斯数量
    int64_t getMaxGaussians() const { return max_gaussians_; }
    
    // 获取参数张量（只读）
    Tensor getMeans3D() const { return means3D_.slice(0, 0, active_count_); }
    Tensor getScales() const { return scales_.slice(0, 0, active_count_); }  // log空间
    Tensor getRotations() const { return rotations_.slice(0, 0, active_count_); }
    Tensor getSHCoeffs() const { return sh_coeffs_.slice(0, 0, active_count_); }
    Tensor getOpacity() const { return opacity_.slice(0, 0, active_count_); }  // logit空间
    Tensor getTimestep() const { return timestep_.slice(0, 0, active_count_); }  // [N] 每个高斯点创建的帧索引
    
    // 获取完整张量（用于渲染）
    Tensor getFullMeans3D() const { return means3D_; }
    Tensor getFullScales() const { return scales_; }
    Tensor getFullRotations() const { return rotations_; }
    Tensor getFullSHCoeffs() const { return sh_coeffs_; }
    Tensor getFullOpacity() const { return opacity_; }
    
    // 获取活跃掩码
    Tensor getAliveMask() const { return alive_mask_; }
    
    // Densification: 添加新高斯（零拷贝）
    void densify(const Indices& split_indices, const Indices& clone_indices);
    
    // Pruning: 移除高斯（零拷贝）
    void prune(const Indices& prune_indices);
    
    // 添加新高斯（从点云）
    void addGaussians(
        const Tensor& points,
        const Tensor& colors,
        const Tensor& mean_sq_dist,
        int time_idx = 0,  // 当前帧索引，用于设置timestep
        const std::string& gaussian_distribution = "anisotropic"  // "isotropic" or "anisotropic"
    );
    
    // 从张量直接加载参数（用于checkpoint加载）
    void loadFromTensors(
        const Tensor& means3D,      // [N, 3]
        const Tensor& scales,       // [N, 3] (log空间)
        const Tensor& rotations,    // [N, 4] (四元数)
        const Tensor& sh_coeffs,    // [N, 48]
        const Tensor& opacity,      // [N, 1] (logit空间)
        const Tensor& timestep = Tensor()  // [N] 可选，如果不提供则初始化为0
    );
    
    // 更新参数（用于优化器更新后写回）
    void updateParams(
        const Tensor& means3D,      // [N, 3]
        const Tensor& scales,       // [N, 3] (log空间)
        const Tensor& rotations,    // [N, 4] (四元数)
        const Tensor& sh_coeffs,    // [N, 48]
        const Tensor& opacity       // [N, 1] (logit空间)
    );
    
    // 获取设备
    Device getDevice() const { return device_; }
    
private:
    int64_t max_gaussians_;
    int64_t active_count_;
    Device device_;
    
    // 预分配的显存张量
    Tensor means3D_;        // [max_gaussians, 3]
    Tensor scales_;         // [max_gaussians, 3] (log空间)
    Tensor rotations_;      // [max_gaussians, 4] (四元数)
    Tensor sh_coeffs_;      // [max_gaussians, 48] (SH系数，RGB用前3个)
    Tensor opacity_;        // [max_gaussians, 1] (logit空间)
    Tensor timestep_;       // [max_gaussians] 每个高斯点创建的帧索引
    
    // 活跃掩码（位掩码）
    Tensor alive_mask_;     // [max_gaussians] (bool)
    
    // 辅助函数：更新活跃掩码
    void updateAliveMask();
    
    // 辅助函数：确保有足够空间
    void ensureCapacity(int64_t additional_count);
};

} // namespace isogs

