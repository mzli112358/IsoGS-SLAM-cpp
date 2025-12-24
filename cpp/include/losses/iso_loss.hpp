#pragma once

#include "core/types.hpp"
#include "utils/spatial_hash.hpp"
#include <torch/torch.h>

namespace isogs {

/**
 * @brief Iso-Surface Loss - 等势面约束
 * 
 * Loss: L_iso = mean((D(p) - target_saturation)^2)
 * 
 * 其中D(p) = sum(alpha_j * exp(-0.5 * delta^T * Sigma_j^{-1} * delta))
 * 
 * 使用空间哈希加速KNN查询
 */
class IsoSurfaceLoss {
public:
    struct Config {
        float target_saturation;
        int K;  // KNN数量
        int sample_size;  // 采样点数量
        int chunk_size;  // 内部批处理大小
        
        Config() : target_saturation(1.0f), K(16), sample_size(8192), chunk_size(128) {}
    };
    
    IsoSurfaceLoss(const Config& config = Config());
    
    /**
     * @brief 计算Iso-Surface Loss和梯度
     * @param means 高斯中心 [N, 3]
     * @param inverse_covariances 逆协方差矩阵 [N, 3, 3]
     * @param opacities 不透明度 [N, 1]
     * @param grad_means 输出的梯度 [N, 3]
     * @param grad_opacities 输出的梯度 [N, 1]
     * @param spatial_hash 空间哈希表（已构建）
     * @return Loss值
     */
    float compute(
        const Tensor& means,
        const Tensor& inverse_covariances,
        const Tensor& opacities,
        Tensor& grad_means,
        Tensor& grad_opacities,
        const SpatialHash& spatial_hash
    );
    
    /**
     * @brief 仅计算Loss（不计算梯度）
     */
    float computeLoss(
        const Tensor& means,
        const Tensor& inverse_covariances,
        const Tensor& opacities,
        const SpatialHash& spatial_hash
    );
    
private:
    Config config_;
};

} // namespace isogs

