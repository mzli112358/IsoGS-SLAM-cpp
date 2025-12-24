#pragma once

#include "core/types.hpp"
#include "core/camera.hpp"
#include <torch/torch.h>
#include <string>

namespace isogs {

/**
 * @brief 点云工具 - 从RGB-D图像提取点云
 */
class PointCloudUtils {
public:
    /**
     * @brief 从RGB-D图像提取点云
     * @param color RGB图像 [H, W, 3] 或 [3, H, W]
     * @param depth 深度图 [H, W]
     * @param intrinsics 相机内参 [3, 3]
     * @param w2c 世界到相机变换 [4, 4]
     * @param mask 有效深度掩码 [H, W] (可选)
     * @param compute_mean_sq_dist 是否计算mean_sq_dist
     * @return {points, colors, mean_sq_dist} 或 {points, colors}
     */
    static std::tuple<Tensor, Tensor, Tensor> extractPointCloud(
        const Tensor& color,
        const Tensor& depth,
        const Tensor& intrinsics,
        const Tensor& w2c,
        const Tensor& mask = Tensor(),
        bool compute_mean_sq_dist = true
    );
    
    /**
     * @brief 仅提取点云（不计算mean_sq_dist）
     */
    static std::tuple<Tensor, Tensor> extractPointCloudSimple(
        const Tensor& color,
        const Tensor& depth,
        const Tensor& intrinsics,
        const Tensor& w2c,
        const Tensor& mask = Tensor()
    );
};

} // namespace isogs

