#pragma once

#include "core/types.hpp"
#include "slam/mapper.hpp"
#include <torch/torch.h>
#include <vector>

namespace isogs {

/**
 * @brief 从采样的像素点反投影到3D点云
 * 
 * @param depth 深度图 [H, W]
 * @param intrinsics 相机内参 [3, 3]
 * @param w2c 世界到相机变换矩阵 [4, 4]
 * @param sampled_indices 采样的像素索引 [N, 2] (y, x)
 * @return Tensor 3D点云 [N', 3] (去除无效点后)
 */
Tensor getPointCloudFromSampledPixels(
    const Tensor& depth,
    const Tensor& intrinsics,
    const Tensor& w2c,
    const Tensor& sampled_indices
);

/**
 * @brief 基于重叠度选择关键帧
 * 
 * 从当前帧深度图随机采样像素，反投影到3D空间，然后将3D点投影到每个关键帧，
 * 计算重叠度（投影点在图像内的百分比），按重叠度排序并选择top-k。
 * 
 * @param current_depth 当前帧深度图 [H, W]
 * @param current_w2c 当前帧世界到相机变换矩阵 [4, 4]
 * @param intrinsics 相机内参 [3, 3]
 * @param keyframes 关键帧列表
 * @param k 要选择的关键帧数量
 * @param pixels 采样像素数量（默认1600）
 * @return std::vector<int> 选中的关键帧索引列表
 */
std::vector<int> keyframeSelectionOverlap(
    const Tensor& current_depth,
    const Tensor& current_w2c,
    const Tensor& intrinsics,
    const std::vector<Keyframe>& keyframes,
    int k,
    int pixels = 1600
);

} // namespace isogs

