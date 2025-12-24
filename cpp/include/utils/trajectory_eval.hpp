#pragma once

#include "core/types.hpp"
#include <torch/torch.h>
#include <vector>
#include <tuple>

namespace isogs {

/**
 * @brief 使用Umeyama算法对齐两条轨迹
 * 
 * @param model_traj 第一条轨迹的平移部分 [3, N]
 * @param data_traj 第二条轨迹的平移部分 [3, N]
 * @return std::tuple<Tensor, Tensor, Tensor> (旋转矩阵 [3, 3], 平移向量 [3, 1], 平移误差 [N])
 */
std::tuple<Tensor, Tensor, Tensor> alignTrajectories(
    const Tensor& model_traj,  // [3, N]
    const Tensor& data_traj     // [3, N]
);

/**
 * @brief 计算绝对轨迹误差（ATE）
 * 
 * @param gt_traj GT轨迹，每个元素是4x4变换矩阵
 * @param est_traj 估计轨迹，每个元素是4x4变换矩阵
 * @return float 平均平移误差
 */
float evaluateATE(
    const std::vector<Tensor>& gt_traj,   // 每个元素是 [4, 4]
    const std::vector<Tensor>& est_traj   // 每个元素是 [4, 4]
);

} // namespace isogs

