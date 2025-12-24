#pragma once

#include "core/types.hpp"
#include <torch/torch.h>

namespace isogs {

/**
 * @brief Flat Loss - 扁平化约束
 * 
 * Loss: L_flat = mean(min(s_x, s_y, s_z))
 * 
 * 梯度推导：
 * 对于每个高斯，如果最小轴是x：
 *   dL/ds_x = 1/N
 *   dL/ds_y = 0
 *   dL/ds_z = 0
 */
class FlatLoss {
public:
    /**
     * @brief 计算Flat Loss和梯度
     * @param scales 高斯scale [N, 3] (已经是exp后的值，不是log空间)
     * @param grad_scales 输出的梯度 [N, 3] (会被累加)
     * @return Loss值
     */
    static float compute(
        const Tensor& scales,
        Tensor& grad_scales
    );
    
    /**
     * @brief 仅计算Loss（不计算梯度）
     */
    static float computeLoss(const Tensor& scales);
};

} // namespace isogs

