#pragma once

#include "core/types.hpp"
#include <torch/torch.h>

namespace isogs {

/**
 * @brief Loss计算工具函数
 */

// L1 Loss
inline Tensor l1_loss(const Tensor& pred, const Tensor& target) {
    return torch::abs(pred - target).mean();
}

// L2 Loss
inline Tensor l2_loss(const Tensor& pred, const Tensor& target) {
    return torch::pow(pred - target, 2).mean();
}

// Weighted L1 Loss
inline Tensor weighted_l1_loss(const Tensor& pred, const Tensor& target, const Tensor& weights) {
    return (torch::abs(pred - target) * weights).mean();
}

} // namespace isogs

