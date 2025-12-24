#pragma once

#include "core/types.hpp"
#include <torch/torch.h>
#include <random>

namespace isogs {

/**
 * @brief 设置随机种子
 * 
 * @param seed 种子值
 */
void seedEverything(int seed = 42);

/**
 * @brief 将参数从GPU转到CPU
 * 
 * @param tensor GPU上的tensor
 * @return Tensor CPU上的tensor
 */
Tensor paramsToCPU(const Tensor& tensor);

/**
 * @brief 将参数转换为numpy格式（用于Python交互）
 * 注意：这需要Pybind11支持，目前返回CPU tensor
 * 
 * @param tensor GPU上的tensor
 * @return Tensor CPU上的tensor（可以转换为numpy）
 */
Tensor paramsToNumpy(const Tensor& tensor);

} // namespace isogs

