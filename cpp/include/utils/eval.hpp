#pragma once

#include "core/types.hpp"
#include <torch/torch.h>
#include <map>
#include <string>

namespace isogs {

/**
 * @brief 评估工具 - 计算PSNR、SSIM等指标
 */
class Evaluator {
public:
    /**
     * @brief 计算PSNR
     */
    static float computePSNR(const Tensor& img1, const Tensor& img2);
    
    /**
     * @brief 计算SSIM
     */
    static float computeSSIM(const Tensor& img1, const Tensor& img2);
    
    /**
     * @brief 计算L1 Loss
     */
    static float computeL1(const Tensor& img1, const Tensor& img2);
    
    /**
     * @brief 评估渲染质量
     */
    static std::map<std::string, float> evaluate(
        const Tensor& rendered,
        const Tensor& ground_truth
    );
};

} // namespace isogs

