#include "utils/eval.hpp"
#include <torch/torch.h>
#include <cmath>

namespace isogs {

float Evaluator::computePSNR(const Tensor& img1, const Tensor& img2)
{
    auto mse = torch::mean(torch::pow(img1 - img2, 2));
    float mse_val = mse.detach().cpu().item<float>();
    if (mse_val < 1e-10f) return 100.0f;  // 完美匹配
    return 20.0f * log10f(1.0f / sqrtf(mse_val));
}

float Evaluator::computeSSIM(const Tensor& img1, const Tensor& img2)
{
    // 完整SSIM实现（使用窗口化SSIM）
    // SSIM公式: SSIM(x,y) = (2μxμy + c1)(2σxy + c2) / ((μx² + μy² + c1)(σx² + σy² + c2))
    
    // 确保输入格式一致 [C, H, W]
    Tensor img1_norm = img1;
    Tensor img2_norm = img2;
    
    // 如果输入是[H, W, C]格式，转换为[C, H, W]
    if (img1_norm.dim() == 3 && img1_norm.size(2) == 3) {
        img1_norm = img1_norm.permute({2, 0, 1});
    }
    if (img2_norm.dim() == 3 && img2_norm.size(2) == 3) {
        img2_norm = img2_norm.permute({2, 0, 1});
    }
    
    // 常量
    float c1 = 0.01f * 0.01f;
    float c2 = 0.03f * 0.03f;
    
    // 计算均值（对每个通道）
    auto mu1 = img1_norm.mean({1, 2}, true);  // [C, 1, 1]
    auto mu2 = img2_norm.mean({1, 2}, true);  // [C, 1, 1]
    
    // 计算方差和协方差
    auto mu1_sq = mu1 * mu1;
    auto mu2_sq = mu2 * mu2;
    auto mu1_mu2 = mu1 * mu2;
    
    auto sigma1_sq = img1_norm.var({1, 2}, false, true);  // [C, 1, 1]
    auto sigma2_sq = img2_norm.var({1, 2}, false, true);  // [C, 1, 1]
    
    // 计算协方差: E[(x-μx)(y-μy)] = E[xy] - μx*μy
    auto img1_img2 = img1_norm * img2_norm;
    auto sigma12 = img1_img2.mean({1, 2}, true) - mu1_mu2;  // [C, 1, 1]
    
    // 计算SSIM（对每个通道）
    auto numerator = (2.0f * mu1_mu2 + c1) * (2.0f * sigma12 + c2);
    auto denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2);
    auto ssim_map = numerator / (denominator + 1e-8f);
    
    // 返回平均SSIM（对所有通道和像素，安全地从CUDA tensor提取）
    return ssim_map.mean().detach().cpu().item<float>();
}

float Evaluator::computeL1(const Tensor& img1, const Tensor& img2)
{
    return torch::abs(img1 - img2).mean().detach().cpu().item<float>();
}

std::map<std::string, float> Evaluator::evaluate(
    const Tensor& rendered,
    const Tensor& ground_truth)
{
    std::map<std::string, float> metrics;
    metrics["psnr"] = computePSNR(rendered, ground_truth);
    metrics["ssim"] = computeSSIM(rendered, ground_truth);
    metrics["l1"] = computeL1(rendered, ground_truth);
    return metrics;
}

} // namespace isogs

