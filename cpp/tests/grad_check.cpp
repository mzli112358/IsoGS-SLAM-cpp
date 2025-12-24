#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>
#include "losses/flat_loss.hpp"
#include "losses/iso_loss.hpp"
#include "utils/spatial_hash.hpp"

using namespace isogs;

/**
 * @brief 计算数值梯度（使用有限差分法）
 * 
 * 数值梯度：df/dx ≈ (f(x + eps) - f(x - eps)) / (2 * eps)
 */
template<typename LossFunc>
Tensor computeNumericalGradient(
    const Tensor& params,
    LossFunc loss_func,
    float eps = 1e-5f
) {
    int64_t num_params = params.numel();
    auto grad = torch::zeros_like(params);
    
    auto params_flat = params.flatten();
    auto grad_flat = grad.flatten();
    
    // 对每个参数计算数值梯度
    for (int64_t i = 0; i < num_params; ++i) {
        // f(x + eps)
        auto params_plus = params_flat.clone();
        params_plus[i] += eps;
        auto params_plus_reshaped = params_plus.reshape(params.sizes());
        float loss_plus = loss_func(params_plus_reshaped);
        
        // f(x - eps)
        auto params_minus = params_flat.clone();
        params_minus[i] -= eps;
        auto params_minus_reshaped = params_minus.reshape(params.sizes());
        float loss_minus = loss_func(params_minus_reshaped);
        
        // 数值梯度
        float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
        grad_flat[i] = num_grad;
    }
    
    return grad;
}

/**
 * @brief 梯度检查统计信息
 */
struct GradCheckStats {
    float max_diff = 0.0f;
    float mean_diff = 0.0f;
    float relative_error = 0.0f;
    int num_failed = 0;
    int total_params = 0;
    
    void compute(const Tensor& grad_analytical, const Tensor& grad_numerical) {
        auto diff = (grad_analytical - grad_numerical).abs();
        max_diff = diff.max().item<float>();
        mean_diff = diff.mean().item<float>();
        
        // 相对误差：|analytical - numerical| / (|analytical| + |numerical| + eps)
        auto denom = grad_analytical.abs() + grad_numerical.abs() + 1e-8f;
        auto rel_diff = diff / denom;
        relative_error = rel_diff.max().item<float>();
        
        total_params = grad_analytical.numel();
        
        // 统计失败的数量（误差 > 1e-4）
        auto failed_mask = diff > 1e-4f;
        num_failed = failed_mask.sum().item<int>();
    }
    
    void print(const std::string& name) const {
        std::cout << "  " << name << " Gradient Check:" << std::endl;
        std::cout << "    Max absolute error: " << std::scientific << std::setprecision(6) << max_diff << std::endl;
        std::cout << "    Mean absolute error: " << std::scientific << std::setprecision(6) << mean_diff << std::endl;
        std::cout << "    Max relative error: " << std::scientific << std::setprecision(6) << relative_error << std::endl;
        std::cout << "    Failed params: " << num_failed << " / " << total_params << std::endl;
    }
    
    bool passed(float threshold = 1e-4f) const {
        return max_diff < threshold && relative_error < 0.1f;  // 相对误差 < 10%
    }
};

/**
 * @brief 测试Flat Loss梯度
 */
bool testFlatLossGradient() {
    std::cout << "\n=== Testing Flat Loss Gradient ===" << std::endl;
    
    // 创建测试数据
    int num_gaussians = 100;
    auto scales = torch::randn({num_gaussians, 3}, torch::kFloat32).cuda().abs() + 0.1f;
    
    // 1. CUDA Kernel计算的解析梯度
    Tensor grad_analytical = torch::zeros_like(scales);
    float loss_analytical = isogs::FlatLoss::compute(scales, grad_analytical);
    
    // 2. 数值梯度计算
    auto loss_func = [](const Tensor& s) -> float {
        return isogs::FlatLoss::computeLoss(s);
    };
    Tensor grad_numerical = computeNumericalGradient(scales, loss_func, 1e-5f);
    
    // 3. PyTorch Autograd（作为参考）
    auto scales_grad = scales.clone().detach().requires_grad_(true);
    auto scales_exp = scales_grad.exp();
    auto min_result = scales_exp.min(1);
    auto loss_pytorch = std::get<0>(min_result).mean();
    loss_pytorch.backward();
    auto grad_pytorch = scales_grad.grad();
    
    // 4. 统计和对比
    GradCheckStats stats_cuda_vs_numerical;
    stats_cuda_vs_numerical.compute(grad_analytical, grad_numerical);
    stats_cuda_vs_numerical.print("CUDA vs Numerical");
    
    GradCheckStats stats_cuda_vs_pytorch;
    stats_cuda_vs_pytorch.compute(grad_analytical, grad_pytorch);
    stats_cuda_vs_pytorch.print("CUDA vs PyTorch");
    
    std::cout << "  Loss (CUDA): " << loss_analytical << std::endl;
    float loss_pytorch_val = loss_pytorch.cpu().item<float>();
    std::cout << "  Loss (PyTorch): " << loss_pytorch_val << std::endl;
    
    bool passed = stats_cuda_vs_numerical.passed(1e-4f);
    if (passed) {
        std::cout << "  ✓ Flat Loss gradient check PASSED" << std::endl;
    } else {
        std::cout << "  ✗ Flat Loss gradient check FAILED" << std::endl;
    }
    
    return passed;
}

/**
 * @brief 测试Iso Loss梯度
 */
bool testIsoLossGradient() {
    std::cout << "\n=== Testing Iso-Surface Loss Gradient ===" << std::endl;
    
    // 创建测试数据
    int num_gaussians = 200;
    auto means = torch::randn({num_gaussians, 3}, torch::kFloat32).cuda() * 2.0f;
    auto log_scales = torch::randn({num_gaussians, 3}, torch::kFloat32).cuda() * 0.5f;
    auto scales = torch::exp(log_scales).clamp(1e-5f);
    auto rotations = torch::randn({num_gaussians, 4}, torch::kFloat32).cuda();
    rotations = torch::nn::functional::normalize(rotations, torch::nn::functional::NormalizeFuncOptions().dim(1));
    auto opacities = torch::sigmoid(torch::randn({num_gaussians, 1}, torch::kFloat32).cuda());
    
    // 构建逆协方差矩阵（简化：使用对角矩阵）
    auto inv_covariances = torch::zeros({num_gaussians, 3, 3}, torch::kFloat32).cuda();
    auto scales_cpu = scales.cpu();
    for (int i = 0; i < num_gaussians; ++i) {
        float sx = scales_cpu[i][0].item<float>();
        float sy = scales_cpu[i][1].item<float>();
        float sz = scales_cpu[i][2].item<float>();
        float inv_sx = 1.0f / (sx * sx + 1e-8f);
        float inv_sy = 1.0f / (sy * sy + 1e-8f);
        float inv_sz = 1.0f / (sz * sz + 1e-8f);
        inv_covariances[i][0][0] = inv_sx;
        inv_covariances[i][1][1] = inv_sy;
        inv_covariances[i][2][2] = inv_sz;
    }
    
    // 构建空间哈希表
    isogs::SpatialHash::Config hash_config;
    hash_config.cell_size = 0.5f;
    isogs::SpatialHash spatial_hash(hash_config);
    spatial_hash.build(means);
    spatial_hash.setMeans(means);
    
    // 1. CUDA Kernel计算的解析梯度
    Tensor grad_means = torch::zeros_like(means);
    Tensor grad_opacities = torch::zeros_like(opacities);
    
    isogs::IsoSurfaceLoss iso_loss;
    float loss_analytical = iso_loss.compute(
        means, inv_covariances, opacities,
        grad_means, grad_opacities,
        spatial_hash
    );
    
    // 2. 数值梯度计算（只测试means梯度，因为opacity梯度类似）
    // 注意：Iso Loss的数值梯度计算较慢，我们只测试一小部分参数
    std::cout << "  Computing numerical gradients (this may take a while)..." << std::endl;
    
    // 简化：只测试前10个高斯的means梯度
    int test_size = std::min(10, num_gaussians);
    auto means_test = means.slice(0, 0, test_size);
    auto grad_means_test = grad_means.slice(0, 0, test_size);
    
    auto loss_func_means = [&](const Tensor& m) -> float {
        // 创建临时means（只修改测试部分）
        auto means_temp = means.clone();
        means_temp.slice(0, 0, test_size).copy_(m);
        
        isogs::IsoSurfaceLoss iso_loss_temp;
        isogs::SpatialHash spatial_hash_temp(hash_config);
        spatial_hash_temp.build(means_temp);
        spatial_hash_temp.setMeans(means_temp);
        return iso_loss_temp.computeLoss(means_temp, inv_covariances, opacities, spatial_hash_temp);
    };
    Tensor grad_means_numerical = computeNumericalGradient(means_test, loss_func_means, 1e-5f);
    
    // 3. 统计和对比
    GradCheckStats stats_means;
    stats_means.compute(grad_means_test, grad_means_numerical);
    stats_means.print("Iso Loss Means (first " + std::to_string(test_size) + " Gaussians)");
    
    // 测试opacity梯度（只测试前10个）
    auto opacities_test = opacities.slice(0, 0, test_size);
    auto grad_opacities_test = grad_opacities.slice(0, 0, test_size);
    
    auto loss_func_opacity = [&](const Tensor& o) -> float {
        // 创建临时opacities（只修改测试部分）
        auto opacities_temp = opacities.clone();
        opacities_temp.slice(0, 0, test_size).copy_(o);
        
        isogs::IsoSurfaceLoss iso_loss_temp;
        isogs::SpatialHash spatial_hash_temp(hash_config);
        spatial_hash_temp.build(means);
        spatial_hash_temp.setMeans(means);
        return iso_loss_temp.computeLoss(means, inv_covariances, opacities_temp, spatial_hash_temp);
    };
    Tensor grad_opacities_numerical = computeNumericalGradient(opacities_test, loss_func_opacity, 1e-5f);
    
    GradCheckStats stats_opacity;
    stats_opacity.compute(grad_opacities_test, grad_opacities_numerical);
    stats_opacity.print("Iso Loss Opacity (first " + std::to_string(test_size) + " Gaussians)");
    
    std::cout << "  Loss: " << loss_analytical << std::endl;
    
    bool passed = stats_means.passed(1e-3f) && stats_opacity.passed(1e-3f);  // Iso Loss允许稍大的误差
    if (passed) {
        std::cout << "  ✓ Iso Loss gradient check PASSED" << std::endl;
    } else {
        std::cout << "  ✗ Iso Loss gradient check FAILED" << std::endl;
    }
    
    return passed;
}

/**
 * @brief 梯度检查工具主函数
 */
int main(int /*argc*/, char* /*argv*/[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "   Gradient Check Tool" << std::endl;
    std::cout << "========================================" << std::endl;
    
    if (!torch::cuda::is_available()) {
        std::cerr << "Error: CUDA is not available!" << std::endl;
        return 1;
    }
    
    std::cout << "CUDA available: " << torch::cuda::is_available() << std::endl;
    if (torch::cuda::is_available()) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "Device: " << prop.name << std::endl;
    }
    
    bool all_passed = true;
    
    // 测试Flat Loss
    bool flat_passed = testFlatLossGradient();
    all_passed = all_passed && flat_passed;
    
    // 测试Iso Loss
    bool iso_passed = testIsoLossGradient();
    all_passed = all_passed && iso_passed;
    
    // 总结
    std::cout << "\n========================================" << std::endl;
    if (all_passed) {
        std::cout << "✓ All gradient checks PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "✗ Some gradient checks FAILED" << std::endl;
        return 1;
    }
}
