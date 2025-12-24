#pragma once

#include "core/gaussian_model.hpp"
#include "core/camera.hpp"
#include "core/types.hpp"
#include "rendering/renderer.hpp"
#include <torch/torch.h>
#include <memory>
#include <map>

namespace isogs {

/**
 * @brief Tracking模块 - 相机位姿优化
 */
class Tracker {
public:
    struct Config {
        int num_iters = 10;
        bool use_sil_for_loss = true;
        float sil_thres = 0.99f;
        bool use_l1 = true;
        bool ignore_outlier_depth_loss = false;
        std::map<std::string, float> loss_weights = {
            {"im", 0.5f},
            {"depth", 1.0f}
        };
        std::map<std::string, float> learning_rates = {
            {"cam_unnorm_rots", 0.0004f},
            {"cam_trans", 0.002f}
        };
    };
    
    Tracker(const Config& config);
    ~Tracker() = default;
    
    /**
     * @brief 优化当前帧的相机位姿
     * @param model 高斯模型
     * @param camera 当前相机（会被更新）
     * @param gt_color 真实RGB图像 [3, H, W]
     * @param gt_depth 真实深度图 [H, W]
     * @return 优化后的Loss值
     */
    float optimizePose(
        GaussianModel& model,
        Camera& camera,
        const Tensor& gt_color,
        const Tensor& gt_depth
    );
    
    /**
     * @brief 计算Loss（不进行优化）
     */
    std::map<std::string, float> computeLoss(
        const GaussianModel& model,
        const Camera& camera,
        const Tensor& gt_color,
        const Tensor& gt_depth
    );
    
private:
    Config config_;
    std::unique_ptr<Renderer> renderer_;
    
    // 优化器（使用LibTorch Autograd，仅用于相机位姿）
    std::unique_ptr<torch::optim::Adam> optimizer_;
    
    // 辅助函数：计算RGB Loss
    Tensor computeRGBLoss(const Tensor& rendered, const Tensor& gt);
    
    // 辅助函数：计算Depth Loss
    Tensor computeDepthLoss(const Tensor& rendered, const Tensor& gt, const Tensor& mask);
};

} // namespace isogs

