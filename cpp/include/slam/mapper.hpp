#pragma once

#include "core/gaussian_model.hpp"
#include "core/camera.hpp"
#include "core/types.hpp"
#include "rendering/renderer.hpp"
#include "utils/optimizer.hpp"
#include <torch/torch.h>
#include <vector>
#include <memory>
#include <map>

namespace isogs {

/**
 * @brief 关键帧信息
 */
struct Keyframe {
    int64_t frame_id;
    Camera camera;
    Tensor color;  // [3, H, W]
    Tensor depth;  // [H, W]
};

/**
 * @brief Mapping模块 - 高斯模型优化
 */
class Mapper {
public:
    struct Config {
        int num_iters = 40;
        bool add_new_gaussians = true;
        bool prune_gaussians = true;
        bool use_gaussian_splatting_densification = true;
        float sil_thres = 0.99f;
        bool use_sil_for_loss = false;
        bool use_l1 = true;
        bool ignore_outlier_depth_loss = false;
        int mapping_window_size = 24;
        std::map<std::string, float> loss_weights = {
            {"im", 1.0f},
            {"depth", 1.0f},
            {"flat", 50.0f},
            {"iso", 2.0f}
        };
        std::map<std::string, float> learning_rates = {
            {"means3D", 0.00016f},
            {"rgb_colors", 0.0025f},
            {"unnorm_rotations", 0.001f},
            {"logit_opacities", 0.05f},
            {"log_scales", 0.005f}
        };
        // Densification配置
        float densify_grad_thresh = 0.0002f;
        int densify_every = 100;
        int densify_start_after = 500;
        int densify_stop_after = 15000;
        int num_to_split_into = 2;
        float scene_radius = 1.0f;  // 需要根据场景计算
        // Pruning配置
        float removal_opacity_threshold = 0.05f;
        float final_removal_opacity_threshold = 0.1f;
        int prune_every = 100;
        int prune_start_after = 0;
        int prune_stop_after = 30000;
        int remove_big_after = 0;
        std::string gaussian_distribution = "isotropic";  // "isotropic" or "anisotropic"
    };
    
    Mapper(const Config& config);
    ~Mapper() = default;
    
    /**
     * @brief 添加关键帧
     */
    void addKeyframe(const Keyframe& keyframe);
    
    /**
     * @brief 优化高斯模型
     * @param model 高斯模型（会被更新）
     * @param current_frame 当前帧信息
     */
    void optimize(
        GaussianModel& model,
        const Keyframe& current_frame
    );
    
    /**
     * @brief 选择用于优化的关键帧
     * @param num_keyframes 要选择的关键帧数量
     * @param current_frame 当前帧信息（用于基于重叠度的选择）
     * @return 选中的关键帧索引列表
     */
    std::vector<int> selectKeyframes(int num_keyframes, const Keyframe& current_frame) const;
    
private:
    Config config_;
    std::vector<Keyframe> keyframes_;
    std::unique_ptr<Renderer> renderer_;
    std::unique_ptr<CudaAdamOptimizer> optimizer_;
    bool optimizer_initialized_;
    
    // 辅助函数：添加新高斯
    void addNewGaussians(
        GaussianModel& model,
        const Keyframe& frame
    );
    
    // 辅助函数：计算Loss（返回loss值和梯度）
    std::map<std::string, float> computeLoss(
        GaussianModel& model,
        const Keyframe& frame,
        std::map<std::string, Tensor>& grads
    );
    
    // 初始化优化器
    void initializeOptimizer(GaussianModel& model);
    
    // 检测需要densify的高斯
    void detectDensification(
        GaussianModel& model,
        const std::map<std::string, Tensor>& grads,
        Indices& split_indices,
        Indices& clone_indices
    );
    
    // 检测需要prune的高斯
    void detectPruning(
        GaussianModel& model,
        int iter,
        Indices& prune_indices
    );
};

} // namespace isogs

