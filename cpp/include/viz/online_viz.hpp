#pragma once

#include "viz/visualizer.hpp"
#include "core/gaussian_model.hpp"
#include "core/camera.hpp"
#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>

namespace isogs {

/**
 * @brief 在线可视化 - 实时显示SLAM重建进度
 */
class OnlineVisualizer {
public:
    OnlineVisualizer() = default;
    ~OnlineVisualizer() = default;
    
    /**
     * @brief 初始化可视化
     */
    bool initialize(int width, int height);
    
    /**
     * @brief 更新渲染结果
     */
    void updateRender(const Tensor& rendered_rgb, const Tensor& rendered_depth);
    
    /**
     * @brief 更新Loss曲线
     */
    void updateLoss(const std::vector<float>& loss_history);
    
    /**
     * @brief 更新网格重建进度
     */
    void updateMeshProgress(float progress);
    
    /**
     * @brief 显示当前状态
     */
    void show();
    
    /**
     * @brief 检查是否应该关闭
     */
    bool shouldClose() const;
    
    /**
     * @brief 关闭可视化
     */
    void close();
    
private:
    std::unique_ptr<Visualizer> visualizer_;
    std::vector<float> loss_history_;
    float mesh_progress_ = 0.0f;
};

} // namespace isogs

