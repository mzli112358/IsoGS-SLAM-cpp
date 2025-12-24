#pragma once

#include "core/gaussian_model.hpp"
#include "core/types.hpp"
#include "viz/visualizer.hpp"
#include <torch/torch.h>
#include <string>
#include <memory>

namespace isogs {

/**
 * @brief 最终重建可视化 - 交互式3D查看器
 */
class FinalVisualizer {
public:
    FinalVisualizer() = default;
    ~FinalVisualizer() = default;
    
    /**
     * @brief 初始化可视化
     */
    bool initialize(int width, int height);
    
    /**
     * @brief 加载checkpoint并显示
     */
    void loadAndShow(const std::string& checkpoint_path);
    
    /**
     * @brief 显示高斯点云
     */
    void showGaussians(const GaussianModel& model);
    
    /**
     * @brief 显示网格
     */
    void showMesh(const std::string& mesh_path);
    
    /**
     * @brief 运行交互式查看器
     */
    void run();
    
    /**
     * @brief 关闭可视化
     */
    void close();
    
private:
    std::unique_ptr<Visualizer> visualizer_;
    bool initialized_ = false;
};

} // namespace isogs

