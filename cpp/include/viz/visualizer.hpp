#pragma once

#include "core/types.hpp"
#include <torch/torch.h>
#include <string>
#include <vector>

namespace isogs {

/**
 * @brief 基础可视化接口
 */
class Visualizer {
public:
    Visualizer() = default;
    virtual ~Visualizer() = default;
    
    /**
     * @brief 初始化可视化窗口
     */
    virtual bool initialize(int width, int height, const std::string& title) = 0;
    
    /**
     * @brief 显示图像
     */
    virtual void showImage(const Tensor& image, const std::string& window_name) = 0;
    
    /**
     * @brief 更新显示
     */
    virtual void update() = 0;
    
    /**
     * @brief 检查是否应该关闭
     */
    virtual bool shouldClose() const = 0;
    
    /**
     * @brief 关闭可视化
     */
    virtual void close() = 0;
};

} // namespace isogs

