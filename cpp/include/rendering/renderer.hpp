#pragma once

#include "core/gaussian_model.hpp"
#include "core/camera.hpp"
#include "core/types.hpp"
#include <torch/torch.h>
#include <memory>

// gsplat headers (forward declarations only, implementation in .cpp)
// 在.cpp文件中包含具体的gsplat头文件

namespace isogs {

/**
 * @brief 渲染器接口 - 封装gsplat渲染功能
 * 
 * 目前是占位实现，等待gsplat集成
 */
class Renderer {
public:
    Renderer() = default;
    ~Renderer() = default;
    
    /**
     * @brief 渲染RGB图像
     * @param model 高斯模型
     * @param camera 相机参数
     * @return 渲染的RGB图像 [3, H, W]
     */
    Tensor renderRGB(
        const GaussianModel& model,
        const Camera& camera
    );
    
    /**
     * @brief 渲染深度图
     * @param model 高斯模型
     * @param camera 相机参数
     * @return 渲染的深度图 [H, W]
     */
    Tensor renderDepth(
        const GaussianModel& model,
        const Camera& camera
    );
    
    /**
     * @brief 渲染RGB和深度
     * @param model 高斯模型
     * @param camera 相机参数
     * @return {RGB图像, 深度图}
     */
    std::tuple<Tensor, Tensor> renderRGBD(
        const GaussianModel& model,
        const Camera& camera
    );
    
private:
    // TODO: 添加gsplat渲染器实例
    // 目前使用占位实现
};

} // namespace isogs

