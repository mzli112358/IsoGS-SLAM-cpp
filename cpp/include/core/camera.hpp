#pragma once

#include "types.hpp"
#include <torch/torch.h>
#include <array>

namespace isogs {

/**
 * @brief 相机参数和位姿管理
 */
class Camera {
public:
    Camera() = default;
    
    // 从内参矩阵和位姿矩阵初始化
    Camera(
        const Tensor& intrinsics,  // [3, 3]
        const Tensor& w2c,          // [4, 4]
        int width,
        int height
    );
    
    // 获取内参矩阵
    Tensor getIntrinsics() const { return intrinsics_; }
    
    // 获取世界到相机变换矩阵
    Tensor getW2C() const { return w2c_; }
    
    // 获取相机到世界变换矩阵
    Tensor getC2W() const { return c2w_; }
    
    // 获取图像尺寸
    int getWidth() const { return width_; }
    int getHeight() const { return height_; }
    
    // 设置位姿
    void setW2C(const Tensor& w2c);
    
    // 获取投影矩阵 (用于渲染)
    Tensor getProjectionMatrix(float near_plane = 0.01f, float far_plane = 100.0f) const;
    
private:
    Tensor intrinsics_;  // [3, 3] 内参矩阵
    Tensor w2c_;         // [4, 4] 世界到相机变换
    Tensor c2w_;         // [4, 4] 相机到世界变换（缓存）
    int width_;
    int height_;
    
    void updateC2W();
};

} // namespace isogs

