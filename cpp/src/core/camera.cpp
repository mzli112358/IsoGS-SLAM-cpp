#include "core/camera.hpp"
#include <cmath>

namespace isogs {

Camera::Camera(
    const Tensor& intrinsics,
    const Tensor& w2c,
    int width,
    int height)
    : intrinsics_(intrinsics.clone())
    , w2c_(w2c.clone())
    , width_(width)
    , height_(height)
{
    // 确保tensor在CUDA上
    if (!intrinsics_.is_cuda()) {
        intrinsics_ = intrinsics_.to(torch::kCUDA);
    }
    if (!w2c_.is_cuda()) {
        w2c_ = w2c_.to(torch::kCUDA);
    }
    updateC2W();
}

void Camera::setW2C(const Tensor& w2c)
{
    w2c_ = w2c.clone();
    // 确保tensor在CUDA上
    if (!w2c_.is_cuda()) {
        w2c_ = w2c_.to(torch::kCUDA);
    }
    updateC2W();
}

void Camera::updateC2W()
{
    // 确保w2c_在CUDA上
    if (!w2c_.is_cuda()) {
        w2c_ = w2c_.to(torch::kCUDA);
    }
    
    // 计算w2c的逆矩阵
    // 在 CPU 上计算逆矩阵，然后移到 CUDA（避免 CUBLAS 初始化问题）
    auto w2c_cpu = w2c_.cpu();
    c2w_ = w2c_cpu.inverse().to(torch::kCUDA);
}

Tensor Camera::getProjectionMatrix(float near_plane, float far_plane) const
{
    // 计算透视投影矩阵
    // 安全地从CUDA tensor提取内参（先移到CPU）
    auto intrinsics_cpu = intrinsics_.cpu();
    float fx = intrinsics_cpu[0][0].item<float>();
    float fy = intrinsics_cpu[1][1].item<float>();
    float cx = intrinsics_cpu[0][2].item<float>();
    float cy = intrinsics_cpu[1][2].item<float>();
    
    float width = static_cast<float>(width_);
    float height = static_cast<float>(height_);
    
    // OpenGL风格的投影矩阵（用于gsplat）
    auto proj = torch::zeros({4, 4}, torch::kFloat32);
    
    proj[0][0] = 2.0f * fx / width;
    proj[0][2] = 2.0f * (cx / width) - 1.0f;
    proj[1][1] = 2.0f * fy / height;
    proj[1][2] = 2.0f * (cy / height) - 1.0f;
    proj[2][2] = -(far_plane + near_plane) / (far_plane - near_plane);
    proj[2][3] = -2.0f * far_plane * near_plane / (far_plane - near_plane);
    proj[3][2] = -1.0f;
    
    return proj;
}

} // namespace isogs

