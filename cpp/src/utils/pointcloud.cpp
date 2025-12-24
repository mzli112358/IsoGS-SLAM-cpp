#include "utils/pointcloud.hpp"
#include <torch/torch.h>
#include <vector>

namespace isogs {

std::tuple<Tensor, Tensor, Tensor> PointCloudUtils::extractPointCloud(
    const Tensor& color,
    const Tensor& depth,
    const Tensor& intrinsics,
    const Tensor& w2c,
    const Tensor& mask,
    bool compute_mean_sq_dist)
{
    // 确保color格式为 [H, W, 3]
    Tensor color_reshaped;
    if (color.dim() == 3 && color.size(0) == 3) {
        // [3, H, W] -> [H, W, 3]
        color_reshaped = color.permute({1, 2, 0});
    } else if (color.dim() == 3 && color.size(2) == 3) {
        color_reshaped = color;
    } else {
        throw std::runtime_error("Invalid color tensor shape");
    }
    
    int height = color_reshaped.size(0);
    int width = color_reshaped.size(1);
    
    // 提取内参（安全地从CUDA tensor提取，先移到CPU）
    auto intrinsics_cpu = intrinsics.cpu();
    float fx = intrinsics_cpu[0][0].item<float>();
    float fy = intrinsics_cpu[1][1].item<float>();
    float cx = intrinsics_cpu[0][2].item<float>();
    float cy = intrinsics_cpu[1][2].item<float>();
    
    // 创建像素坐标网格
    auto x_grid = torch::arange(width, torch::kFloat32).to(depth.device());
    auto y_grid = torch::arange(height, torch::kFloat32).to(depth.device());
    auto meshgrid_result = torch::meshgrid({x_grid, y_grid}, "ij");
    auto xx = meshgrid_result[0];
    auto yy = meshgrid_result[1];
    
    // 归一化坐标
    xx = (xx - cx) / fx;
    yy = (yy - cy) / fy;
    
    // 展平
    xx = xx.reshape({-1});
    yy = yy.reshape({-1});
    auto depth_z = depth.reshape({-1});
    
    // 创建有效深度掩码
    Tensor valid_mask;
    if (mask.defined() && mask.numel() > 0) {
        valid_mask = mask.reshape({-1}) & (depth_z > 0.0f);
    } else {
        valid_mask = (depth_z > 0.0f);
    }
    
    // 应用掩码
    xx = xx.masked_select(valid_mask);
    yy = yy.masked_select(valid_mask);
    depth_z = depth_z.masked_select(valid_mask);
    
    // 计算相机坐标系下的点
    auto pts_cam = torch::stack({xx * depth_z, yy * depth_z, depth_z}, 1);  // [N, 3]
    
    // 转换到世界坐标系
    // 在 CPU 上计算所有矩阵操作（避免 CUBLAS 初始化问题）
    Tensor c2w;
    if (w2c.is_cuda()) {
        auto w2c_cpu = w2c.cpu();
        c2w = w2c_cpu.inverse();
    } else {
        c2w = w2c.inverse();
    }
    
    // 将点移到 CPU 上进行矩阵乘法
    auto pts_cam_cpu = pts_cam.cpu();
    auto ones = torch::ones({pts_cam_cpu.size(0), 1}, torch::kFloat32);
    auto pts4 = torch::cat({pts_cam_cpu, ones}, 1);  // [N, 4]
    auto pts4_t = pts4.t();  // [4, N]
    auto pts_world_4d = torch::matmul(c2w, pts4_t).t();  // [N, 4]
    auto pts_world = pts_world_4d.slice(1, 0, 3);  // [N, 3]
    
    // 移回 CUDA
    pts_world = pts_world.to(torch::kCUDA);
    
    // 提取颜色
    auto colors_flat = color_reshaped.reshape({height * width, 3});
    // 应用掩码选择有效像素的颜色
    std::vector<int64_t> valid_indices;
    auto valid_mask_cpu = valid_mask.cpu();
    for (int64_t i = 0; i < valid_mask_cpu.size(0); ++i) {
        if (valid_mask_cpu[i].item<bool>()) {
            valid_indices.push_back(i);
        }
    }
    auto indices_tensor = torch::tensor(valid_indices, torch::kInt64).to(colors_flat.device());
    auto colors = colors_flat.index_select(0, indices_tensor);
    
    // 计算mean_sq_dist（如果需要）
    Tensor mean_sq_dist;
    if (compute_mean_sq_dist) {
        // Projective Geometry方法
        float avg_focal = (fx + fy) / 2.0f;
        auto avg_focal_tensor = torch::tensor(avg_focal, torch::kFloat32).to(depth_z.device());
        auto scale_gaussian = depth_z / avg_focal_tensor;
        mean_sq_dist = scale_gaussian.pow(2);
    } else {
        mean_sq_dist = torch::zeros({pts_world.size(0)}, torch::kFloat32).to(pts_world.device());
    }
    
    return std::make_tuple(pts_world, colors, mean_sq_dist);
}

std::tuple<Tensor, Tensor> PointCloudUtils::extractPointCloudSimple(
    const Tensor& color,
    const Tensor& depth,
    const Tensor& intrinsics,
    const Tensor& w2c,
    const Tensor& mask)
{
    auto [points, colors, _] = extractPointCloud(color, depth, intrinsics, w2c, mask, false);
    return std::make_tuple(points, colors);
}

} // namespace isogs
