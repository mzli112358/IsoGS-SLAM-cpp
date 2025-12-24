#include "utils/keyframe_selection.hpp"
#include <torch/torch.h>
#include <algorithm>
#include <random>
#include <cmath>

namespace isogs {

Tensor getPointCloudFromSampledPixels(
    const Tensor& depth,
    const Tensor& intrinsics,
    const Tensor& w2c,
    const Tensor& sampled_indices)
{
    // 提取内参
    // 安全地从CUDA tensor提取内参（先移到CPU）
    auto intrinsics_cpu = intrinsics.cpu();
    float CX = intrinsics_cpu[0][2].item<float>();
    float CY = intrinsics_cpu[1][2].item<float>();
    float FX = intrinsics_cpu[0][0].item<float>();
    float FY = intrinsics_cpu[1][1].item<float>();
    
    int num_samples = sampled_indices.size(0);
    
    // 提取采样像素的坐标 (y, x)
    auto y_coords = sampled_indices.select(1, 0);  // [N]
    auto x_coords = sampled_indices.select(1, 1);  // [N]
    
    // 计算归一化坐标
    auto xx = (x_coords.to(torch::kFloat32) - CX) / FX;  // [N]
    auto yy = (y_coords.to(torch::kFloat32) - CY) / FY;  // [N]
    
    // 获取深度值 (depth是[H, W]，需要Long类型索引)
    auto y_coords_long = y_coords.to(torch::kInt64);
    auto x_coords_long = x_coords.to(torch::kInt64);
    auto depth_vals = depth.index({y_coords_long, x_coords_long});  // [N]
    
    // 构建相机坐标系下的点
    auto pts_cam = torch::stack({
        xx * depth_vals,
        yy * depth_vals,
        depth_vals
    }, 1);  // [N, 3]
    
    // 转换为齐次坐标
    auto ones = torch::ones({num_samples, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(kCUDA));
    auto pts4 = torch::cat({pts_cam, ones}, 1);  // [N, 4]
    
    // 转换到世界坐标系
    // 注意：在 CPU 上计算逆矩阵，避免 CUBLAS 初始化问题
    Tensor c2w;
    if (w2c.is_cuda()) {
        auto w2c_cpu = w2c.cpu();
        c2w = w2c_cpu.inverse().to(torch::kCUDA);
    } else {
        c2w = w2c.inverse().to(torch::kCUDA);
    }
    // 在 CPU 上执行矩阵乘法，然后移回 CUDA
    auto pts4_cpu = pts4.cpu();
    auto pts4_t_cpu = pts4_cpu.t();  // [4, N]
    auto c2w_cpu = c2w.cpu();
    auto pts_world_4d_cpu = torch::matmul(c2w_cpu, pts4_t_cpu).t();  // [N, 4]
    auto pts_world = pts_world_4d_cpu.to(torch::kCUDA);
    auto pts = pts_world.slice(1, 0, 3);  // [N, 3]
    
    // 移除在相机原点的点（通过检查点是否接近原点）
    // 计算每个点到原点的距离
    auto dist_to_origin = torch::norm(pts, 2, 1);  // [N]
    auto valid_mask = dist_to_origin > 1e-4f;  // 距离原点大于阈值的点
    
    // 过滤有效点
    auto valid_indices = torch::where(valid_mask)[0];
    // 确保 valid_indices 在正确的设备上
    if (valid_indices.is_cuda()) {
        auto valid_count = valid_indices.size(0);
        if (valid_count > 0) {
            // 确保索引在 CPU 上用于 index 操作
            auto valid_indices_cpu = valid_indices.cpu();
            return pts.index({valid_indices_cpu.to(torch::kInt64)});
        } else {
            return torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(kCUDA));
        }
    } else {
        if (valid_indices.size(0) > 0) {
            return pts.index({valid_indices.to(torch::kInt64)});
        } else {
            return torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(kCUDA));
        }
    }
}

std::vector<int> keyframeSelectionOverlap(
    const Tensor& current_depth,
    const Tensor& current_w2c,
    const Tensor& intrinsics,
    const std::vector<Keyframe>& keyframes,
    int k,
    int pixels)
{
    if (keyframes.empty()) {
        return {};
    }
    
    int64_t height = current_depth.size(0);
    int64_t width = current_depth.size(1);
    
    // 随机采样有效深度像素
    auto valid_mask = current_depth > 0;
    auto valid_indices_2d = torch::where(valid_mask);
    
    // 确保在 CPU 上获取大小，避免 CUDA 内存访问问题
    auto valid_y_cpu = valid_indices_2d[0].cpu();
    // 安全地从CPU tensor获取size（size()不会访问CUDA内存，但为了安全起见确保在CPU上）
    int64_t num_valid_int64 = valid_y_cpu.size(0);
    int num_valid = static_cast<int>(num_valid_int64);
    if (num_valid == 0) {
        return {};
    }
    
    int num_samples = std::min(pixels, num_valid);
    
    // 随机选择索引（在 CPU 上生成，避免 CUDA 内存访问问题）
    auto rand_indices = torch::randperm(num_valid, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
        .slice(0, 0, num_samples);
    
    // 获取采样的像素坐标（确保索引在 CPU 上）
    // 注意：valid_y_cpu 和 valid_x_cpu 已经在上面定义了
    auto valid_x_cpu = valid_indices_2d[1].cpu();
    auto sampled_y = valid_y_cpu.index({rand_indices}).to(torch::kCUDA);
    auto sampled_x = valid_x_cpu.index({rand_indices}).to(torch::kCUDA);
    auto sampled_indices = torch::stack({sampled_y, sampled_x}, 1);  // [N, 2]
    
    // 反投影到3D点云
    auto pts_3d = getPointCloudFromSampledPixels(
        current_depth, intrinsics, current_w2c, sampled_indices);
    
    if (pts_3d.size(0) == 0) {
        return {};
    }
    
    // 计算每个关键帧的重叠度
    std::vector<std::pair<int, float>> keyframe_overlaps;
    
    for (size_t keyframe_id = 0; keyframe_id < keyframes.size(); ++keyframe_id) {
        const auto& keyframe = keyframes[keyframe_id];
        auto est_w2c = keyframe.camera.getW2C();
        
        // 将3D点转换到关键帧的相机坐标系
        auto ones = torch::ones({pts_3d.size(0), 1}, 
                                torch::TensorOptions().dtype(torch::kFloat32).device(kCUDA));
        auto pts4 = torch::cat({pts_3d, ones}, 1);  // [N, 4]
        // 在 CPU 上执行矩阵乘法，避免 CUBLAS 初始化问题
        auto pts4_cpu = pts4.cpu();
        auto pts4_t_cpu = pts4_cpu.t();  // [4, N]
        auto est_w2c_cpu = est_w2c.cpu();
        auto transformed_pts_4d_cpu = torch::matmul(est_w2c_cpu, pts4_t_cpu).t();  // [N, 4]
        auto transformed_pts_4d = transformed_pts_4d_cpu.to(torch::kCUDA);
        auto transformed_pts_3d = transformed_pts_4d.slice(1, 0, 3);  // [N, 3]
        
        // 投影到图像空间（在 CPU 上执行）
        auto transformed_pts_3d_cpu = transformed_pts_3d.cpu();
        auto transformed_pts_3d_t_cpu = transformed_pts_3d_cpu.t();  // [3, N]
        auto intrinsics_cpu = intrinsics.cpu();
        auto points_2d_homo_cpu = torch::matmul(intrinsics_cpu, transformed_pts_3d_t_cpu);  // [3, N]
        auto points_2d_homo = points_2d_homo_cpu.t().to(torch::kCUDA);  // [N, 3]
        
        auto points_z = points_2d_homo.select(1, 2) + 1e-5f;  // [N]
        auto points_2d = points_2d_homo.slice(1, 0, 2) / points_z.unsqueeze(1);  // [N, 2]
        
        // 过滤在图像外的点
        int edge = 20;
        auto x_coords = points_2d.select(1, 0);  // [N]
        auto y_coords = points_2d.select(1, 1);  // [N]
        
        auto mask = (x_coords > edge) & (x_coords < (width - edge)) &
                   (y_coords > edge) & (y_coords < (height - edge)) &
                   (points_z > 0);
        
        // 计算重叠度（确保在 CPU 上获取值）
        auto mask_sum_cpu = mask.sum().cpu();
        float points_2d_size = static_cast<float>(points_2d.size(0));
        float percent_inside = mask_sum_cpu.item<float>() / points_2d_size;
        
        if (percent_inside > 0.0f) {
            keyframe_overlaps.push_back({static_cast<int>(keyframe_id), percent_inside});
        }
    }
    
    // 按重叠度排序（降序）
    std::sort(keyframe_overlaps.begin(), keyframe_overlaps.end(),
              [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                  return a.second > b.second;
              });
    
    // 选择top-k（随机打乱后选择）
    std::vector<int> selected_indices;
    for (const auto& pair : keyframe_overlaps) {
        selected_indices.push_back(pair.first);
    }
    
    // 随机打乱
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(selected_indices.begin(), selected_indices.end(), g);
    
    // 选择前k个
    if (static_cast<int>(selected_indices.size()) > k) {
        selected_indices.resize(k);
    }
    
    return selected_indices;
}

} // namespace isogs

