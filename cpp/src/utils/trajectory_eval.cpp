#include "utils/trajectory_eval.hpp"
#include <torch/torch.h>
#include <cmath>
#include <algorithm>

namespace isogs {

std::tuple<Tensor, Tensor, Tensor> alignTrajectories(
    const Tensor& model_traj,
    const Tensor& data_traj)
{
    // 计算中心化后的轨迹
    auto model_mean = model_traj.mean(1, true);  // [3, 1]
    auto data_mean = data_traj.mean(1, true);    // [3, 1]
    
    auto model_centered = model_traj - model_mean;  // [3, N]
    auto data_centered = data_traj - data_mean;     // [3, N]
    
    // 计算协方差矩阵 W = sum(model_centered @ data_centered^T)
    Tensor W = torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(kCUDA));
    int N = model_traj.size(1);
    
    for (int i = 0; i < N; ++i) {
        auto model_col = model_centered.select(1, i).unsqueeze(1);  // [3, 1]
        auto data_col = data_centered.select(1, i).unsqueeze(1);    // [3, 1]
        W = W + torch::matmul(model_col, data_col.transpose(0, 1));
    }
    
    // SVD分解
    auto [U, S, Vh] = torch::linalg_svd(W, true);
    
    // 计算旋转矩阵
    Tensor R = torch::matmul(U, Vh.transpose(0, 1));  // [3, 3]
    
    // 检查行列式，如果为负，需要调整
    auto det = torch::det(R);
    if (det.item<float>() < 0) {
        // 修改最后一个奇异值
        auto S_adj = torch::eye(3, torch::TensorOptions().dtype(torch::kFloat32).device(kCUDA));
        S_adj[2][2] = -1.0f;
        R = torch::matmul(torch::matmul(U, S_adj), Vh.transpose(0, 1));
    }
    
    // 计算平移向量
    Tensor t = data_mean - torch::matmul(R, model_mean);  // [3, 1]
    
    // 计算对齐后的轨迹
    Tensor model_aligned = torch::matmul(R, model_traj) + t;  // [3, N]
    
    // 计算平移误差
    Tensor alignment_error = model_aligned - data_traj;  // [3, N]
    Tensor trans_error = torch::norm(alignment_error, 2, 0);  // [N]
    
    return std::make_tuple(R, t, trans_error);
}

float evaluateATE(
    const std::vector<Tensor>& gt_traj,
    const std::vector<Tensor>& est_traj)
{
    if (gt_traj.size() != est_traj.size() || gt_traj.empty()) {
        return -1.0f;  // 错误值
    }
    
    // 提取平移部分
    std::vector<Tensor> gt_translations, est_translations;
    for (size_t i = 0; i < gt_traj.size(); ++i) {
        gt_translations.push_back(gt_traj[i].slice(0, 0, 3).select(1, 3));   // [3]
        est_translations.push_back(est_traj[i].slice(0, 0, 3).select(1, 3)); // [3]
    }
    
    // 堆叠成矩阵 [3, N]
    Tensor gt_traj_pts = torch::stack(gt_translations, 1);   // [3, N]
    Tensor est_traj_pts = torch::stack(est_translations, 1); // [3, N]
    
    // 对齐轨迹
    auto [R, t, trans_error] = alignTrajectories(est_traj_pts, gt_traj_pts);
    
    // 计算平均平移误差
    float avg_trans_error = trans_error.mean().item<float>();
    
    return avg_trans_error;
}

} // namespace isogs

