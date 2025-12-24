#include "utils/mesh_eval.hpp"
#include "utils/spatial_hash.hpp"
#include <torch/torch.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>

namespace isogs {

float MeshEvaluator::computeChamferDistance(
    const Tensor& pred_points,
    const Tensor& gt_points)
{
    if (pred_points.size(0) == 0 || gt_points.size(0) == 0) {
        return 0.0f;
    }
    
    // 构建空间哈希表用于加速
    SpatialHash::Config hash_config;
    hash_config.cell_size = 0.01f;
    SpatialHash spatial_hash(hash_config);
    spatial_hash.build(gt_points);
    spatial_hash.setMeans(gt_points);
    
    // 计算pred到gt的距离
    float pred_to_gt_sum = 0.0f;
    for (int64_t i = 0; i < pred_points.size(0); ++i) {
        auto query_point = pred_points[i].unsqueeze(0);  // [1, 3]
        auto neighbors = spatial_hash.queryKNN(query_point, 1);  // [1, 1]
        if (neighbors.size(0) > 0 && neighbors.size(1) > 0) {
            // 安全地从CUDA tensor提取索引（先移到CPU）
            auto neighbors_cpu = neighbors.cpu();
            int64_t nearest_idx = neighbors_cpu[0][0].item<int64_t>();
            auto dist_vec = pred_points[i] - gt_points[nearest_idx];
            float dist = torch::norm(dist_vec, 2).detach().cpu().item<float>();
            pred_to_gt_sum += dist;
        }
    }
    float pred_to_gt = pred_to_gt_sum / pred_points.size(0);
    
    // 计算gt到pred的距离（需要为pred构建空间哈希）
    SpatialHash pred_hash(hash_config);
    pred_hash.build(pred_points);
    pred_hash.setMeans(pred_points);
    float gt_to_pred_sum = 0.0f;
    for (int64_t i = 0; i < gt_points.size(0); ++i) {
        auto query_point = gt_points[i].unsqueeze(0);  // [1, 3]
        auto neighbors = pred_hash.queryKNN(query_point, 1);  // [1, 1]
        if (neighbors.size(0) > 0 && neighbors.size(1) > 0) {
            // 安全地从CUDA tensor提取索引（先移到CPU）
            auto neighbors_cpu = neighbors.cpu();
            int64_t nearest_idx = neighbors_cpu[0][0].item<int64_t>();
            auto dist_vec = gt_points[i] - pred_points[nearest_idx];
            float dist = torch::norm(dist_vec, 2).detach().cpu().item<float>();
            gt_to_pred_sum += dist;
        }
    }
    float gt_to_pred = gt_to_pred_sum / gt_points.size(0);
    
    return (pred_to_gt + gt_to_pred) / 2.0f;
}

float MeshEvaluator::computeFScore(
    const Tensor& pred_points,
    const Tensor& gt_points,
    float threshold)
{
    if (pred_points.size(0) == 0 || gt_points.size(0) == 0) {
        return 0.0f;
    }
    
    // 构建空间哈希表
    SpatialHash::Config hash_config;
    hash_config.cell_size = threshold;
    SpatialHash spatial_hash(hash_config);
    spatial_hash.build(gt_points);
    spatial_hash.setMeans(gt_points);
    
    // 计算precision: pred中距离gt < threshold的点比例
    int correct_pred = 0;
    for (int64_t i = 0; i < pred_points.size(0); ++i) {
        auto query_point = pred_points[i].unsqueeze(0);
        auto neighbors = spatial_hash.queryKNN(query_point, 1);
        if (neighbors.size(0) > 0 && neighbors.size(1) > 0) {
            // 安全地从CUDA tensor提取索引（先移到CPU）
            auto neighbors_cpu = neighbors.cpu();
            int64_t nearest_idx = neighbors_cpu[0][0].item<int64_t>();
            auto dist_vec = pred_points[i] - gt_points[nearest_idx];
            float dist = torch::norm(dist_vec, 2).detach().cpu().item<float>();
            if (dist < threshold) {
                correct_pred++;
            }
        }
    }
    float precision = static_cast<float>(correct_pred) / pred_points.size(0);
    
    // 计算recall: gt中距离pred < threshold的点比例
    SpatialHash pred_hash(hash_config);
    pred_hash.build(pred_points);
    pred_hash.setMeans(pred_points);
    int correct_gt = 0;
    for (int64_t i = 0; i < gt_points.size(0); ++i) {
        auto query_point = gt_points[i].unsqueeze(0);
        auto neighbors = pred_hash.queryKNN(query_point, 1);
        if (neighbors.size(0) > 0 && neighbors.size(1) > 0) {
            // 安全地从CUDA tensor提取索引（先移到CPU）
            auto neighbors_cpu = neighbors.cpu();
            int64_t nearest_idx = neighbors_cpu[0][0].item<int64_t>();
            auto dist_vec = gt_points[i] - pred_points[nearest_idx];
            float dist = torch::norm(dist_vec, 2).detach().cpu().item<float>();
            if (dist < threshold) {
                correct_gt++;
            }
        }
    }
    float recall = static_cast<float>(correct_gt) / gt_points.size(0);
    
    // 计算F-score
    if (precision + recall == 0.0f) {
        return 0.0f;
    }
    return 2.0f * precision * recall / (precision + recall);
}

float MeshEvaluator::computeNormalConsistency(
    const Tensor& pred_points,
    const Tensor& pred_normals,
    const Tensor& gt_points,
    const Tensor& gt_normals)
{
    if (pred_points.size(0) == 0 || gt_points.size(0) == 0) {
        return 0.0f;
    }
    
    // 构建空间哈希表
    SpatialHash::Config hash_config;
    hash_config.cell_size = 0.01f;
    SpatialHash spatial_hash(hash_config);
    spatial_hash.build(gt_points);
    spatial_hash.setMeans(gt_points);
    
    float consistency_sum = 0.0f;
    int valid_count = 0;
    
    for (int64_t i = 0; i < pred_points.size(0); ++i) {
        auto query_point = pred_points[i].unsqueeze(0);
        auto neighbors = spatial_hash.queryKNN(query_point, 1);
        
        if (neighbors.size(0) > 0 && neighbors.size(1) > 0) {
            // 安全地从CUDA tensor提取索引（先移到CPU）
            auto neighbors_cpu = neighbors.cpu();
            int64_t nearest_idx = neighbors_cpu[0][0].item<int64_t>();
            auto pred_normal = pred_normals[i];
            auto gt_normal = gt_normals[nearest_idx];
            
            // 计算法线点积（余弦相似度）
            float dot_product = (pred_normal * gt_normal).sum().detach().cpu().item<float>();
            consistency_sum += std::abs(dot_product);
            valid_count++;
        }
    }
    
    return valid_count > 0 ? consistency_sum / valid_count : 0.0f;
}

MeshEvalResult MeshEvaluator::evaluateMeshGeometry(
    const std::string& pred_mesh_path,
    const std::string& gt_mesh_path)
{
    MeshEvalResult result;
    
    // TODO: 实现PLY文件读取
    // 这里需要实现PLY文件解析，提取点云和法线
    // 可以使用简单的PLY解析器或Open3D C++
    
    return result;
}

} // namespace isogs

