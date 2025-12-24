#include "utils/spatial_hash.hpp"
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <iostream>

// CUDA Kernel声明
extern "C" {
void launch_build_hash_table(
    torch::Tensor means,
    torch::Tensor cell_indices,
    float cell_size,
    int num_cells_x,
    int num_cells_y,
    int num_cells_z,
    float min_x,
    float min_y,
    float min_z
);

void launch_knn_query(
    torch::Tensor query_points,
    torch::Tensor means,
    torch::Tensor cell_indices,
    torch::Tensor cell_offsets,
    torch::Tensor cell_counts,
    torch::Tensor cell_contents,
    torch::Tensor knn_indices,
    torch::Tensor knn_distances,
    int K,
    float cell_size,
    int num_cells_x,
    int num_cells_y,
    int num_cells_z,
    int max_gaussians_per_cell,
    float min_x,
    float min_y,
    float min_z
);

void launch_radius_query(
    torch::Tensor query_point,
    torch::Tensor means,
    torch::Tensor cell_offsets,
    torch::Tensor cell_counts,
    torch::Tensor cell_contents,
    torch::Tensor result_indices,
    torch::Tensor result_count,
    float radius,
    int max_results,
    float cell_size,
    int num_cells_x,
    int num_cells_y,
    int num_cells_z,
    int max_gaussians_per_cell,
    float min_x,
    float min_y,
    float min_z
);
}
#include <algorithm>
#include <cmath>

namespace isogs {

SpatialHash::SpatialHash(const Config& config)
    : config_(config)
    , num_cells_x_(0)
    , num_cells_y_(0)
    , num_cells_z_(0)
    , num_cells_(0)
    , min_x_(0.0f)
    , min_y_(0.0f)
    , min_z_(0.0f)
{
}

void SpatialHash::build(const Tensor& means)
{
    // 计算边界框
    auto min_vals = std::get<0>(means.min(0));
    auto max_vals = std::get<0>(means.max(0));
    
    min_x_ = min_vals[0].item<float>();
    min_y_ = min_vals[1].item<float>();
    min_z_ = min_vals[2].item<float>();
    float max_x = max_vals[0].item<float>();
    float max_y = max_vals[1].item<float>();
    float max_z = max_vals[2].item<float>();
    
    // 计算格子数量
    num_cells_x_ = static_cast<int>(std::ceil((max_x - min_x_) / config_.cell_size)) + 1;
    num_cells_y_ = static_cast<int>(std::ceil((max_y - min_y_) / config_.cell_size)) + 1;
    num_cells_z_ = static_cast<int>(std::ceil((max_z - min_z_) / config_.cell_size)) + 1;
    num_cells_ = num_cells_x_ * num_cells_y_ * num_cells_z_;
    
    int num_gaussians = means.size(0);
    
    // 分配内存
    cell_indices_ = torch::zeros({num_gaussians}, torch::kInt64).to(means.device());
    cell_offsets_ = torch::zeros({num_cells_}, torch::kInt32).to(means.device());
    cell_counts_ = torch::zeros({num_cells_}, torch::kInt32).to(means.device());
    cell_contents_ = torch::zeros({num_cells_ * config_.max_gaussians_per_cell}, 
                                  torch::kInt32).to(means.device());
    
    // 保存means用于KNN查询
    means_ = means.clone();
    
    // 调用CUDA Kernel构建哈希表
    launch_build_hash_table(
        means,
        cell_indices_,
        config_.cell_size,
        num_cells_x_,
        num_cells_y_,
        num_cells_z_,
        min_x_,
        min_y_,
        min_z_
    );
    
    // 使用LibTorch进行排序和计数，填充cell_offsets_和cell_contents_
    // 方案：使用torch::sort和torch::unique等操作
    
    // 1. 对cell_indices进行排序，同时保持原始索引
    auto sorted_result = torch::sort(cell_indices_);
    auto sorted_cell_indices = std::get<0>(sorted_result);  // 排序后的cell索引
    auto sort_indices = std::get<1>(sorted_result);  // 排序后的原始索引
    
    // 2. 计算每个cell的计数
    cell_counts_.zero_();
    
    // 使用torch::unique_consecutive找到连续相同值的区间
    // 然后计算每个区间的长度
    int64_t current_cell = -1;
    int64_t start_idx = 0;
    
    // 遍历排序后的cell_indices，计算每个cell的数量（先移到CPU）
    auto sorted_cell_indices_cpu = sorted_cell_indices.cpu();
    for (int64_t i = 0; i < num_gaussians; ++i) {
        int64_t cell_idx = sorted_cell_indices_cpu[i].item<int64_t>();
        
        if (cell_idx != current_cell) {
            // 新的cell开始
            if (current_cell >= 0) {
                // 更新前一个cell的计数
                int count = static_cast<int>(i - start_idx);
                cell_counts_[current_cell] = count;
            }
            current_cell = cell_idx;
            start_idx = i;
        }
    }
    // 处理最后一个cell
    if (current_cell >= 0) {
        int count = static_cast<int>(num_gaussians - start_idx);
        cell_counts_[current_cell] = count;
    }
    
    // 3. 计算每个cell的偏移量（前缀和）
    cell_offsets_.zero_();
    int current_offset = 0;
    // 安全地从CUDA tensor提取计数（先移到CPU）
    auto cell_counts_cpu = cell_counts_.cpu();
    for (int64_t i = 0; i < num_cells_; ++i) {
        cell_offsets_[i] = current_offset;
        int count = cell_counts_cpu[i].item<int>();
        current_offset += count;
    }
    
    // 4. 填充cell_contents_（每个cell存储的高斯索引）
    cell_contents_.zero_();
    cell_contents_.fill_(-1);  // 初始化为-1表示空
    
    // 重新遍历排序后的数据，填充cell_contents
    current_cell = -1;
    int cell_pos = 0;
    
    // 安全地从CUDA tensor提取索引（先移到CPU）
    auto sort_indices_cpu = sort_indices.cpu();
    for (int64_t i = 0; i < num_gaussians; ++i) {
        int64_t cell_idx = sorted_cell_indices_cpu[i].item<int64_t>();
        int64_t gaussian_idx = sort_indices_cpu[i].item<int64_t>();
        
        if (cell_idx != current_cell) {
            current_cell = cell_idx;
            cell_pos = 0;
        }
        
        // 检查是否超出每个cell的最大容量
        if (cell_pos < config_.max_gaussians_per_cell) {
            int64_t content_idx = cell_idx * config_.max_gaussians_per_cell + cell_pos;
            cell_contents_[content_idx] = gaussian_idx;
            cell_pos++;
        }
    }
}

Tensor SpatialHash::queryKNN(const Tensor& query_points, int K) const
{
    int num_queries = query_points.size(0);
    
    if (!means_.defined() || means_.size(0) == 0) {
        // 如果没有means，返回空结果
        return torch::zeros({num_queries, K}, torch::kInt64).to(query_points.device()).fill_(-1);
    }
    
    auto knn_indices = torch::zeros({num_queries, K}, torch::kInt64).to(query_points.device());
    auto knn_distances = torch::zeros({num_queries, K}, torch::kFloat32).to(query_points.device());
    
    // 调用CUDA Kernel进行KNN查询
    launch_knn_query(
        query_points,
        means_,
        cell_indices_,
        cell_offsets_,
        cell_counts_,
        cell_contents_,
        knn_indices,
        knn_distances,
        K,
        config_.cell_size,
        num_cells_x_,
        num_cells_y_,
        num_cells_z_,
        config_.max_gaussians_per_cell,
        min_x_,
        min_y_,
        min_z_
    );
    
    return knn_indices;
}

void SpatialHash::setMeans(const Tensor& means)
{
    means_ = means.clone();
}

std::vector<int64_t> SpatialHash::queryRadius(const Tensor& query_point, float radius) const
{
    // 检查输入
    if (!means_.defined() || means_.size(0) == 0) {
        return {};
    }
    
    if (query_point.size(0) != 3) {
        throw std::runtime_error("query_point must have shape [3]");
    }
    
    // 确保查询点在GPU上
    auto query_point_gpu = query_point.to(means_.device()).contiguous();
    
    // 分配结果缓冲区（固定大小256）
    const int max_results = 256;
    auto result_indices = torch::zeros({max_results}, torch::kInt64).to(means_.device());
    auto result_count = torch::zeros({1}, torch::kInt32).to(means_.device());
    
    // 调用CUDA Kernel进行半径查询
    launch_radius_query(
        query_point_gpu,
        means_,
        cell_offsets_,
        cell_counts_,
        cell_contents_,
        result_indices,
        result_count,
        radius,
        max_results,
        config_.cell_size,
        num_cells_x_,
        num_cells_y_,
        num_cells_z_,
        config_.max_gaussians_per_cell,
        min_x_,
        min_y_,
        min_z_
    );
    
    // 同步CUDA操作
    cudaDeviceSynchronize();
    
    // 获取结果数量
    int count = result_count.cpu().item<int>();
    
    // 如果结果数量达到最大值，发出警告
    if (count >= max_results) {
        std::cerr << "Warning: radius query returned maximum number of results (" 
                  << max_results << "). Some results may be missing." << std::endl;
    }
    
    // 将结果从GPU拷贝到CPU
    auto result_indices_cpu = result_indices.cpu();
    const int64_t* indices_data = result_indices_cpu.data_ptr<int64_t>();
    
    // 收集有效结果（去除-1占位符）
    std::vector<int64_t> results;
    results.reserve(count);
    
    for (int i = 0; i < count; ++i) {
        int64_t idx = indices_data[i];
        if (idx >= 0) {  // 有效索引
            results.push_back(idx);
        }
    }
    
    return results;
}

void SpatialHash::clear()
{
    cell_indices_ = Tensor();
    cell_offsets_ = Tensor();
    cell_counts_ = Tensor();
    cell_contents_ = Tensor();
}

} // namespace isogs

