#include <cuda_runtime.h>
#include <torch/torch.h>
#include "../device_utils.cuh"

/**
 * @brief CUDA Kernel: 构建空间哈希表
 * 
 * 为每个高斯计算其所属的格子索引
 */
__global__ void build_hash_table_kernel(
    const float* means,      // [N, 3]
    int64_t* cell_indices,   // [N]
    int num_gaussians,
    float cell_size,
    int num_cells_x,
    int num_cells_y,
    int num_cells_z,
    float min_x,
    float min_y,
    float min_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;
    
    float x = means[idx * 3 + 0];
    float y = means[idx * 3 + 1];
    float z = means[idx * 3 + 2];
    
    // 计算格子坐标
    int cell_x = static_cast<int>((x - min_x) / cell_size);
    int cell_y = static_cast<int>((y - min_y) / cell_size);
    int cell_z = static_cast<int>((z - min_z) / cell_size);
    
    // 边界检查
    cell_x = max(0, min(cell_x, num_cells_x - 1));
    cell_y = max(0, min(cell_y, num_cells_y - 1));
    cell_z = max(0, min(cell_z, num_cells_z - 1));
    
    // 计算一维索引
    int64_t cell_idx = cell_z * num_cells_x * num_cells_y + 
                       cell_y * num_cells_x + 
                       cell_x;
    
    cell_indices[idx] = cell_idx;
}

/**
 * @brief CUDA Kernel: KNN查询
 * 
 * 对每个查询点，在其所在格子及26个邻域格子中查找K个最近邻
 */
__global__ void knn_query_kernel(
    const float* query_points,  // [M, 3]
    const float* means,         // [N, 3]
    const int64_t* cell_indices, // [N]
    const int* cell_offsets,    // [num_cells]
    const int* cell_counts,     // [num_cells]
    const int* cell_contents,   // [max_per_cell * num_cells]
    int64_t* knn_indices,      // [M, K]
    float* knn_distances,       // [M, K]
    int num_queries,
    int K,
    float cell_size,
    int num_cells_x,
    int num_cells_y,
    int num_cells_z,
    int max_gaussians_per_cell,
    float min_x,
    float min_y,
    float min_z
) {
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= num_queries) return;
    
    float qx = query_points[query_idx * 3 + 0];
    float qy = query_points[query_idx * 3 + 1];
    float qz = query_points[query_idx * 3 + 2];
    
    // 计算查询点所在的格子
    int cell_x = static_cast<int>((qx - min_x) / cell_size);
    int cell_y = static_cast<int>((qy - min_y) / cell_size);
    int cell_z = static_cast<int>((qz - min_z) / cell_size);
    
    cell_x = max(0, min(cell_x, num_cells_x - 1));
    cell_y = max(0, min(cell_y, num_cells_y - 1));
    cell_z = max(0, min(cell_z, num_cells_z - 1));
    
    // 存储候选最近邻（使用简单的堆或排序）
    struct Candidate {
        int64_t idx;
        float dist;
    };
    
    Candidate candidates[256];  // 最多256个候选
    int num_candidates = 0;
    
    // 遍历当前格子及26个邻域格子
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                int nx = cell_x + dx;
                int ny = cell_y + dy;
                int nz = cell_z + dz;
                
                if (nx < 0 || nx >= num_cells_x ||
                    ny < 0 || ny >= num_cells_y ||
                    nz < 0 || nz >= num_cells_z) {
                    continue;
                }
                
                int64_t neighbor_cell_idx = nz * num_cells_x * num_cells_y + 
                                            ny * num_cells_x + 
                                            nx;
                
                int count = cell_counts[neighbor_cell_idx];
                
                // 遍历该格子中的所有高斯
                for (int i = 0; i < count && num_candidates < 256; ++i) {
                    int64_t gaussian_idx = cell_contents[neighbor_cell_idx * max_gaussians_per_cell + i];
                    
                    float gx = means[gaussian_idx * 3 + 0];
                    float gy = means[gaussian_idx * 3 + 1];
                    float gz = means[gaussian_idx * 3 + 2];
                    
                    // 计算距离
                    float dx_val = qx - gx;
                    float dy_val = qy - gy;
                    float dz_val = qz - gz;
                    float dist = dx_val * dx_val + dy_val * dy_val + dz_val * dz_val;
                    
                    // 插入到候选列表（保持排序）
                    int insert_pos = num_candidates;
                    for (int j = 0; j < num_candidates; ++j) {
                        if (dist < candidates[j].dist) {
                            insert_pos = j;
                            break;
                        }
                    }
                    
                    if (insert_pos < K) {
                        // 移动后面的元素
                        for (int j = min(num_candidates, K - 1); j > insert_pos; --j) {
                            candidates[j] = candidates[j - 1];
                        }
                        candidates[insert_pos] = {gaussian_idx, dist};
                        if (num_candidates < K) num_candidates++;
                    }
                }
            }
        }
    }
    
    // 写入结果
    for (int k = 0; k < K && k < num_candidates; ++k) {
        knn_indices[query_idx * K + k] = candidates[k].idx;
        knn_distances[query_idx * K + k] = sqrtf(candidates[k].dist);
    }
    // 填充剩余位置
    for (int k = num_candidates; k < K; ++k) {
        knn_indices[query_idx * K + k] = -1;
        knn_distances[query_idx * K + k] = 1e10f;
    }
}

/**
 * @brief CUDA Kernel: 半径查询
 * 
 * 对查询点，在其所在格子及26个邻域格子中查找所有在指定半径内的高斯
 */
__global__ void radius_query_kernel(
    const float* query_point,  // [3]
    const float* means,        // [N, 3]
    const int* cell_offsets,   // [num_cells]
    const int* cell_counts,    // [num_cells]
    const int* cell_contents,  // [max_per_cell * num_cells]
    int64_t* result_indices,   // [max_results]
    int* result_count,         // [1] 输出实际结果数量
    float radius_sq,           // 半径的平方
    int max_results,           // 最大结果数量
    float cell_size,
    int num_cells_x,
    int num_cells_y,
    int num_cells_z,
    int max_gaussians_per_cell,
    float min_x,
    float min_y,
    float min_z
) {
    // 每个线程处理一个查询点（这里只有一个查询点）
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    float qx = query_point[0];
    float qy = query_point[1];
    float qz = query_point[2];
    
    // 计算查询点所在的格子
    int cell_x = static_cast<int>((qx - min_x) / cell_size);
    int cell_y = static_cast<int>((qy - min_y) / cell_size);
    int cell_z = static_cast<int>((qz - min_z) / cell_size);
    
    cell_x = max(0, min(cell_x, num_cells_x - 1));
    cell_y = max(0, min(cell_y, num_cells_y - 1));
    cell_z = max(0, min(cell_z, num_cells_z - 1));
    
    int count = 0;
    
    // 遍历当前格子及26个邻域格子
    for (int dx = -1; dx <= 1 && count < max_results; ++dx) {
        for (int dy = -1; dy <= 1 && count < max_results; ++dy) {
            for (int dz = -1; dz <= 1 && count < max_results; ++dz) {
                int nx = cell_x + dx;
                int ny = cell_y + dy;
                int nz = cell_z + dz;
                
                if (nx < 0 || nx >= num_cells_x ||
                    ny < 0 || ny >= num_cells_y ||
                    nz < 0 || nz >= num_cells_z) {
                    continue;
                }
                
                int64_t neighbor_cell_idx = nz * num_cells_x * num_cells_y + 
                                            ny * num_cells_x + 
                                            nx;
                
                int cell_count = cell_counts[neighbor_cell_idx];
                
                // 遍历该格子中的所有高斯
                for (int i = 0; i < cell_count && count < max_results; ++i) {
                    int64_t gaussian_idx = cell_contents[neighbor_cell_idx * max_gaussians_per_cell + i];
                    
                    float gx = means[gaussian_idx * 3 + 0];
                    float gy = means[gaussian_idx * 3 + 1];
                    float gz = means[gaussian_idx * 3 + 2];
                    
                    // 计算距离的平方
                    float dx_val = qx - gx;
                    float dy_val = qy - gy;
                    float dz_val = qz - gz;
                    float dist_sq = dx_val * dx_val + dy_val * dy_val + dz_val * dz_val;
                    
                    // 如果距离在半径内，添加到结果
                    if (dist_sq <= radius_sq) {
                        result_indices[count] = gaussian_idx;
                        count++;
                    }
                }
            }
        }
    }
    
    // 写入结果数量
    *result_count = count;
    
    // 填充剩余位置为-1（表示无效）
    for (int i = count; i < max_results; ++i) {
        result_indices[i] = -1;
    }
}

// 包装函数
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
) {
    int num_gaussians = means.size(0);
    int threads = 256;
    int blocks = (num_gaussians + threads - 1) / threads;
    
    build_hash_table_kernel<<<blocks, threads>>>(
        means.data_ptr<float>(),
        cell_indices.data_ptr<int64_t>(),
        num_gaussians,
        cell_size,
        num_cells_x,
        num_cells_y,
        num_cells_z,
        min_x,
        min_y,
        min_z
    );
}

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
) {
    int num_queries = query_points.size(0);
    int threads = 256;
    int blocks = (num_queries + threads - 1) / threads;
    
    knn_query_kernel<<<blocks, threads>>>(
        query_points.data_ptr<float>(),
        means.data_ptr<float>(),
        cell_indices.data_ptr<int64_t>(),
        cell_offsets.data_ptr<int>(),
        cell_counts.data_ptr<int>(),
        cell_contents.data_ptr<int>(),
        knn_indices.data_ptr<int64_t>(),
        knn_distances.data_ptr<float>(),
        num_queries,
        K,
        cell_size,
        num_cells_x,
        num_cells_y,
        num_cells_z,
        max_gaussians_per_cell,
        min_x,
        min_y,
        min_z
    );
}

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
) {
    float radius_sq = radius * radius;
    
    // 使用单个线程块，单个线程（因为只有一个查询点）
    radius_query_kernel<<<1, 1>>>(
        query_point.data_ptr<float>(),
        means.data_ptr<float>(),
        cell_offsets.data_ptr<int>(),
        cell_counts.data_ptr<int>(),
        cell_contents.data_ptr<int>(),
        result_indices.data_ptr<int64_t>(),
        result_count.data_ptr<int>(),
        radius_sq,
        max_results,
        cell_size,
        num_cells_x,
        num_cells_y,
        num_cells_z,
        max_gaussians_per_cell,
        min_x,
        min_y,
        min_z
    );
}

} // extern "C"
