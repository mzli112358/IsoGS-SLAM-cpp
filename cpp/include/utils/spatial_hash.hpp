#pragma once

#include "core/types.hpp"
#include <torch/torch.h>
#include <vector>
#include <cstdint>

namespace isogs {

/**
 * @brief 空间哈希表 - 用于加速KNN查询
 * 
 * 将3D空间划分为固定大小的格子，每个高斯分配到对应的格子
 * 查询时只遍历当前格子及26个邻域格子
 */
class SpatialHash {
public:
    struct Config {
        float cell_size = 0.1f;  // 格子大小（米）
        int max_gaussians_per_cell = 256;  // 每个格子最多存储的高斯数量
    };
    
    SpatialHash(const Config& config);
    ~SpatialHash() = default;
    
    /**
     * @brief 构建哈希表
     * @param means 高斯中心位置 [N, 3]
     */
    void build(const Tensor& means);
    
    /**
     * @brief 设置means（用于KNN查询）
     * @param means 高斯中心位置 [N, 3]
     */
    void setMeans(const Tensor& means);
    
    /**
     * @brief 查询K个最近邻
     * @param query_points 查询点 [M, 3]
     * @param K 最近邻数量
     * @return 最近邻索引 [M, K]
     */
    Tensor queryKNN(const Tensor& query_points, int K) const;
    
    /**
     * @brief 查询指定半径内的所有高斯
     * @param query_point 查询点 [3]
     * @param radius 查询半径
     * @return 索引列表
     */
    std::vector<int64_t> queryRadius(const Tensor& query_point, float radius) const;
    
    /**
     * @brief 清空哈希表
     */
    void clear();
    
private:
    Config config_;
    
    // 哈希表数据结构（在GPU上）
    Tensor cell_indices_;  // 每个高斯所属的格子索引 [N]
    Tensor cell_offsets_;  // 每个格子的起始偏移 [num_cells]
    Tensor cell_counts_;   // 每个格子的高斯数量 [num_cells]
    Tensor cell_contents_; // 格子内容（高斯索引） [max_gaussians_per_cell * num_cells]
    Tensor means_;         // 高斯中心位置 [N, 3]（用于KNN查询）
    
    int num_cells_x_, num_cells_y_, num_cells_z_;
    int num_cells_;
    float min_x_, min_y_, min_z_;  // 边界框最小值
    
    // 辅助函数：计算3D位置对应的格子索引（在.cu文件中实现，这里只是声明）
    // 注意：__device__ 只能在.cu文件中使用
    int64_t getCellIndex(float x, float y, float z) const;
    
    // 辅助函数：获取26个邻域格子的索引
    void getNeighborCells(int cell_idx, std::vector<int>& neighbors) const;
};

} // namespace isogs

