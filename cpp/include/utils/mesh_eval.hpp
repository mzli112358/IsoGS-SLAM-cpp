#pragma once

#include "core/types.hpp"
#include <torch/torch.h>
#include <string>
#include <vector>

namespace isogs {

/**
 * @brief 网格几何评估结果
 */
struct MeshEvalResult {
    float chamfer_distance = 0.0f;
    float f_score = 0.0f;
    float normal_consistency = 0.0f;
};

/**
 * @brief 网格几何评估工具
 */
class MeshEvaluator {
public:
    /**
     * @brief 计算Chamfer Distance
     * 
     * @param pred_points 预测点云 [N, 3]
     * @param gt_points GT点云 [M, 3]
     * @return float Chamfer Distance
     */
    static float computeChamferDistance(
        const Tensor& pred_points,
        const Tensor& gt_points
    );
    
    /**
     * @brief 计算F-score
     * 
     * @param pred_points 预测点云 [N, 3]
     * @param gt_points GT点云 [M, 3]
     * @param threshold 距离阈值
     * @return float F-score
     */
    static float computeFScore(
        const Tensor& pred_points,
        const Tensor& gt_points,
        float threshold = 0.01f
    );
    
    /**
     * @brief 计算法线一致性
     * 
     * @param pred_points 预测点云 [N, 3]
     * @param pred_normals 预测法线 [N, 3]
     * @param gt_points GT点云 [M, 3]
     * @param gt_normals GT法线 [M, 3]
     * @return float 法线一致性分数
     */
    static float computeNormalConsistency(
        const Tensor& pred_points,
        const Tensor& pred_normals,
        const Tensor& gt_points,
        const Tensor& gt_normals
    );
    
    /**
     * @brief 完整的网格几何评估流程
     * 
     * @param pred_mesh_path 预测网格PLY文件路径
     * @param gt_mesh_path GT网格PLY文件路径
     * @return MeshEvalResult 评估结果
     */
    static MeshEvalResult evaluateMeshGeometry(
        const std::string& pred_mesh_path,
        const std::string& gt_mesh_path
    );
};

} // namespace isogs

