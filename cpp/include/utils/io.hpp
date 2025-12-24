#pragma once

#include "core/gaussian_model.hpp"
#include <string>
#include <map>
#include <vector>
#include <torch/torch.h>

namespace isogs {

/**
 * @brief I/O工具 - Checkpoint保存/加载
 */
class CheckpointIO {
public:
    /**
     * @brief 保存checkpoint到.npz文件（完整版本，包含相机参数）
     * @param path 保存路径
     * @param model 高斯模型
     * @param cam_unnorm_rots 相机旋转四元数 [1, 4, num_frames]
     * @param cam_trans 相机平移 [1, 3, num_frames]
     * @param intrinsics 相机内参 [3, 3]
     * @param first_frame_w2c 第一帧的w2c矩阵 [4, 4]
     * @param org_width 原始图像宽度
     * @param org_height 原始图像高度
     * @param gt_w2c_all_frames 所有帧的GT相机位姿 [num_frames, 4, 4]（可选）
     * @param keyframe_time_indices 关键帧索引列表（可选）
     */
    static void save(
        const std::string& path,
        const GaussianModel& model,
        const Tensor& cam_unnorm_rots,  // [1, 4, num_frames]
        const Tensor& cam_trans,          // [1, 3, num_frames]
        const Tensor& intrinsics,         // [3, 3]
        const Tensor& first_frame_w2c,    // [4, 4]
        int org_width,
        int org_height,
        const Tensor& gt_w2c_all_frames = Tensor(),  // [num_frames, 4, 4] 可选
        const std::vector<int>& keyframe_time_indices = {}  // 可选
    );
    
    /**
     * @brief 保存checkpoint到.npz文件（简化版本，用于不需要相机参数的工具）
     * @param path 保存路径
     * @param model 高斯模型
     * 
     * 注意：此版本会创建默认的相机参数（恒等变换，单帧）
     */
    static void save(const std::string& path, const GaussianModel& model);
    
    /**
     * @brief 从.npz文件加载checkpoint
     */
    static void load(const std::string& path, GaussianModel& model);
    
    /**
     * @brief 将旋转矩阵转换为四元数 (w, x, y, z)
     */
    static Tensor rotmat_to_quat(const Tensor& R);  // [3, 3] -> [4]
};

} // namespace isogs

