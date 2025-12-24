#pragma once

#include "slam/tracker.hpp"
#include "slam/mapper.hpp"
#include "datasets/dataset_base.hpp"
#include "core/gaussian_model.hpp"
#include <memory>
#include <string>

namespace isogs {

/**
 * @brief SLAM主循环
 */
class SLAMLoop {
public:
    struct Config {
        // 数据集配置
        std::string dataset_type = "replica";
        std::string dataset_path;
        std::string sequence;
        int start_frame = 0;
        int end_frame = -1;
        int stride = 1;
        
        // SLAM配置
        int map_every = 5;  // 每N帧进行一次Mapping
        int keyframe_every = 5;  // 每N帧添加一个关键帧
        bool use_gt_poses = false;  // 是否使用GT位姿
        std::string gaussian_distribution = "isotropic";  // "isotropic" or "anisotropic"
        
        // Tracking配置
        Tracker::Config tracking_config;
        
        // Mapping配置
        Mapper::Config mapping_config;
        
        // Checkpoint配置
        bool save_checkpoints = true;
        int checkpoint_interval = 50;
        std::string checkpoint_dir = "./checkpoints";
        std::string load_checkpoint_path = "";  // 如果非空，从该路径加载checkpoint
    };
    
    SLAMLoop(const Config& config);
    ~SLAMLoop() = default;
    
    /**
     * @brief 运行SLAM循环
     */
    void run();
    
    /**
     * @brief 加载checkpoint
     */
    bool loadCheckpoint(const std::string& path);
    
    /**
     * @brief 保存checkpoint
     * @param path 保存路径
     * @param time_idx 当前帧索引（用于文件命名和确定保存范围）
     */
    void saveCheckpoint(const std::string& path, int time_idx);
    
private:
    Config config_;
    std::unique_ptr<DatasetBase> dataset_;
    std::unique_ptr<GaussianModel> model_;
    std::unique_ptr<Tracker> tracker_;
    std::unique_ptr<Mapper> mapper_;
    
    // 相机姿态存储（用于checkpoint保存）
    Tensor cam_unnorm_rots_;  // [1, 4, num_frames] 相机旋转四元数
    Tensor cam_trans_;         // [1, 3, num_frames] 相机平移
    Tensor first_frame_w2c_;   // [4, 4] 第一帧的w2c矩阵
    Tensor intrinsics_;        // [3, 3] 相机内参
    std::vector<Tensor> gt_w2c_all_frames_;  // 所有帧的GT相机位姿 [4, 4] each
    std::vector<int> keyframe_time_indices_;  // 关键帧索引列表
    int org_width_;
    int org_height_;
    
    // 初始化第一帧
    void initializeFirstFrame();
    
    // 处理一帧
    void processFrame(int frame_idx);
    
    // 更新相机姿态存储
    void updateCameraPose(int frame_idx, const Tensor& w2c);
};

} // namespace isogs

