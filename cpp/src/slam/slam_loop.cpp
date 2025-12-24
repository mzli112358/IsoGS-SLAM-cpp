#include "slam/slam_loop.hpp"
#include "datasets/replica_dataset.hpp"
#include "core/camera.hpp"
#include "utils/pointcloud.hpp"
#include "utils/io.hpp"
#include <torch/torch.h>
#include <iostream>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

namespace isogs {

SLAMLoop::SLAMLoop(const Config& config)
    : config_(config)
    , tracker_(std::make_unique<Tracker>(config_.tracking_config))
{
    // 同步gaussian_distribution配置到mapper配置
    config_.mapping_config.gaussian_distribution = config_.gaussian_distribution;
    
    // 创建mapper（使用更新后的配置）
    mapper_ = std::make_unique<Mapper>(config_.mapping_config);
    // 创建数据集
    if (config_.dataset_type == "replica") {
        dataset_ = std::make_unique<ReplicaDataset>(
            config_.dataset_path,
            config_.sequence,
            config_.start_frame,
            config_.end_frame,
            config_.stride
        );
    } else {
        throw std::runtime_error("Unsupported dataset type: " + config_.dataset_type);
    }
    
    // 创建高斯模型
    model_ = std::make_unique<GaussianModel>(20000000, kCUDA);  // 2000万高斯
    
    // 初始化GT位姿和关键帧列表
    gt_w2c_all_frames_.clear();
    keyframe_time_indices_.clear();
    
    // 初始化相机姿态存储
    int64_t num_frames = dataset_->size();
    int end_frame = (config_.end_frame < 0) ? static_cast<int>(num_frames) : config_.end_frame;
    int64_t actual_frames = static_cast<int64_t>(end_frame);
    
    // 初始化相机旋转为恒等四元数 [1, 0, 0, 0]
    cam_unnorm_rots_ = torch::zeros({1, 4, actual_frames}, torch::kFloat32).to(kCUDA);
    cam_unnorm_rots_[0][0].fill_(1.0f);  // w = 1, x = y = z = 0
    
    // 初始化相机平移为0
    cam_trans_ = torch::zeros({1, 3, actual_frames}, torch::kFloat32).to(kCUDA);
    
    // 获取图像尺寸
    auto [width, height] = dataset_->getImageSize();
    org_width_ = width;
    org_height_ = height;
}

void SLAMLoop::run()
{
    std::cout << "Starting SLAM loop..." << std::endl;
    std::cout << "Dataset: " << config_.dataset_type << " / " << config_.sequence << std::endl;
    std::cout << "Total frames: " << dataset_->size() << std::endl;
    
    // 如果提供了checkpoint路径，尝试加载
    if (!config_.load_checkpoint_path.empty()) {
        std::cout << "Attempting to load checkpoint from: " << config_.load_checkpoint_path << std::endl;
        if (loadCheckpoint(config_.load_checkpoint_path)) {
            std::cout << "Checkpoint loaded, resuming from checkpoint..." << std::endl;
            // 注意：如果从checkpoint加载，可能需要跳过初始化第一帧
            // 这里假设checkpoint已经包含了初始化后的模型
        } else {
            std::cout << "Failed to load checkpoint, initializing from first frame..." << std::endl;
            initializeFirstFrame();
        }
    } else {
        // 初始化第一帧
        initializeFirstFrame();
    }
    
    // 处理后续帧
    int64_t num_frames = dataset_->size();
    int end_frame = (config_.end_frame < 0) ? static_cast<int>(num_frames) : config_.end_frame;
    
    for (int frame_idx = 1; frame_idx < end_frame; ++frame_idx) {
        std::cout << "Processing frame " << frame_idx << " / " << end_frame << std::endl;
        
        processFrame(frame_idx);
        
        // 保存checkpoint
        if (config_.save_checkpoints && frame_idx % config_.checkpoint_interval == 0) {
            std::string checkpoint_path = config_.checkpoint_dir + "/params" + 
                                         std::to_string(frame_idx) + ".npz";
            saveCheckpoint(checkpoint_path, frame_idx);
        }
    }
    
    std::cout << "SLAM loop completed!" << std::endl;
}

void SLAMLoop::initializeFirstFrame()
{
    std::cout << "Initializing first frame..." << std::endl;
    
    // 加载第一帧数据
    auto [color, depth, intrinsics, pose] = (*dataset_)[0];
    
    // 保存内参
    intrinsics_ = intrinsics.clone();
    
    // 创建相机
    // 注意：inverse() 需要 CUBLAS，但可能存在初始化问题
    // 先尝试在 CPU 上计算，然后移到 CUDA
    Tensor w2c;
    if (pose.is_cuda()) {
        // 如果 pose 在 CUDA 上，先移到 CPU 计算逆矩阵，再移回 CUDA
        auto pose_cpu = pose.cpu();
        w2c = pose_cpu.inverse().to(torch::kCUDA);
    } else {
        // 如果 pose 在 CPU 上，直接计算
        w2c = pose.inverse().to(torch::kCUDA);
    }
    
    // 保存第一帧w2c
    first_frame_w2c_ = w2c.clone();
    
    auto camera = std::make_unique<Camera>(intrinsics, w2c, 
                                          dataset_->getImageSize().first,
                                          dataset_->getImageSize().second);
    
    // 从点云初始化高斯模型
    auto [points, colors, mean_sq_dist] = PointCloudUtils::extractPointCloud(
        color, depth, intrinsics, w2c, Tensor(), true
    );
    
    // 初始化高斯模型
    model_->initializeFromPointCloud(points, colors, mean_sq_dist, config_.gaussian_distribution);
    
    // 保存第一帧GT位姿（如果使用GT位姿）
    if (config_.use_gt_poses) {
        gt_w2c_all_frames_.push_back(pose.clone());  // 保存c2w，之后转换为w2c
    }
    
    // 添加第一帧为关键帧
    keyframe_time_indices_.push_back(0);
    
    // 更新第一帧的相机姿态（相对于第一帧，第一帧的位姿是恒等变换）
    updateCameraPose(0, w2c);
    
    std::cout << "First frame initialized with " << model_->getActiveCount() << " Gaussians" << std::endl;
}

void SLAMLoop::processFrame(int frame_idx)
{
    // 加载当前帧数据
    auto [color, depth, intrinsics, pose] = (*dataset_)[frame_idx];
    
    // 创建相机
    Tensor w2c;
    if (config_.use_gt_poses) {
        // 在 CPU 上计算逆矩阵，然后移到 CUDA（避免 CUBLAS 初始化问题）
        if (pose.is_cuda()) {
            auto pose_cpu = pose.cpu();
            w2c = pose_cpu.inverse().to(torch::kCUDA);
        } else {
            w2c = pose.inverse().to(torch::kCUDA);
        }
    } else {
        // 使用上一帧的位姿初始化
        w2c = torch::eye(4, torch::kFloat32).to(kCUDA);
    }
    
    auto camera = std::make_unique<Camera>(intrinsics, w2c,
                                          dataset_->getImageSize().first,
                                          dataset_->getImageSize().second);
    
    // Tracking
    if (!config_.use_gt_poses) {
        // 转换color和depth格式
        auto color_t = color.permute({2, 0, 1});  // [H, W, 3] -> [3, H, W]
        auto depth_t = depth;  // [H, W]
        
        float loss = tracker_->optimizePose(*model_, *camera, color_t, depth_t);
        std::cout << "  Tracking loss: " << loss << std::endl;
        
        // 更新w2c（tracking后可能改变了）
        w2c = camera->getW2C();
    }
    
    // Mapping（如果需要）
    if (frame_idx % config_.map_every == 0) {
        Keyframe keyframe;
        keyframe.frame_id = frame_idx;
        keyframe.camera = *camera;
        keyframe.color = color.permute({2, 0, 1});
        keyframe.depth = depth;
        
        mapper_->addKeyframe(keyframe);
        mapper_->optimize(*model_, keyframe);
        
        std::cout << "  Mapping completed" << std::endl;
    }
    
    // 添加关键帧（如果需要）
    if (frame_idx % config_.keyframe_every == 0) {
        Keyframe keyframe;
        keyframe.frame_id = frame_idx;
        keyframe.camera = *camera;
        keyframe.color = color.permute({2, 0, 1});
        keyframe.depth = depth;
        
        mapper_->addKeyframe(keyframe);
        
        // 记录关键帧索引
        keyframe_time_indices_.push_back(frame_idx);
    }
    
    // 保存GT位姿（如果使用GT位姿）
    if (config_.use_gt_poses) {
        gt_w2c_all_frames_.push_back(pose.clone());
    }
    
    // 更新相机姿态存储
    updateCameraPose(frame_idx, w2c);
}

bool SLAMLoop::loadCheckpoint(const std::string& path)
{
    try {
        CheckpointIO::load(path, *model_);
        std::cout << "Checkpoint loaded successfully from: " << path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load checkpoint: " << e.what() << std::endl;
        return false;
    }
}

void SLAMLoop::saveCheckpoint(const std::string& path, int time_idx)
{
    try {
        // 计算实际保存的帧数（到当前帧为止，包含当前帧）
        int64_t num_frames_to_save = static_cast<int64_t>(time_idx + 1);
        
        // 截取到当前帧的相机姿态
        auto cam_rots_slice = cam_unnorm_rots_.slice(2, 0, num_frames_to_save);
        auto cam_trans_slice = cam_trans_.slice(2, 0, num_frames_to_save);
        
        // 准备GT位姿（转换为w2c格式，只到当前帧）
        Tensor gt_w2c_all;
        if (!gt_w2c_all_frames_.empty() && static_cast<int>(gt_w2c_all_frames_.size()) > time_idx) {
            // 转换为w2c（inverse of c2w）并stack
            std::vector<Tensor> gt_w2c_list;
            for (int i = 0; i <= time_idx && i < static_cast<int>(gt_w2c_all_frames_.size()); ++i) {
                auto c2w_cpu = gt_w2c_all_frames_[i].cpu();
                auto w2c = c2w_cpu.inverse().to(torch::kCUDA);
                gt_w2c_list.push_back(w2c);
            }
            gt_w2c_all = torch::stack(gt_w2c_list, 0);  // [num_frames, 4, 4]
        }
        
        // 过滤关键帧索引（只保留<=time_idx的）
        std::vector<int> keyframe_indices_filtered;
        for (int idx : keyframe_time_indices_) {
            if (idx <= time_idx) {
                keyframe_indices_filtered.push_back(idx);
            }
        }
        
        CheckpointIO::save(
            path, 
            *model_,
            cam_rots_slice,
            cam_trans_slice,
            intrinsics_,
            first_frame_w2c_,
            org_width_,
            org_height_,
            gt_w2c_all,
            keyframe_indices_filtered
        );
        std::cout << "Checkpoint saved successfully to: " << path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to save checkpoint: " << e.what() << std::endl;
    }
}

void SLAMLoop::updateCameraPose(int frame_idx, const Tensor& w2c)
{
    // 计算相对于第一帧的变换
    // rel_w2c = w2c * first_frame_w2c^-1
    auto first_frame_c2w_cpu = first_frame_w2c_.cpu().inverse();
    auto rel_w2c_cpu = (w2c.cpu().matmul(first_frame_c2w_cpu));
    
    // 提取旋转矩阵和平移
    auto R = rel_w2c_cpu.slice(0, 0, 3).slice(1, 0, 3);  // [3, 3]
    auto t = rel_w2c_cpu.slice(0, 0, 3).select(1, 3);  // [3]
    
    // 将旋转矩阵转换为四元数（返回CPU tensor）
    auto quat = CheckpointIO::rotmat_to_quat(R);
    
    // 存储到对应的帧索引
    if (frame_idx >= 0 && frame_idx < static_cast<int>(cam_unnorm_rots_.size(2))) {
        // 将CPU tensor的值复制到CUDA tensor
        auto quat_cuda = quat.to(kCUDA);
        auto t_cuda = t.to(kCUDA);
        
        // 更新旋转四元数 [1, 4, num_frames]
        // cam_unnorm_rots_[0] 是 [4, num_frames]
        // 选择第frame_idx列并复制quat_cuda [4]
        cam_unnorm_rots_[0].select(1, frame_idx).copy_(quat_cuda);
        
        // 更新平移 [1, 3, num_frames]
        // cam_trans_[0] 是 [3, num_frames]
        // 选择第frame_idx列并复制t_cuda [3]
        cam_trans_[0].select(1, frame_idx).copy_(t_cuda);
    }
}

} // namespace isogs

