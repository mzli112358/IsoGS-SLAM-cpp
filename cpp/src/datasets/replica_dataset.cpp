#include "datasets/replica_dataset.hpp"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

namespace isogs {

ReplicaDataset::ReplicaDataset(
    const std::string& basedir,
    const std::string& sequence,
    int start,
    int end,
    int stride,
    int desired_height,
    int desired_width)
    : input_folder_(basedir + "/" + sequence)
    , pose_path_(input_folder_ + "/traj.txt")
    , start_(start)
    , end_(end)
    , stride_(stride)
    , desired_height_(desired_height)
    , desired_width_(desired_width)
{
    loadFilePaths();
    loadPoses();
    
    // 计算实际帧数
    int total_frames = static_cast<int>(color_paths_.size());
    if (end_ < 0) {
        end_ = total_frames;
    }
    num_frames_ = (end_ - start_ + stride_ - 1) / stride_;
}

void ReplicaDataset::loadFilePaths()
{
    // 加载color图像路径
    std::string color_pattern = input_folder_ + "/results/frame*.jpg";
    std::vector<std::string> all_color_paths;
    
    for (const auto& entry : fs::directory_iterator(input_folder_ + "/results")) {
        std::string path = entry.path().string();
        if (path.find("frame") != std::string::npos && 
            path.find(".jpg") != std::string::npos) {
            all_color_paths.push_back(path);
        }
    }
    
    // 排序
    std::sort(all_color_paths.begin(), all_color_paths.end());
    
    // 加载depth图像路径
    std::vector<std::string> all_depth_paths;
    for (const auto& entry : fs::directory_iterator(input_folder_ + "/results")) {
        std::string path = entry.path().string();
        if (path.find("depth") != std::string::npos && 
            path.find(".png") != std::string::npos) {
            all_depth_paths.push_back(path);
        }
    }
    std::sort(all_depth_paths.begin(), all_depth_paths.end());
    
    // 根据start, end, stride筛选
    for (int i = start_; i < end_ && i < static_cast<int>(all_color_paths.size()); i += stride_) {
        color_paths_.push_back(all_color_paths[i]);
        if (i < static_cast<int>(all_depth_paths.size())) {
            depth_paths_.push_back(all_depth_paths[i]);
        }
    }
}

void ReplicaDataset::loadPoses()
{
    std::ifstream file(pose_path_);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open pose file: " + pose_path_);
    }
    
    std::string line;
    int line_idx = 0;
    while (std::getline(file, line) && line_idx < static_cast<int>(color_paths_.size())) {
        if (line_idx >= start_ && (end_ < 0 || line_idx < end_)) {
            if ((line_idx - start_) % stride_ == 0) {
                std::istringstream iss(line);
                std::vector<float> values;
                float val;
                while (iss >> val) {
                    values.push_back(val);
                }
                
                if (values.size() == 16) {
                    // 转换为4x4矩阵 (c2w格式) and move to CUDA
                    auto pose = torch::zeros({4, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
                    for (int i = 0; i < 4; ++i) {
                        for (int j = 0; j < 4; ++j) {
                            pose[i][j] = values[i * 4 + j];
                        }
                    }
                    poses_.push_back(pose);
                }
            }
        }
        line_idx++;
    }
}

Tensor ReplicaDataset::loadImage(const std::string& path) const
{
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Cannot load image: " + path);
    }
    
    // BGR to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    
    // Resize if needed
    if (img.rows != desired_height_ || img.cols != desired_width_) {
        cv::resize(img, img, cv::Size(desired_width_, desired_height_));
    }
    
    // Convert to tensor [H, W, 3] and move to CUDA
    auto tensor = torch::from_blob(
        img.data,
        {img.rows, img.cols, 3},
        torch::kUInt8
    ).clone().to(torch::kFloat32) / 255.0f;
    
    // Move to CUDA
    tensor = tensor.to(torch::kCUDA);
    
    return tensor;
}

Tensor ReplicaDataset::loadDepth(const std::string& path) const
{
    cv::Mat depth = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (depth.empty()) {
        throw std::runtime_error("Cannot load depth: " + path);
    }
    
    // Resize if needed
    if (depth.rows != desired_height_ || depth.cols != desired_width_) {
        cv::resize(depth, depth, cv::Size(desired_width_, desired_height_));
    }
    
    // Convert to tensor [H, W]
    auto tensor = torch::from_blob(
        depth.data,
        {depth.rows, depth.cols},
        torch::kInt16  // 注意：LibTorch可能没有kUInt16，使用kInt16或直接读取为uint16_t
    ).clone().to(torch::kFloat32);
    
    // 转换为米单位 (Replica数据集depth是毫米)
    tensor = tensor / 1000.0f;
    
    // Move to CUDA
    tensor = tensor.to(torch::kCUDA);
    
    return tensor;
}

Tensor ReplicaDataset::getIntrinsics() const
{
    // Replica数据集的标准内参
    // 根据图像尺寸计算内参（假设FOV约为60度）
    float fov = 60.0f * M_PI / 180.0f;  // 60度FOV
    float fx = static_cast<float>(desired_width_) / (2.0f * tanf(fov / 2.0f));
    float fy = static_cast<float>(desired_height_) / (2.0f * tanf(fov / 2.0f));
    float cx = static_cast<float>(desired_width_) / 2.0f;
    float cy = static_cast<float>(desired_height_) / 2.0f;
    
    auto intrinsics = torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    intrinsics[0][0] = fx;
    intrinsics[1][1] = fy;
    intrinsics[0][2] = cx;
    intrinsics[1][2] = cy;
    intrinsics[2][2] = 1.0f;
    
    return intrinsics;
}

std::tuple<Tensor, Tensor, Tensor, Tensor> ReplicaDataset::operator[](int64_t idx)
{
    if (idx < 0 || idx >= num_frames_) {
        throw std::out_of_range("Frame index out of range");
    }
    
    // 加载color和depth
    Tensor color = loadImage(color_paths_[idx]);
    Tensor depth = loadDepth(depth_paths_[idx]);
    
    // 获取内参
    Tensor intrinsics = getIntrinsics();
    
    // 获取位姿 (c2w格式) - 确保在CUDA上
    Tensor pose = poses_[idx].clone();
    if (!pose.is_cuda()) {
        pose = pose.to(torch::kCUDA);
    }
    
    return std::make_tuple(color, depth, intrinsics, pose);
}

} // namespace isogs

