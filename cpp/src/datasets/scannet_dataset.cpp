#include "datasets/scannet_dataset.hpp"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <filesystem>

namespace isogs {

ScanNetDataset::ScanNetDataset(
    const std::string& basedir,
    const std::string& sequence,
    int start,
    int end,
    int stride,
    int desired_height,
    int desired_width)
    : basedir_(basedir)
    , sequence_(sequence)
    , start_(start)
    , end_(end)
    , stride_(stride)
    , desired_height_(desired_height)
    , desired_width_(desired_width)
{
    loadDataset();
}

void ScanNetDataset::loadDataset() {
    std::string seq_path = basedir_ + "/" + sequence_ + "/frames";
    
    // Scan for color and depth files
    std::string color_dir = seq_path + "/color";
    std::string depth_dir = seq_path + "/depth";
    
    // TODO: Implement file scanning
    // For now, assume files are numbered 0.jpg, 1.jpg, etc.
    
    // Load intrinsics
    std::string intrinsic_file = seq_path + "/intrinsic";
    intrinsics_ = loadIntrinsics(intrinsic_file);
}

int64_t ScanNetDataset::size() const {
    return color_files_.size();
}

std::tuple<Tensor, Tensor, Tensor, Tensor> ScanNetDataset::operator[](int64_t idx) {
    std::string seq_path = basedir_ + "/" + sequence_ + "/frames";
    
    // Load RGB
    std::string rgb_path = seq_path + "/color/" + color_files_[idx];
    auto color = loadImage(rgb_path, false);
    
    // Load Depth
    std::string depth_path = seq_path + "/depth/" + depth_files_[idx];
    auto depth = loadImage(depth_path, true);
    
    // Load pose
    std::string pose_path = seq_path + "/pose/" + std::to_string(idx) + ".txt";
    auto pose = loadPose(pose_path);
    
    return std::make_tuple(color, depth, intrinsics_, pose);
}

std::pair<int, int> ScanNetDataset::getImageSize() const {
    return {desired_height_, desired_width_};
}

Tensor ScanNetDataset::loadImage(const std::string& path, bool is_depth) {
    cv::Mat img = cv::imread(path, is_depth ? cv::IMREAD_UNCHANGED : cv::IMREAD_COLOR);
    if (img.empty()) {
        throw std::runtime_error("Cannot load image: " + path);
    }
    
    // Resize if needed
    if (img.rows != desired_height_ || img.cols != desired_width_) {
        cv::resize(img, img, cv::Size(desired_width_, desired_height_));
    }
    
    // Convert to tensor
    if (is_depth) {
        cv::Mat depth_float;
        img.convertTo(depth_float, CV_32F, 1.0 / 1000.0);  // ScanNet depth scale
        return torch::from_blob(depth_float.data, {desired_height_, desired_width_},
                               torch::kFloat32).clone().to(kCUDA);
    } else {
        cv::Mat rgb;
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
        return torch::from_blob(rgb.data, {desired_height_, desired_width_, 3},
                               torch::kUInt8).clone().permute({2, 0, 1}).to(kCUDA);
    }
}

Tensor ScanNetDataset::loadPose(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32).device(kCUDA));
    }
    
    Tensor pose = torch::zeros({4, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(kCUDA));
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float val;
            file >> val;
            pose[i][j] = val;
        }
    }
    
    return pose;
}

Tensor ScanNetDataset::loadIntrinsics(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        // Default intrinsics
        Tensor intrinsics = torch::eye(3, torch::TensorOptions().dtype(torch::kFloat32).device(kCUDA));
        intrinsics[0][0] = 577.87f;
        intrinsics[1][1] = 577.87f;
        intrinsics[0][2] = 319.5f;
        intrinsics[1][2] = 239.5f;
        return intrinsics;
    }
    
    Tensor intrinsics = torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(kCUDA));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float val;
            file >> val;
            intrinsics[i][j] = val;
        }
    }
    
    return intrinsics;
}

} // namespace isogs

