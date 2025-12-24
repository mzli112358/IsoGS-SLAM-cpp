#include "datasets/tum_dataset.hpp"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <filesystem>

namespace isogs {

TUMDataset::TUMDataset(
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

void TUMDataset::loadDataset() {
    std::string seq_path = basedir_ + "/" + sequence_;
    std::string assoc_file = seq_path + "/associations.txt";
    
    // Load associations
    loadAssociations(assoc_file);
    
    // Load intrinsics (from first frame or separate file)
    // TUM format: fx fy cx cy from depth camera
    intrinsics_ = torch::eye(3, torch::TensorOptions().dtype(torch::kFloat32).device(kCUDA));
    // TODO: Load actual intrinsics from file or first frame
    intrinsics_[0][0] = 525.0f;  // fx
    intrinsics_[1][1] = 525.0f;  // fy
    intrinsics_[0][2] = 319.5f;  // cx
    intrinsics_[1][2] = 239.5f;  // cy
}

void TUMDataset::loadAssociations(const std::string& assoc_file) {
    std::ifstream file(assoc_file);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open association file: " + assoc_file);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::string timestamp_rgb, rgb_file, timestamp_depth, depth_file;
        iss >> timestamp_rgb >> rgb_file >> timestamp_depth >> depth_file;
        
        rgb_files_.push_back(rgb_file);
        depth_files_.push_back(depth_file);
    }
    
    // Apply start/end/stride
    if (end_ < 0) end_ = rgb_files_.size();
    // TODO: Apply filtering
}

int64_t TUMDataset::size() const {
    return rgb_files_.size();
}

std::tuple<Tensor, Tensor, Tensor, Tensor> TUMDataset::operator[](int64_t idx) {
    std::string seq_path = basedir_ + "/" + sequence_;
    
    // Load RGB
    std::string rgb_path = seq_path + "/" + rgb_files_[idx];
    auto color = loadImage(rgb_path, false);
    
    // Load Depth
    std::string depth_path = seq_path + "/" + depth_files_[idx];
    auto depth = loadImage(depth_path, true);
    
    // Load pose (from groundtruth.txt)
    // TODO: Implement pose loading
    
    // Default pose (identity)
    Tensor pose = torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32).device(kCUDA));
    
    return std::make_tuple(color, depth, intrinsics_, pose);
}

std::pair<int, int> TUMDataset::getImageSize() const {
    return {desired_height_, desired_width_};
}

Tensor TUMDataset::loadImage(const std::string& path, bool is_depth) {
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
        // Depth image: single channel, convert to float and scale
        cv::Mat depth_float;
        img.convertTo(depth_float, CV_32F, 1.0 / 5000.0);  // TUM depth scale
        return torch::from_blob(depth_float.data, {desired_height_, desired_width_},
                               torch::kFloat32).clone().to(kCUDA);
    } else {
        // RGB image: BGR to RGB
        cv::Mat rgb;
        cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
        return torch::from_blob(rgb.data, {desired_height_, desired_width_, 3},
                               torch::kUInt8).clone().permute({2, 0, 1}).to(kCUDA);
    }
}

Tensor TUMDataset::loadPose(const std::string& path) {
    // TODO: Implement pose loading from TUM format
    return torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32).device(kCUDA));
}

} // namespace isogs

