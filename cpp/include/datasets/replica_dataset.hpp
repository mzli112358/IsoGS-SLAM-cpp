#pragma once

#include "datasets/dataset_base.hpp"
#include <string>
#include <vector>
#include <filesystem>

namespace isogs {

/**
 * @brief Replica数据集加载器
 */
class ReplicaDataset : public DatasetBase {
public:
    ReplicaDataset(
        const std::string& basedir,
        const std::string& sequence,
        int start = 0,
        int end = -1,
        int stride = 1,
        int desired_height = 480,
        int desired_width = 640
    );
    
    ~ReplicaDataset() override = default;
    
    int64_t size() const override { return num_frames_; }
    
    std::tuple<Tensor, Tensor, Tensor, Tensor> operator[](int64_t idx) override;
    
    std::string getName() const override { return "Replica"; }
    
    std::pair<int, int> getImageSize() const override {
        return {desired_width_, desired_height_};
    }
    
private:
    std::string input_folder_;
    std::string pose_path_;
    int start_;
    int end_;
    int stride_;
    int desired_height_;
    int desired_width_;
    int64_t num_frames_;
    
    std::vector<std::string> color_paths_;
    std::vector<std::string> depth_paths_;
    std::vector<Tensor> poses_;  // c2w格式
    
    void loadFilePaths();
    void loadPoses();
    Tensor loadImage(const std::string& path) const;
    Tensor loadDepth(const std::string& path) const;
    Tensor getIntrinsics() const;
};

} // namespace isogs

