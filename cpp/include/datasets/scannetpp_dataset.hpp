#pragma once

#include "datasets/dataset_base.hpp"
#include "core/types.hpp"
#include <torch/torch.h>
#include <string>
#include <vector>

namespace isogs {

/**
 * @brief ScanNet++数据集加载器
 */
class ScanNetPPDataset : public DatasetBase {
public:
    ScanNetPPDataset(
        const std::string& basedir,
        const std::string& sequence,
        int start = 0,
        int end = -1,
        int stride = 1,
        int desired_height = 480,
        int desired_width = 640
    );
    
    ~ScanNetPPDataset() override = default;
    
    int64_t size() const override;
    std::tuple<Tensor, Tensor, Tensor, Tensor> operator[](int64_t idx) override;
    std::string getName() const override { return "ScanNet++"; }
    std::pair<int, int> getImageSize() const override;
    
private:
    std::string basedir_;
    std::string sequence_;
    int start_;
    int end_;
    int stride_;
    int desired_height_;
    int desired_width_;
    
    std::vector<std::string> color_files_;
    std::vector<std::string> depth_files_;
    std::vector<Tensor> poses_;
    Tensor intrinsics_;
    
    void loadDataset();
    Tensor loadImage(const std::string& path, bool is_depth = false);
    Tensor loadPose(const std::string& path);
    Tensor loadIntrinsics(const std::string& path);
};

} // namespace isogs

