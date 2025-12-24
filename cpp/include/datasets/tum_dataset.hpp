#pragma once

#include "datasets/dataset_base.hpp"
#include "core/types.hpp"
#include <torch/torch.h>
#include <string>
#include <vector>

namespace isogs {

/**
 * @brief TUM RGB-D数据集加载器
 */
class TUMDataset : public DatasetBase {
public:
    TUMDataset(
        const std::string& basedir,
        const std::string& sequence,
        int start = 0,
        int end = -1,
        int stride = 1,
        int desired_height = 480,
        int desired_width = 640
    );
    
    ~TUMDataset() override = default;
    
    int64_t size() const override;
    std::tuple<Tensor, Tensor, Tensor, Tensor> operator[](int64_t idx) override;
    std::string getName() const override { return "TUM"; }
    std::pair<int, int> getImageSize() const override;
    
private:
    std::string basedir_;
    std::string sequence_;
    int start_;
    int end_;
    int stride_;
    int desired_height_;
    int desired_width_;
    
    std::vector<std::string> rgb_files_;
    std::vector<std::string> depth_files_;
    std::vector<Tensor> poses_;  // 缓存位姿
    Tensor intrinsics_;
    
    void loadDataset();
    void loadAssociations(const std::string& assoc_file);
    Tensor loadImage(const std::string& path, bool is_depth = false);
    Tensor loadPose(const std::string& path);
};

} // namespace isogs

