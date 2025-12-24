#pragma once

#include "core/types.hpp"
#include "core/camera.hpp"
#include <torch/torch.h>
#include <memory>
#include <string>

namespace isogs {

/**
 * @brief 数据集基类
 */
class DatasetBase {
public:
    virtual ~DatasetBase() = default;
    
    // 获取数据集大小
    virtual int64_t size() const = 0;
    
    // 获取第idx帧数据
    // 返回: {color, depth, intrinsics, pose}
    virtual std::tuple<Tensor, Tensor, Tensor, Tensor> operator[](int64_t idx) = 0;
    
    // 获取数据集名称
    virtual std::string getName() const = 0;
    
    // 获取图像尺寸
    virtual std::pair<int, int> getImageSize() const = 0;
};

} // namespace isogs

