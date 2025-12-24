#pragma once

#include "datasets/dataset_base.hpp"
#include <memory>
#include <string>

namespace isogs {

/**
 * @brief 数据集工厂 - 根据类型字符串创建数据集实例
 */
class DatasetFactory {
public:
    /**
     * @brief 创建数据集实例
     * 
     * @param dataset_type 数据集类型 ("replica", "tum", "scannet", etc.)
     * @param basedir 数据集根目录
     * @param sequence 序列名称
     * @param start 起始帧
     * @param end 结束帧（-1表示到末尾）
     * @param stride 帧步长
     * @param desired_height 期望图像高度
     * @param desired_width 期望图像宽度
     * @return std::unique_ptr<DatasetBase> 数据集实例
     */
    static std::unique_ptr<DatasetBase> create(
        const std::string& dataset_type,
        const std::string& basedir,
        const std::string& sequence,
        int start = 0,
        int end = -1,
        int stride = 1,
        int desired_height = 480,
        int desired_width = 640
    );
};

} // namespace isogs

