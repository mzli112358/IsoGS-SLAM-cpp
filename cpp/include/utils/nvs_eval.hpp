#pragma once

#include "core/gaussian_model.hpp"
#include "core/camera.hpp"
#include "datasets/dataset_base.hpp"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <map>

namespace isogs {

/**
 * @brief 新视角合成评估结果
 */
struct NVSEvalResult {
    float psnr = 0.0f;
    float ssim = 0.0f;
    float l1 = 0.0f;
    float rmse = 0.0f;
    int valid_frames = 0;
    int total_frames = 0;
};

/**
 * @brief 新视角合成评估工具
 */
class NVSEvaluator {
public:
    /**
     * @brief 在新视角上评估渲染质量
     * 
     * @param model 高斯模型
     * @param dataset 数据集
     * @param num_frames 评估帧数
     * @param sil_thres silhouette阈值
     * @return NVSEvalResult 评估结果
     */
    static NVSEvalResult evaluateNovelViewSynthesis(
        GaussianModel& model,
        DatasetBase& dataset,
        int num_frames,
        float sil_thres = 0.99f
    );
};

} // namespace isogs

