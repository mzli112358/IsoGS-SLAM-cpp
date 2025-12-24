#include "utils/nvs_eval.hpp"
#include "utils/eval.hpp"
#include "rendering/renderer.hpp"
#include "core/camera.hpp"
#include <torch/torch.h>
#include <iostream>

namespace isogs {

NVSEvalResult NVSEvaluator::evaluateNovelViewSynthesis(
    GaussianModel& model,
    DatasetBase& dataset,
    int num_frames,
    float sil_thres)
{
    NVSEvalResult result;
    result.total_frames = num_frames;
    
    std::vector<float> psnr_list, ssim_list, l1_list;
    
    Renderer renderer;
    
    for (int time_idx = 0; time_idx < num_frames; ++time_idx) {
        // 加载帧数据
        auto [color, depth, intrinsics, pose] = dataset[time_idx];
        
        // 处理数据格式
        auto color_t = color.permute({2, 0, 1}).to(torch::kFloat32) / 255.0f;  // [3, H, W]
        auto depth_t = depth.to(torch::kFloat32);  // [H, W]
        
        // 创建相机
        auto w2c = pose.inverse();
        Camera camera(intrinsics, w2c, dataset.getImageSize().first, dataset.getImageSize().second);
        
        // 渲染
        auto [rendered_rgb, rendered_depth] = renderer.renderRGBD(model, camera);
        
        // 计算指标
        float psnr = Evaluator::computePSNR(rendered_rgb, color_t);
        float ssim = Evaluator::computeSSIM(rendered_rgb, color_t);
        float l1 = Evaluator::computeL1(rendered_rgb, color_t);
        
        psnr_list.push_back(psnr);
        ssim_list.push_back(ssim);
        l1_list.push_back(l1);
        
        result.valid_frames++;
    }
    
    // 计算平均值
    if (!psnr_list.empty()) {
        float sum_psnr = 0.0f, sum_ssim = 0.0f, sum_l1 = 0.0f;
        for (size_t i = 0; i < psnr_list.size(); ++i) {
            sum_psnr += psnr_list[i];
            sum_ssim += ssim_list[i];
            sum_l1 += l1_list[i];
        }
        result.psnr = sum_psnr / psnr_list.size();
        result.ssim = sum_ssim / ssim_list.size();
        result.l1 = sum_l1 / l1_list.size();
    }
    
    return result;
}

} // namespace isogs

