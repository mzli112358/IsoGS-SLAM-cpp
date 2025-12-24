#include "rendering/renderer.hpp"
#include <torch/torch.h>
#include <c10/util/Exception.h>

// gsplat headers
#include "Ops.h"
#include "Common.h"
#include "Cameras.h"
#include "Rasterization.h"

namespace isogs {

Tensor Renderer::renderRGB(
    const GaussianModel& model,
    const Camera& camera)
{
    // 获取活跃高斯数量
    int64_t N = model.getActiveCount();
    if (N == 0) {
        // 返回黑色图像
        int height = camera.getHeight();
        int width = camera.getWidth();
        return torch::zeros({3, height, width}, 
            torch::TensorOptions().dtype(torch::kFloat32).device(model.getDevice()));
    }
    
    // 获取高斯参数（只取活跃部分）
    auto means = model.getMeans3D();  // [N, 3]
    auto scales = model.getScales();  // [N, 3] (log空间)
    auto rotations = model.getRotations();  // [N, 4]
    auto opacities = model.getOpacity();  // [N, 1] (logit空间)
    auto sh_coeffs = model.getSHCoeffs();  // [N, 48]
    
    // 转换到实际空间（添加clamp避免exp溢出）
    auto scales_clamped = torch::clamp(scales, -10.0f, 10.0f);
    auto scales_exp = torch::exp(scales_clamped);  // [N, 3]
    auto opacities_sigmoid = torch::sigmoid(opacities).squeeze(1);  // [N]
    
    // 准备相机参数
    auto w2c = camera.getW2C();  // [4, 4]
    auto K = camera.getIntrinsics();  // [3, 3]
    int width = camera.getWidth();
    int height = camera.getHeight();
    
    // 确保所有tensor都在CUDA上，并保持梯度
    auto device = means.device();
    w2c = w2c.to(device);
    K = K.to(device);
    
    // 如果w2c支持梯度，确保所有相关tensor都在计算图中
    bool need_grad = w2c.requires_grad();
    if (need_grad) {
        // 确保means等参数也在计算图中（如果需要）
        // 注意：对于tracking，我们只优化w2c，所以means等不需要梯度
        // 但w2c需要梯度，所以渲染结果应该自动有梯度
    }
    
    // 扩展维度以匹配gsplat的接口
    // projection_ewa_3dgs_fused_fwd期望:
    // means: [..., N, 3]
    // viewmats: [..., C, 4, 4]
    // 返回: [..., C, N, ...]
    // 我们传入 [1, N, 3] 和 [1, 1, 4, 4]，应该返回 [1, 1, N, ...]
    auto means_expanded = means.unsqueeze(0);  // [1, N, 3]
    auto quats_expanded = rotations.unsqueeze(0);  // [1, N, 4]
    auto scales_expanded = scales_exp.unsqueeze(0);  // [1, N, 3]
    auto opacities_expanded = opacities_sigmoid.unsqueeze(0);  // [1, N]
    
    // viewmat需要是 [..., C, 4, 4]，我们传入 [1, 1, 4, 4]
    auto viewmat = w2c.unsqueeze(0).unsqueeze(0);  // [1, 1, 4, 4]
    auto K_expanded = K.unsqueeze(0).unsqueeze(0);  // [1, 1, 3, 3]
    
    // 投影：将3D高斯投影到2D
    float eps2d = 0.3f;
    float near_plane = 0.01f;
    float far_plane = 100.0f;
    float radius_clip = 0.0f;
    bool calc_compensations = false;
    gsplat::CameraModelType camera_model = gsplat::CameraModelType::PINHOLE;
    
    // 注意：projection_ewa_3dgs_fused_fwd 返回的顺序是：
    // (radii, means2d, depths, conics, compensations)
    auto [radii, means2d, depths, conics, compensations] = 
        gsplat::projection_ewa_3dgs_fused_fwd(
            means_expanded,
            c10::nullopt,  // covars
            quats_expanded,
            scales_expanded,
            opacities_expanded,
            viewmat,
            K_expanded,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
            camera_model
        );
    
    // projection_ewa_3dgs_fused_fwd 返回的形状是 [batch_dims, C, N, ...]
    // 我们传入 means_expanded: [1, N, 3], viewmat: [1, 1, 4, 4]
    // 所以返回应该是:
    // - means2d: [1, 1, N, 2]
    // - radii: [1, 1, N, 2]  
    // - depths: [1, 1, N]
    // - conics: [1, 1, N, 3]
    
    // 检查实际返回的形状
    // 如果means2d是[1, 1, N]（缺少最后一个维度），说明返回的形状不对
    // 这不应该发生，但如果发生了，我们需要检查是否是depths被错误地赋值给了means2d
    if (means2d.dim() == 3 && means2d.size(-1) == N && means2d.size(-2) == 1) {
        // 可能是depths被错误地赋值给了means2d，或者返回顺序错误
        // 检查depths的形状
        if (depths.dim() == 3 && depths.size(-1) == N && depths.size(-2) == 1) {
            // 可能是返回值的顺序错误，尝试交换
            std::swap(means2d, depths);
            // 但这样也不对，因为depths应该是[1, 1, N]，而means2d应该是[1, 1, N, 2]
            TORCH_CHECK(false, "means2d has wrong shape: ", means2d.sizes(), ", expected [..., N, 2]. depths shape: ", depths.sizes());
        }
    }
    
    // 计算view directions用于SH计算
    auto c2w = camera.getC2W();
    auto campos = c2w.slice(0, 0, 3).slice(1, 3, 4).squeeze(1);  // [3]
    auto dirs = means - campos.unsqueeze(0);  // [N, 3]
    dirs = dirs / (torch::norm(dirs, 2, 1, true) + 1e-6f);  // 归一化
    
    // SH系数转换（使用degree 3，即16个系数）
    // sh_coeffs: [N, 48] -> [N, 16, 3]
    int sh_degree = 3;
    int sh_coeffs_per_channel = (sh_degree + 1) * (sh_degree + 1);  // 16
    auto sh_coeffs_reshaped = sh_coeffs.view({N, sh_coeffs_per_channel, 3});  // [N, 16, 3]
    
    // 转换SH系数到RGB颜色
    auto colors = gsplat::spherical_harmonics_fwd(
        sh_degree,
        dirs,  // [N, 3]
        sh_coeffs_reshaped,  // [N, 16, 3]
        c10::nullopt  // masks
    );  // [N, 3]
    
    // Clamp colors (与Python版本一致)
    colors = torch::clamp_min(colors + 0.5f, 0.0f);
    
    // Tile intersection
    uint32_t tile_size = 16;
    uint32_t tile_width = (width + tile_size - 1) / tile_size;
    uint32_t tile_height = (height + tile_size - 1) / tile_size;
    
    // 注意：intersect_tile期望means2d为[..., C, N, 2]或[nnz, 2]
    // projection_ewa_3dgs_fused_fwd返回的means2d应该是[batch_dims, C, N, 2]
    // 我们传入means_expanded: [1, N, 3], viewmat: [1, 1, 4, 4]
    // 所以返回应该是[1, 1, N, 2]
    
    // 检查并修复维度
    // 如果最后一个维度不是2，可能需要转置
    if (means2d.size(-1) != 2) {
        // 可能是维度顺序错误，尝试找到正确的维度
        // 如果最后一个维度是N，说明可能是[..., 2, N]，需要转置最后两个维度
        if (means2d.size(-1) == N && means2d.size(-2) == 2) {
            // [..., 2, N] -> [..., N, 2]
            means2d = means2d.transpose(-1, -2);
            if (radii.size(-1) == N && radii.size(-2) == 2) {
                radii = radii.transpose(-1, -2);
            }
            if (conics.size(-1) == N && conics.size(-2) == 3) {
                conics = conics.transpose(-1, -2);
            }
        }
    }
    
    // 如果means2d是[1, N, 2]，需要添加C维度变成[1, 1, N, 2]
    if (means2d.dim() == 3 && means2d.size(-1) == 2) {
        means2d = means2d.unsqueeze(1);  // [1, N, 2] -> [1, 1, N, 2]
        if (radii.dim() == 3) {
            radii = radii.unsqueeze(1);
        }
        if (depths.dim() == 2) {
            depths = depths.unsqueeze(1);  // [1, N] -> [1, 1, N]
        }
        if (conics.dim() == 3) {
            conics = conics.unsqueeze(1);
        }
    }
    
    // 验证维度
    TORCH_CHECK(means2d.dim() >= 3, "means2d must have at least 3 dimensions, got dim=", means2d.dim());
    TORCH_CHECK(means2d.size(-1) == 2, "means2d last dim must be 2, got ", means2d.size(-1), ", shape=", means2d.sizes());
    
    auto [tiles_per_gauss, isect_ids, flatten_ids] = gsplat::intersect_tile(
        means2d,      // [1, 1, N, 2]
        radii,        // [1, 1, N, 2]
        depths,       // [1, 1, N]
        c10::nullopt, // image_ids - 让函数内部生成
        c10::nullopt, // gaussian_ids - 让函数内部生成
        1,            // I (number of images)
        tile_size,
        tile_width,
        tile_height,
        false,        // sort - 先禁用排序，看看是否能解决问题
        false         // segmented
    );
    
    // 计算isect_offsets
    auto isect_offsets = gsplat::intersect_offset(
        isect_ids,
        1,  // I
        tile_width,
        tile_height
    );  // [tile_height, tile_width]
    
    // 准备用于光栅化的tensor，需要squeeze掉batch和camera维度
    auto means2d_flat = means2d.squeeze(0).squeeze(0);  // [1, 1, N, 2] -> [N, 2]
    auto conics_flat = conics.squeeze(0).squeeze(0);    // [1, 1, N, 3] -> [N, 3]
    auto opacities_flat = opacities_sigmoid;            // [N]
    
    // 光栅化
    auto [renders, alphas, last_ids] = gsplat::rasterize_to_pixels_3dgs_fwd(
        means2d_flat,      // [N, 2]
        conics_flat,       // [N, 3]
        colors,            // [N, 3]
        opacities_flat,    // [N]
        c10::nullopt,      // backgrounds
        c10::nullopt,      // masks
        width,
        height,
        tile_size,
        isect_offsets,     // [tile_height, tile_width]
        flatten_ids
    );  // renders: [1, height, width, 3], alphas: [1, height, width, 1]
    
    // 转换格式：[1, H, W, 3] -> [3, H, W]
    renders = renders.squeeze(0);  // [H, W, 3]
    renders = renders.permute({2, 0, 1});  // [3, H, W]
    
    // 注意：renders 应该已经继承了计算图（如果输入支持梯度）
    // 不需要手动设置 requires_grad_，因为这是非叶子变量
    
    return renders;
}

Tensor Renderer::renderDepth(
    const GaussianModel& model,
    const Camera& camera)
{
    // 获取活跃高斯数量
    int64_t N = model.getActiveCount();
    if (N == 0) {
        // 返回零深度图
        int height = camera.getHeight();
        int width = camera.getWidth();
        return torch::zeros({height, width}, 
            torch::TensorOptions().dtype(torch::kFloat32).device(model.getDevice()));
    }
    
    // 获取高斯参数
    auto means = model.getMeans3D();  // [N, 3]
    auto scales = model.getScales();  // [N, 3] (log空间)
    auto rotations = model.getRotations();  // [N, 4]
    auto opacities = model.getOpacity();  // [N, 1] (logit空间)
    
    // 转换到实际空间（添加clamp避免exp溢出）
    auto scales_clamped = torch::clamp(scales, -10.0f, 10.0f);
    auto scales_exp = torch::exp(scales_clamped);  // [N, 3]
    auto opacities_sigmoid = torch::sigmoid(opacities).squeeze(1);  // [N]
    
    // 准备相机参数
    auto w2c = camera.getW2C();  // [4, 4]
    auto K = camera.getIntrinsics();  // [3, 3]
    int width = camera.getWidth();
    int height = camera.getHeight();
    
    // 确保所有tensor都在CUDA上
    auto device = means.device();
    w2c = w2c.to(device);
    K = K.to(device);
    
    // 扩展维度
    auto means_expanded = means.unsqueeze(0);  // [1, N, 3]
    auto quats_expanded = rotations.unsqueeze(0);  // [1, N, 4]
    auto scales_expanded = scales_exp.unsqueeze(0);  // [1, N, 3]
    auto opacities_expanded = opacities_sigmoid.unsqueeze(0);  // [1, N]
    
    auto viewmat = w2c.unsqueeze(0).unsqueeze(0);  // [1, 1, 4, 4]
    auto K_expanded = K.unsqueeze(0).unsqueeze(0);  // [1, 1, 3, 3]
    
    // 投影：获取depths
    float eps2d = 0.3f;
    float near_plane = 0.01f;
    float far_plane = 100.0f;
    float radius_clip = 0.0f;
    bool calc_compensations = false;
    gsplat::CameraModelType camera_model = gsplat::CameraModelType::PINHOLE;
    
    // 注意：projection_ewa_3dgs_fused_fwd 返回的顺序是：
    // (radii, means2d, depths, conics, compensations)
    auto [radii, means2d, depths, conics, compensations] = 
        gsplat::projection_ewa_3dgs_fused_fwd(
            means_expanded,
            c10::nullopt,  // covars
            quats_expanded,
            scales_expanded,
            opacities_expanded,
            viewmat,
            K_expanded,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
            camera_model
        );
    
    // 使用depths作为颜色通道（深度渲染）
    // depths: [1, 1, N] -> [N]
    auto depths_1d = depths.squeeze(0).squeeze(0);  // [N]
    auto depth_colors = depths_1d.unsqueeze(1);  // [N, 1] - 作为单通道颜色
    
    // Tile intersection
    uint32_t tile_size = 16;
    uint32_t tile_width = (width + tile_size - 1) / tile_size;
    uint32_t tile_height = (height + tile_size - 1) / tile_size;
    
    auto [tiles_per_gauss, isect_ids, flatten_ids] = gsplat::intersect_tile(
        means2d,      // [1, 1, N, 2]
        radii,        // [1, 1, N, 2]
        depths,       // [1, 1, N]
        c10::nullopt, // image_ids
        c10::nullopt, // gaussian_ids
        1,            // I
        tile_size,
        tile_width,
        tile_height,
        true,         // sort
        false         // segmented
    );
    
    // 计算isect_offsets
    auto isect_offsets = gsplat::intersect_offset(
        isect_ids,
        1,  // I
        tile_width,
        tile_height
    );  // [tile_height, tile_width]
    
    // 准备用于光栅化的tensor
    auto means2d_flat = means2d.squeeze(0).squeeze(0);  // [1, 1, N, 2] -> [N, 2]
    auto conics_flat = conics.squeeze(0).squeeze(0);    // [1, 1, N, 3] -> [N, 3]
    
    // 光栅化深度（使用depths作为颜色）
    auto [renders, alphas, last_ids] = gsplat::rasterize_to_pixels_3dgs_fwd(
        means2d_flat,      // [N, 2]
        conics_flat,       // [N, 3]
        depth_colors,      // [N, 1] - 深度作为颜色
        opacities_sigmoid, // [N]
        c10::nullopt,      // backgrounds
        c10::nullopt,      // masks
        width,
        height,
        tile_size,
        isect_offsets,     // [tile_height, tile_width]
        flatten_ids
    );  // renders: [1, height, width, 1]
    
    // 转换格式：[1, H, W, 1] -> [H, W]
    auto depth_map = renders.squeeze(0).squeeze(-1);  // [H, W]
    
    // 注意：depth_map 应该已经继承了计算图（如果输入支持梯度）
    // 不需要手动设置 requires_grad_，因为这是非叶子变量
    
    return depth_map;
}

std::tuple<Tensor, Tensor> Renderer::renderRGBD(
    const GaussianModel& model,
    const Camera& camera)
{
    auto rgb = renderRGB(model, camera);
    auto depth = renderDepth(model, camera);
    return std::make_tuple(rgb, depth);
}

} // namespace isogs
