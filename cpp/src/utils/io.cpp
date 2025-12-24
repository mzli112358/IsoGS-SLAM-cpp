#include "utils/io.hpp"
#include <torch/torch.h>
#include <cnpy.h>
#include <map>
#include <vector>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace isogs {

Tensor CheckpointIO::rotmat_to_quat(const Tensor& R)
{
    // R: [3, 3] 旋转矩阵 (CPU tensor)
    // 返回: [4] 四元数 (w, x, y, z) (CPU tensor)
    
    // 确保R在CPU上
    auto R_cpu = R.cpu();
    
    // 计算迹
    float trace = R_cpu[0][0].item<float>() + R_cpu[1][1].item<float>() + R_cpu[2][2].item<float>();
    
    Tensor quat = torch::zeros({4}, torch::kFloat32);
    
    if (trace > 0.0f) {
        float s = sqrtf(trace + 1.0f) * 2.0f;  // s = 4 * qw
        quat[0] = 0.25f * s;  // w
        quat[1] = (R_cpu[2][1] - R_cpu[1][2]).item<float>() / s;  // x
        quat[2] = (R_cpu[0][2] - R_cpu[2][0]).item<float>() / s;  // y
        quat[3] = (R_cpu[1][0] - R_cpu[0][1]).item<float>() / s;  // z
    } else if ((R_cpu[0][0].item<float>() > R_cpu[1][1].item<float>()) && 
               (R_cpu[0][0].item<float>() > R_cpu[2][2].item<float>())) {
        float s = sqrtf(1.0f + R_cpu[0][0].item<float>() - R_cpu[1][1].item<float>() - R_cpu[2][2].item<float>()) * 2.0f;
        quat[0] = (R_cpu[2][1] - R_cpu[1][2]).item<float>() / s;  // w
        quat[1] = 0.25f * s;  // x
        quat[2] = (R_cpu[0][1] + R_cpu[1][0]).item<float>() / s;  // y
        quat[3] = (R_cpu[0][2] + R_cpu[2][0]).item<float>() / s;  // z
    } else if (R_cpu[1][1].item<float>() > R_cpu[2][2].item<float>()) {
        float s = sqrtf(1.0f + R_cpu[1][1].item<float>() - R_cpu[0][0].item<float>() - R_cpu[2][2].item<float>()) * 2.0f;
        quat[0] = (R_cpu[0][2] - R_cpu[2][0]).item<float>() / s;  // w
        quat[1] = (R_cpu[0][1] + R_cpu[1][0]).item<float>() / s;  // x
        quat[2] = 0.25f * s;  // y
        quat[3] = (R_cpu[1][2] + R_cpu[2][1]).item<float>() / s;  // z
    } else {
        float s = sqrtf(1.0f + R_cpu[2][2].item<float>() - R_cpu[0][0].item<float>() - R_cpu[1][1].item<float>()) * 2.0f;
        quat[0] = (R_cpu[1][0] - R_cpu[0][1]).item<float>() / s;  // w
        quat[1] = (R_cpu[0][2] + R_cpu[2][0]).item<float>() / s;  // x
        quat[2] = (R_cpu[1][2] + R_cpu[2][1]).item<float>() / s;  // y
        quat[3] = 0.25f * s;  // z
    }
    
    return quat;
}

void CheckpointIO::save(
    const std::string& path,
    const GaussianModel& model,
    const Tensor& cam_unnorm_rots,
    const Tensor& cam_trans,
    const Tensor& intrinsics,
    const Tensor& first_frame_w2c,
    int org_width,
    int org_height,
    const Tensor& gt_w2c_all_frames,
    const std::vector<int>& keyframe_time_indices)
{
    // 创建目录（如果不存在）
    fs::path file_path(path);
    fs::create_directories(file_path.parent_path());
    
    // 获取活跃高斯数量
    int64_t N = model.getActiveCount();
    if (N == 0) {
        throw std::runtime_error("Cannot save empty GaussianModel");
    }
    
    // 获取参数（只取活跃部分）
    auto means3D = model.getMeans3D();  // [N, 3]
    auto scales = model.getScales();  // [N, 3] (log空间)
    auto rotations = model.getRotations();  // [N, 4]
    auto sh_coeffs = model.getSHCoeffs();  // [N, 48]
    auto opacity = model.getOpacity();  // [N, 1] (logit空间)
    auto timestep = model.getTimestep();  // [N] 每个高斯点创建的帧索引
    
    // 转换到CPU并转为连续内存
    auto means3D_cpu = means3D.detach().cpu().contiguous();
    auto scales_cpu = scales.detach().cpu().contiguous();
    auto rotations_cpu = rotations.detach().cpu().contiguous();
    auto sh_coeffs_cpu = sh_coeffs.detach().cpu().contiguous();
    auto opacity_cpu = opacity.detach().cpu().contiguous();
    auto timestep_cpu = timestep.detach().cpu().contiguous();
    
    // 保存为npz文件（逐个保存数组）
    // means3D: [N, 3]
    std::vector<size_t> shape_means = {static_cast<size_t>(N), 3};
    cnpy::npz_save(path, "means3D", means3D_cpu.data_ptr<float>(), shape_means, "w");
    
    // log_scales: [N, 3] (保存log空间的值)
    std::vector<size_t> shape_scales = {static_cast<size_t>(N), 3};
    cnpy::npz_save(path, "log_scales", scales_cpu.data_ptr<float>(), shape_scales, "a");
    
    // unnorm_rotations: [N, 4] (高斯点的旋转四元数)
    std::vector<size_t> shape_rots = {static_cast<size_t>(N), 4};
    cnpy::npz_save(path, "unnorm_rotations", rotations_cpu.data_ptr<float>(), shape_rots, "a");
    
    // sh_coeffs: [N, 48] -> 保存为sh_coeffs_flat
    std::vector<size_t> shape_sh = {static_cast<size_t>(N), 48};
    cnpy::npz_save(path, "sh_coeffs_flat", sh_coeffs_cpu.data_ptr<float>(), shape_sh, "a");
    
    // logit_opacities: [N, 1] -> [N]
    auto opacity_squeezed = opacity_cpu.squeeze(1);  // [N, 1] -> [N]
    std::vector<size_t> shape_opacity = {static_cast<size_t>(N)};
    cnpy::npz_save(path, "logit_opacities", opacity_squeezed.data_ptr<float>(), shape_opacity, "a");
    
    // rgb_colors: 从SH0系数计算 [N, 3]
    // SH0系数是sh_coeffs的前3个通道
    // rgb = sh0 * C0 + 0.5, 其中 C0 = 0.28209479177387814
    const float C0 = 0.28209479177387814f;
    auto sh0 = sh_coeffs_cpu.slice(1, 0, 3);  // [N, 3] SH0系数
    auto rgb_colors_cpu = sh0 * C0 + 0.5f;  // [N, 3]
    // 限制在[0, 1]范围内
    rgb_colors_cpu = torch::clamp(rgb_colors_cpu, 0.0f, 1.0f);
    std::vector<size_t> shape_rgb = {static_cast<size_t>(N), 3};
    cnpy::npz_save(path, "rgb_colors", rgb_colors_cpu.data_ptr<float>(), shape_rgb, "a");
    
    // 保存相机姿态: cam_unnorm_rots [1, 4, num_frames]
    auto cam_rots_cpu = cam_unnorm_rots.detach().cpu().contiguous();
    std::vector<size_t> shape_cam_rots = {
        static_cast<size_t>(cam_rots_cpu.size(0)),
        static_cast<size_t>(cam_rots_cpu.size(1)),
        static_cast<size_t>(cam_rots_cpu.size(2))
    };
    cnpy::npz_save(path, "cam_unnorm_rots", cam_rots_cpu.data_ptr<float>(), shape_cam_rots, "a");
    
    // 保存相机平移: cam_trans [1, 3, num_frames]
    auto cam_trans_cpu = cam_trans.detach().cpu().contiguous();
    std::vector<size_t> shape_cam_trans = {
        static_cast<size_t>(cam_trans_cpu.size(0)),
        static_cast<size_t>(cam_trans_cpu.size(1)),
        static_cast<size_t>(cam_trans_cpu.size(2))
    };
    cnpy::npz_save(path, "cam_trans", cam_trans_cpu.data_ptr<float>(), shape_cam_trans, "a");
    
    // 保存相机内参: intrinsics [3, 3]
    auto intrinsics_cpu = intrinsics.detach().cpu().contiguous();
    std::vector<size_t> shape_intrinsics = {3, 3};
    cnpy::npz_save(path, "intrinsics", intrinsics_cpu.data_ptr<float>(), shape_intrinsics, "a");
    
    // 保存第一帧w2c: [4, 4]
    auto w2c_cpu = first_frame_w2c.detach().cpu().contiguous();
    std::vector<size_t> shape_w2c = {4, 4};
    cnpy::npz_save(path, "w2c", w2c_cpu.data_ptr<float>(), shape_w2c, "a");
    
    // 保存图像尺寸
    int32_t width_val = static_cast<int32_t>(org_width);
    int32_t height_val = static_cast<int32_t>(org_height);
    std::vector<size_t> shape_scalar = {1};
    cnpy::npz_save(path, "org_width", &width_val, shape_scalar, "a");
    cnpy::npz_save(path, "org_height", &height_val, shape_scalar, "a");
    
    // 保存timestep: [N]
    std::vector<size_t> shape_timestep = {static_cast<size_t>(N)};
    cnpy::npz_save(path, "timestep", timestep_cpu.data_ptr<float>(), shape_timestep, "a");
    
    // 保存gt_w2c_all_frames: [num_frames, 4, 4]（如果提供）
    if (gt_w2c_all_frames.defined() && gt_w2c_all_frames.numel() > 0) {
        auto gt_w2c_cpu = gt_w2c_all_frames.detach().cpu().contiguous();
        std::vector<size_t> shape_gt_w2c = {
            static_cast<size_t>(gt_w2c_cpu.size(0)),
            static_cast<size_t>(gt_w2c_cpu.size(1)),
            static_cast<size_t>(gt_w2c_cpu.size(2))
        };
        cnpy::npz_save(path, "gt_w2c_all_frames", gt_w2c_cpu.data_ptr<float>(), shape_gt_w2c, "a");
    }
    
    // 保存keyframe_time_indices（如果提供）
    if (!keyframe_time_indices.empty()) {
        std::vector<int32_t> keyframe_indices_int32(keyframe_time_indices.begin(), keyframe_time_indices.end());
        std::vector<size_t> shape_keyframes = {keyframe_indices_int32.size()};
        cnpy::npz_save(path, "keyframe_time_indices", keyframe_indices_int32.data(), shape_keyframes, "a");
    }
    
    std::cout << "Checkpoint saved to: " << path << " (N=" << N << ")" << std::endl;
}

void CheckpointIO::save(const std::string& path, const GaussianModel& model)
{
    // 简化版本：创建默认的相机参数（恒等变换，单帧）
    // 这对于不需要相机参数的工具（如post_splatam_opt, gaussian_splatting）很有用
    
    // 默认相机旋转：恒等四元数 [1, 0, 0, 0]
    auto cam_unnorm_rots = torch::zeros({1, 4, 1}, torch::kFloat32).to(kCUDA);
    cam_unnorm_rots[0][0][0] = 1.0f;  // w = 1
    
    // 默认相机平移：零平移
    auto cam_trans = torch::zeros({1, 3, 1}, torch::kFloat32).to(kCUDA);
    
    // 默认内参：假设640x480图像，FOV 60度
    auto intrinsics = torch::zeros({3, 3}, torch::kFloat32).to(kCUDA);
    float fx = 600.0f;
    float fy = 600.0f;
    float cx = 320.0f;
    float cy = 240.0f;
    intrinsics[0][0] = fx;
    intrinsics[1][1] = fy;
    intrinsics[0][2] = cx;
    intrinsics[1][2] = cy;
    intrinsics[2][2] = 1.0f;
    
    // 默认第一帧w2c：单位矩阵
    auto first_frame_w2c = torch::eye(4, torch::kFloat32).to(kCUDA);
    
    // 默认图像尺寸
    int org_width = 640;
    int org_height = 480;
    
    // 调用完整版本的save
    save(path, model, cam_unnorm_rots, cam_trans, intrinsics, first_frame_w2c, org_width, org_height);
}

void CheckpointIO::load(const std::string& path, GaussianModel& model)
{
    if (!fs::exists(path)) {
        throw std::runtime_error("Checkpoint file not found: " + path);
    }
    
    // 加载npz文件
    cnpy::npz_t npz_data = cnpy::npz_load(path);
    
    // 检查必需的关键字
    if (npz_data.find("means3D") == npz_data.end()) {
        throw std::runtime_error("Checkpoint missing 'means3D'");
    }
    
    // 获取高斯数量
    auto& means_array = npz_data["means3D"];
    int64_t N = means_array.shape[0];
    
    if (N == 0) {
        throw std::runtime_error("Checkpoint contains zero Gaussians");
    }
    
    // 检查模型容量
    if (N > model.getMaxGaussians()) {
        throw std::runtime_error("Checkpoint contains more Gaussians than model capacity: " + 
                                 std::to_string(N) + " > " + std::to_string(model.getMaxGaussians()));
    }
    
    // 加载means3D
    auto means3D_data = means_array.data<float>();
    auto means3D = torch::from_blob(
        const_cast<float*>(means3D_data), 
        {N, 3}, 
        torch::kFloat32
    ).clone().to(model.getDevice());
    
    // 加载log_scales (如果存在) 或 scales
    Tensor scales;
    if (npz_data.find("log_scales") != npz_data.end()) {
        auto& scales_array = npz_data["log_scales"];
        scales = torch::from_blob(
            const_cast<float*>(scales_array.data<float>()),
            {N, 3},
            torch::kFloat32
        ).clone().to(model.getDevice());
    } else if (npz_data.find("scales") != npz_data.end()) {
        auto& scales_array = npz_data["scales"];
        auto scales_exp = torch::from_blob(
            const_cast<float*>(scales_array.data<float>()),
            {N, 3},
            torch::kFloat32
        ).clone();
        scales = torch::log(scales_exp + 1e-8f).to(model.getDevice());
    } else {
        throw std::runtime_error("Checkpoint missing 'log_scales' or 'scales'");
    }
    
    // 加载rotations (cam_unnorm_rots 或 rotations)
    Tensor rotations;
    if (npz_data.find("cam_unnorm_rots") != npz_data.end()) {
        auto& rots_array = npz_data["cam_unnorm_rots"];
        rotations = torch::from_blob(
            const_cast<float*>(rots_array.data<float>()),
            {N, 4},
            torch::kFloat32
        ).clone().to(model.getDevice());
    } else if (npz_data.find("rotations") != npz_data.end()) {
        auto& rots_array = npz_data["rotations"];
        rotations = torch::from_blob(
            const_cast<float*>(rots_array.data<float>()),
            {N, 4},
            torch::kFloat32
        ).clone().to(model.getDevice());
    } else {
        throw std::runtime_error("Checkpoint missing 'cam_unnorm_rots' or 'rotations'");
    }
    
    // 加载sh_coeffs (sh_coeffs_flat 或 sh_coeffs)
    Tensor sh_coeffs;
    if (npz_data.find("sh_coeffs_flat") != npz_data.end()) {
        auto& sh_array = npz_data["sh_coeffs_flat"];
        int64_t sh_dim = (sh_array.shape.size() == 2) ? static_cast<int64_t>(sh_array.shape[1]) : 48;
        sh_coeffs = torch::from_blob(
            const_cast<float*>(sh_array.data<float>()),
            {N, sh_dim},
            torch::kFloat32
        ).clone().to(model.getDevice());
        
        // 如果维度不是48，需要padding或截断
        if (sh_dim < 48) {
            auto padding = torch::zeros({N, 48 - sh_dim}, torch::kFloat32).to(model.getDevice());
            sh_coeffs = torch::cat({sh_coeffs, padding}, 1);
        } else if (sh_dim > 48) {
            sh_coeffs = sh_coeffs.slice(1, 0, 48);
        }
    } else if (npz_data.find("sh_coeffs") != npz_data.end()) {
        auto& sh_array = npz_data["sh_coeffs"];
        // 可能是[N, 16, 3]格式，需要flatten
        if (sh_array.shape.size() == 3) {
            int64_t sh_N = static_cast<int64_t>(sh_array.shape[0]);
            int64_t sh_K = static_cast<int64_t>(sh_array.shape[1]);
            int64_t sh_C = static_cast<int64_t>(sh_array.shape[2]);
            auto sh_flat = torch::from_blob(
                const_cast<float*>(sh_array.data<float>()),
                {sh_N, sh_K * sh_C},
                torch::kFloat32
            ).clone().to(model.getDevice());
            
            if (sh_K * sh_C < 48) {
                auto padding = torch::zeros({N, 48 - sh_K * sh_C}, torch::kFloat32).to(model.getDevice());
                sh_coeffs = torch::cat({sh_flat, padding}, 1);
            } else {
                sh_coeffs = sh_flat.slice(1, 0, 48);
            }
        } else {
            sh_coeffs = torch::from_blob(
                const_cast<float*>(sh_array.data<float>()),
                {N, static_cast<int64_t>(sh_array.shape[1])},
                torch::kFloat32
            ).clone().to(model.getDevice());
        }
    } else {
        throw std::runtime_error("Checkpoint missing 'sh_coeffs_flat' or 'sh_coeffs'");
    }
    
    // 加载opacity (logit_opacities 或 opacity)
    Tensor opacity;
    if (npz_data.find("logit_opacities") != npz_data.end()) {
        auto& opacity_array = npz_data["logit_opacities"];
        auto opacity_1d = torch::from_blob(
            const_cast<float*>(opacity_array.data<float>()),
            {N},
            torch::kFloat32
        ).clone();
        opacity = opacity_1d.unsqueeze(1).to(model.getDevice());  // [N] -> [N, 1]
    } else if (npz_data.find("opacity") != npz_data.end()) {
        auto& opacity_array = npz_data["opacity"];
        auto opacity_raw = torch::from_blob(
            const_cast<float*>(opacity_array.data<float>()),
            {N},
            torch::kFloat32
        ).clone();
        // 假设是sigmoid空间，转换为logit空间
        opacity = torch::logit(opacity_raw.clamp(1e-7f, 1.0f - 1e-7f)).unsqueeze(1).to(model.getDevice());
    } else {
        throw std::runtime_error("Checkpoint missing 'logit_opacities' or 'opacity'");
    }
    
    // 加载timestep（可选）
    Tensor timestep;
    if (npz_data.find("timestep") != npz_data.end()) {
        auto& timestep_array = npz_data["timestep"];
        timestep = torch::from_blob(
            const_cast<float*>(timestep_array.data<float>()),
            {N},
            torch::kFloat32
        ).clone().to(model.getDevice());
    } else {
        // 如果没有timestep，创建一个空tensor（loadFromTensors会将其初始化为0）
        timestep = Tensor();
    }
    
    // 使用loadFromTensors直接加载参数
    model.loadFromTensors(means3D, scales, rotations, sh_coeffs, opacity, timestep);
    
    std::cout << "Checkpoint loaded from: " << path << " (N=" << N << ")" << std::endl;
}

} // namespace isogs
