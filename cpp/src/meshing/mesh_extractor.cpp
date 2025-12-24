#include "meshing/mesh_extractor.hpp"
#include "utils/spatial_hash.hpp"
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <filesystem>

namespace isogs {

// 从四元数构建旋转矩阵
Tensor buildRotationFromQuaternion(const Tensor& quats) {
    // quats: [N, 4] (w, x, y, z)
    int64_t N = quats.size(0);
    
    // 归一化四元数
    auto quats_norm = torch::nn::functional::normalize(quats, torch::nn::functional::NormalizeFuncOptions().dim(1).eps(1e-8));
    
    auto r = quats_norm.select(1, 0).unsqueeze(1);  // [N, 1] w
    auto x = quats_norm.select(1, 1).unsqueeze(1); // [N, 1] x
    auto y = quats_norm.select(1, 2).unsqueeze(1); // [N, 1] y
    auto z = quats_norm.select(1, 3).unsqueeze(1); // [N, 1] z
    
    // 构建旋转矩阵 [N, 3, 3]
    auto rot = torch::zeros({N, 3, 3}, quats.options());
    
    // 使用更简单的方式构建
    rot.select(1, 0).select(1, 0) = 1.0f - 2.0f * (y * y + z * z);
    rot.select(1, 0).select(1, 1) = 2.0f * (x * y - r * z);
    rot.select(1, 0).select(1, 2) = 2.0f * (x * z + r * y);
    
    rot.select(1, 1).select(1, 0) = 2.0f * (x * y + r * z);
    rot.select(1, 1).select(1, 1) = 1.0f - 2.0f * (x * x + z * z);
    rot.select(1, 1).select(1, 2) = 2.0f * (y * z - r * x);
    
    rot.select(1, 2).select(1, 0) = 2.0f * (x * z - r * y);
    rot.select(1, 2).select(1, 1) = 2.0f * (y * z + r * x);
    rot.select(1, 2).select(1, 2) = 1.0f - 2.0f * (x * x + y * y);
    
    return rot;
}

MeshExtractor::MeshExtractor()
    : marching_cubes_(MarchingCubes::Config())
{
}

std::tuple<Tensor, Tensor, Tensor> MeshExtractor::extractMesh(const GaussianModel& model)
{
    std::cout << "Extracting mesh from Gaussian model..." << std::endl;
    
    int64_t N = model.getActiveCount();
    if (N == 0) {
        std::cerr << "Warning: Model has no active Gaussians" << std::endl;
        auto vertices = torch::zeros({0, 3}, torch::kFloat32);
        auto faces = torch::zeros({0, 3}, torch::kInt64);
        auto normals = torch::zeros({0, 3}, torch::kFloat32);
        return std::make_tuple(vertices, faces, normals);
    }
    
    // 获取高斯参数
    auto means = model.getMeans3D();  // [N, 3]
    auto log_scales = model.getScales();  // [N, 3] (log空间)
    auto rotations = model.getRotations();  // [N, 4] (四元数)
    auto opacity = model.getOpacity();  // [N, 1] (logit空间)
    
    // 转换到实际空间
    auto scales = torch::exp(log_scales).clamp(1e-5f);  // [N, 3]
    auto opacities = torch::sigmoid(opacity).squeeze(1);  // [N]
    
    // 计算边界框
    auto min_result = means.min(0);
    auto max_result = means.max(0);
    auto means_min = std::get<0>(min_result);  // [3]
    auto means_max = std::get<0>(max_result);  // [3]
    
    MarchingCubes::Config mc_config;
    float padding = mc_config.padding;
    auto min_bounds = means_min - padding;
    auto max_bounds = means_max + padding;
    
    auto min_bounds_cpu = min_bounds.cpu();
    auto max_bounds_cpu = max_bounds.cpu();
    float min_x = min_bounds_cpu[0].item<float>();
    float min_y = min_bounds_cpu[1].item<float>();
    float min_z = min_bounds_cpu[2].item<float>();
    float max_x = max_bounds_cpu[0].item<float>();
    float max_y = max_bounds_cpu[1].item<float>();
    float max_z = max_bounds_cpu[2].item<float>();
    std::cout << "Bounding box: min=[" << min_x << ", " << min_y << ", " << min_z << "], "
              << "max=[" << max_x << ", " << max_y << ", " << max_z << "]" << std::endl;
    
    // 创建体素网格
    float voxel_size = mc_config.voxel_size;
    auto dims = (max_bounds - min_bounds) / voxel_size;
    auto dims_cpu = dims.cpu();
    float dim_x = dims_cpu[0].item<float>();
    float dim_y = dims_cpu[1].item<float>();
    float dim_z = dims_cpu[2].item<float>();
    int64_t nx = static_cast<int64_t>(std::ceil(dim_x));
    int64_t ny = static_cast<int64_t>(std::ceil(dim_y));
    int64_t nz = static_cast<int64_t>(std::ceil(dim_z));
    
    std::cout << "Voxel grid dimensions: [" << nx << ", " << ny << ", " << nz << "]" << std::endl;
    std::cout << "Total voxels: " << (nx * ny * nz) << std::endl;
    
    // 构建空间哈希表用于加速查询
    SpatialHash::Config hash_config;
    hash_config.cell_size = voxel_size * 2.0f;  // 使用稍大的cell size
    SpatialHash spatial_hash(hash_config);
    spatial_hash.build(means);
    spatial_hash.setMeans(means);
    
    // 构建逆协方差矩阵
    auto R = buildRotationFromQuaternion(rotations);  // [N, 3, 3]
    auto S_inv_sq = 1.0f / (scales * scales + 1e-8f);  // [N, 3]
    auto S_inv_sq_diag = torch::diag_embed(S_inv_sq);  // [N, 3, 3]
    auto R_S_inv_sq = torch::bmm(R, S_inv_sq_diag);  // [N, 3, 3]
    auto inv_covariances = torch::bmm(R_S_inv_sq, R.transpose(1, 2));  // [N, 3, 3]
    
    // 计算密度场
    auto density_grid = torch::zeros({nx * ny * nz}, means.options());
    float truncate_sigma = 3.0f;
    
    std::cout << "Computing density field..." << std::endl;
    
    // 对每个体素计算密度
    int64_t processed = 0;
    for (int64_t z = 0; z < nz; ++z) {
        for (int64_t y = 0; y < ny; ++y) {
            for (int64_t x = 0; x < nx; ++x) {
                // 计算体素中心的世界坐标
                float voxel_x = min_x + (x + 0.5f) * voxel_size;
                float voxel_y = min_y + (y + 0.5f) * voxel_size;
                float voxel_z = min_z + (z + 0.5f) * voxel_size;
                
                auto query_point = torch::tensor({{voxel_x, voxel_y, voxel_z}}, means.options());
                
                // 使用空间哈希查询K个最近邻
                int K = 32;  // 查询32个最近邻
                auto knn_indices = spatial_hash.queryKNN(query_point, K);  // [1, K]
                
                if (knn_indices.size(1) == 0) {
                    continue;
                }
                
                // 计算密度
                float density = 0.0f;
                for (int64_t k = 0; k < knn_indices.size(1); ++k) {
                    // 安全地从CUDA tensor提取索引（先移到CPU）
                    auto knn_indices_cpu = knn_indices.cpu();
                    int64_t gaussian_idx = knn_indices_cpu[0][k].item<int64_t>();
                    
                    // 计算delta = query_point - mean
                    auto mean = means[gaussian_idx];  // [3]
                    auto delta = query_point[0] - mean;  // [3]
                    
                    // 计算exp(-0.5 * delta^T * inv_cov * delta)
                    auto inv_cov = inv_covariances[gaussian_idx];  // [3, 3]
                    auto delta_expanded = delta.unsqueeze(0);  // [1, 3]
                    auto inv_cov_delta = torch::mm(delta_expanded, inv_cov);  // [1, 3]
                    auto quad_form = torch::mm(inv_cov_delta, delta.unsqueeze(1));  // [1, 1]
                    float quad_value = quad_form[0][0].item<float>();
                    
                    float exp_term = std::exp(-0.5f * quad_value);
                    float alpha = opacities[gaussian_idx].item<float>();
                    
                    density += alpha * exp_term;
                }
                
                int64_t voxel_idx = z * nx * ny + y * nx + x;
                density_grid[voxel_idx] = density;
                
                processed++;
                if (processed % 10000 == 0) {
                    std::cout << "  Processed " << processed << " / " << (nx * ny * nz) << " voxels" << std::endl;
                }
            }
        }
    }
    
    std::cout << "Density field computed" << std::endl;
    
    // 调用Marching Cubes提取网格
    auto origin = min_bounds;
    auto spacing = torch::tensor({voxel_size, voxel_size, voxel_size}, means.options());
    std::vector<int64_t> dims_vec = {nx, ny, nz};
    
    auto [vertices, faces, normals] = marching_cubes_.extractMesh(density_grid, origin, spacing, dims_vec);
    
    std::cout << "Mesh extraction completed: " << vertices.size(0) << " vertices, " 
              << faces.size(0) << " faces" << std::endl;
    
    return std::make_tuple(vertices, faces, normals);
}

void MeshExtractor::savePLY(const std::string& path,
                            const Tensor& vertices,
                            const Tensor& faces,
                            const Tensor& normals)
{
    std::cout << "Saving mesh to PLY file: " << path << std::endl;
    
    // 确保目录存在
    std::filesystem::path file_path(path);
    std::filesystem::create_directories(file_path.parent_path());
    
    // 转换到CPU
    auto vertices_cpu = vertices.cpu().contiguous();
    auto faces_cpu = faces.cpu().contiguous();
    auto normals_cpu = normals.cpu().contiguous();
    
    int64_t num_vertices = vertices_cpu.size(0);
    int64_t num_faces = faces_cpu.size(0);
    
    // 打开文件
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    
    // 写入PLY头部（ASCII格式）
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << num_vertices << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property float nx\n";
    file << "property float ny\n";
    file << "property float nz\n";
    file << "element face " << num_faces << "\n";
    file << "property list uchar int vertex_indices\n";
    file << "end_header\n";
    
    // 写入顶点和法线
    const float* vertices_data = vertices_cpu.data_ptr<float>();
    const float* normals_data = normals_cpu.data_ptr<float>();
    
    for (int64_t i = 0; i < num_vertices; ++i) {
        file << vertices_data[i * 3 + 0] << " "
             << vertices_data[i * 3 + 1] << " "
             << vertices_data[i * 3 + 2] << " "
             << normals_data[i * 3 + 0] << " "
             << normals_data[i * 3 + 1] << " "
             << normals_data[i * 3 + 2] << "\n";
    }
    
    // 写入面
    const int64_t* faces_data = faces_cpu.data_ptr<int64_t>();
    
    for (int64_t i = 0; i < num_faces; ++i) {
        file << "3 "  // 三角形有3个顶点
             << faces_data[i * 3 + 0] << " "
             << faces_data[i * 3 + 1] << " "
             << faces_data[i * 3 + 2] << "\n";
    }
    
    file.close();
    
    std::cout << "PLY file saved: " << num_vertices << " vertices, " 
              << num_faces << " faces" << std::endl;
}

} // namespace isogs
