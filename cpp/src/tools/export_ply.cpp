#include "core/gaussian_model.hpp"
#include "utils/io.hpp"
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace isogs;

void writePLY(const std::string& path, const Tensor& points, const Tensor& colors, const Tensor& normals) {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << path << std::endl;
        return;
    }
    
    int num_points = points.size(0);
    
    // PLY header
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << num_points << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    if (colors.size(0) > 0) {
        file << "property uchar red\n";
        file << "property uchar green\n";
        file << "property uchar blue\n";
    }
    if (normals.size(0) > 0) {
        file << "property float nx\n";
        file << "property float ny\n";
        file << "property float nz\n";
    }
    file << "end_header\n";
    
    // Write data
    auto points_cpu = points.cpu();
    auto colors_cpu = colors.cpu();
    auto normals_cpu = normals.cpu();
    
    for (int i = 0; i < num_points; ++i) {
        // Position
        file << std::fixed << std::setprecision(6)
             << points_cpu[i][0].item<float>() << " "
             << points_cpu[i][1].item<float>() << " "
             << points_cpu[i][2].item<float>();
        
        // Color
        if (colors.size(0) > 0) {
            int r = static_cast<int>(colors_cpu[i][0].item<float>() * 255);
            int g = static_cast<int>(colors_cpu[i][1].item<float>() * 255);
            int b = static_cast<int>(colors_cpu[i][2].item<float>() * 255);
            file << " " << r << " " << g << " " << b;
        }
        
        // Normal
        if (normals.size(0) > 0) {
            file << " " << normals_cpu[i][0].item<float>() << " "
                 << normals_cpu[i][1].item<float>() << " "
                 << normals_cpu[i][2].item<float>();
        }
        
        file << "\n";
    }
    
    file.close();
    std::cout << "PLY file saved to: " << path << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <checkpoint_path> <output_ply_path>" << std::endl;
        return 1;
    }
    
    std::string checkpoint_path = argv[1];
    std::string output_path = argv[2];
    
    // Load checkpoint
    GaussianModel model(10000000, kCUDA);  // 假设最大1000万高斯
    CheckpointIO::load(checkpoint_path, model);
    
    // Extract point cloud
    auto means = model.getMeans3D();
    auto sh_coeffs = model.getSHCoeffs();
    auto opacity = model.getOpacity();
    
    // Get active Gaussians
    auto alive_mask = model.getAliveMask();
    auto active_indices = torch::where(alive_mask)[0];
    
    if (active_indices.size(0) == 0) {
        std::cerr << "Error: No active Gaussians found" << std::endl;
        return 1;
    }
    
    // Extract active points
    auto active_means = means.index({active_indices});
    auto active_sh = sh_coeffs.index({active_indices});
    
    // Convert SH to RGB (simplified: use first 3 coefficients)
    auto colors = active_sh.slice(1, 0, 3);
    colors = torch::sigmoid(colors);  // Convert to [0, 1]
    
    // Extract normals (if available, otherwise use zeros)
    Tensor normals = torch::zeros({active_means.size(0), 3}, 
                                 torch::TensorOptions().dtype(torch::kFloat32).device(kCUDA));
    
    // Write PLY
    writePLY(output_path, active_means, colors, normals);
    
    return 0;
}

