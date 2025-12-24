#include <iostream>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include "slam/slam_loop.hpp"
#include <filesystem>

namespace fs = std::filesystem;
using namespace isogs;

int main(int argc, char* argv[]) {
    std::cout << "IsoGS-SLAM C++ Version" << std::endl;
    std::cout << "=======================" << std::endl;
    
    // 检查CUDA可用性
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available!" << std::endl;
        return 1;
    }
    
    // 显式设置CUDA设备并初始化CUBLAS
    cudaSetDevice(0);
    // 通过创建一个小的CUDA tensor和矩阵乘法来触发CUBLAS初始化
    try {
        auto dummy = torch::eye(4, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        auto dummy2 = torch::matmul(dummy, dummy);
        torch::cuda::synchronize();
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to initialize CUBLAS: " << e.what() << std::endl;
    }
    
    // 打印GPU信息
    std::cout << "CUDA available: " << torch::cuda::is_available() << std::endl;
    std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
    if (torch::cuda::is_available()) {
        int device_id = 0;
        cudaGetDevice(&device_id);
        std::cout << "Current device: " << device_id << std::endl;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        std::cout << "Device name: " << prop.name << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Total memory: " << prop.totalGlobalMem / (1024 * 1024 * 1024) << " GB" << std::endl;
    }
    std::cout << std::endl;
    
    // 检查参数
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <dataset_path> <sequence> [end_frame] [checkpoint_dir]" << std::endl;
        std::cerr << "Example: " << argv[0] << " ./data/Replica room0" << std::endl;
        std::cerr << "Example: " << argv[0] << " ./data/Replica room0 100 ./checkpoints" << std::endl;
        return 1;
    }
    
    std::string dataset_path = argv[1];
    std::string sequence = argv[2];
    int end_frame = (argc > 3) ? std::stoi(argv[3]) : -1;  // -1表示处理所有帧
    std::string checkpoint_dir = (argc > 4) ? argv[4] : "./checkpoints";
    
    // 创建checkpoint目录
    fs::create_directories(checkpoint_dir);
    
    // 配置SLAM
    SLAMLoop::Config config;
    config.dataset_type = "replica";
    config.dataset_path = dataset_path;
    config.sequence = sequence;
    config.start_frame = 0;
    config.end_frame = end_frame;
    config.stride = 1;
    config.map_every = 5;  // 每5帧进行一次Mapping
    config.keyframe_every = 5;  // 每5帧添加一个关键帧
    config.use_gt_poses = false;  // 使用Tracking优化位姿
    
    // Tracking配置
    config.tracking_config.num_iters = 10;
    config.tracking_config.use_sil_for_loss = true;
    config.tracking_config.sil_thres = 0.99f;
    config.tracking_config.use_l1 = true;
    
    // Mapping配置
    config.mapping_config.num_iters = 40;
    config.mapping_config.add_new_gaussians = true;
    config.mapping_config.prune_gaussians = true;
    
    // Checkpoint配置
    config.save_checkpoints = true;
    config.checkpoint_interval = 50;  // 每50帧保存一次checkpoint
    config.checkpoint_dir = checkpoint_dir;
    config.load_checkpoint_path = "";  // 不加载checkpoint，从头开始
    
    // 运行SLAM
    try {
        SLAMLoop slam_loop(config);
        slam_loop.run();
        
        // 保存最终checkpoint（使用end_frame作为time_idx，如果为-1则使用一个大数字）
        // 注意：实际在run()中已经按checkpoint_interval保存了，这里只是保存最终状态
        int final_frame_idx = (end_frame > 0) ? end_frame : 10000;  // 如果未指定end_frame，使用一个大数字作为占位符
        std::string final_checkpoint = checkpoint_dir + "/params_final.npz";
        slam_loop.saveCheckpoint(final_checkpoint, final_frame_idx);
        std::cout << "Final checkpoint saved to: " << final_checkpoint << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error running SLAM: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

