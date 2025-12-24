#include "core/gaussian_model.hpp"
#include "utils/io.hpp"
#include "slam/mapper.hpp"
#include <torch/torch.h>
#include <iostream>
#include <string>

using namespace isogs;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <checkpoint_path> <output_path> [num_iters]" << std::endl;
        return 1;
    }
    
    std::string checkpoint_path = argv[1];
    std::string output_path = argv[2];
    int num_iters = (argc > 3) ? std::stoi(argv[3]) : 30000;
    
    // Load checkpoint
    GaussianModel model(10000000, kCUDA);
    CheckpointIO::load(checkpoint_path, model);
    
    // Mapper config with more iterations
    Mapper::Config mapper_config;
    mapper_config.num_iters = num_iters;
    
    Mapper mapper(mapper_config);
    
    // TODO: Load all keyframes from checkpoint or dataset
    // For now, this is a placeholder
    
    // Global optimization
    // TODO: Implement global optimization using all keyframes
    
    // Save optimized model
    CheckpointIO::save(output_path, model);
    
    std::cout << "Post-optimization completed!" << std::endl;
    return 0;
}

