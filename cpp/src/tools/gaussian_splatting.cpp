#include "core/gaussian_model.hpp"
#include "datasets/dataset_factory.hpp"
#include "slam/mapper.hpp"
#include "utils/io.hpp"
#include "utils/common_utils.hpp"
#include <torch/torch.h>
#include <iostream>
#include <string>

using namespace isogs;

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <dataset_type> <dataset_path> <sequence> [output_dir]" << std::endl;
        return 1;
    }
    
    std::string dataset_type = argv[1];
    std::string dataset_path = argv[2];
    std::string sequence = argv[3];
    std::string output_dir = (argc > 4) ? argv[4] : "./output";
    
    // Set seed
    seedEverything(42);
    
    // Load dataset
    auto dataset = DatasetFactory::create(dataset_type, dataset_path, sequence);
    
    // Initialize model from first frame
    GaussianModel model(10000000, kCUDA);
    // TODO: Initialize from first frame
    
    // Mapper config
    Mapper::Config mapper_config;
    mapper_config.num_iters = 30000;  // More iterations for offline training
    
    Mapper mapper(mapper_config);
    
    // Training loop
    int num_frames = dataset->size();
    for (int frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
        auto [color, depth, intrinsics, pose] = (*dataset)[frame_idx];
        
        // Create keyframe
        Keyframe keyframe;
        keyframe.frame_id = frame_idx;
        // TODO: Setup camera and add keyframe
        
        // Optimize
        mapper.addKeyframe(keyframe);
        mapper.optimize(model, keyframe);
        
        // Save checkpoint periodically
        if (frame_idx % 100 == 0) {
            std::string ckpt_path = output_dir + "/params" + std::to_string(frame_idx) + ".npz";
            CheckpointIO::save(ckpt_path, model);
        }
    }
    
    // Save final checkpoint
    std::string final_ckpt = output_dir + "/params.npz";
    CheckpointIO::save(final_ckpt, model);
    
    std::cout << "Training completed!" << std::endl;
    return 0;
}

