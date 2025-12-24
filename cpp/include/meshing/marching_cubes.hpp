#pragma once

#include "core/types.hpp"
#include <torch/torch.h>
#include <vector>

namespace isogs {

/**
 * @brief CUDA Marching Cubes网格提取
 */
class MarchingCubes {
public:
    struct Config {
        float voxel_size;
        float iso_level;
        float padding;
        
        Config() : voxel_size(0.02f), iso_level(1.0f), padding(0.5f) {}
    };
    
    MarchingCubes(const Config& config = Config());
    
    /**
     * @brief 从密度场提取网格
     * @param density_grid 密度场 [X*Y*Z] (flattened) 或 [X, Y, Z] (3D)
     * @param dims 维度 [X, Y, Z] (如果density_grid是flattened，需要提供)
     * @param origin 原点位置 [3]
     * @param spacing 体素间距 [3]
     * @return {vertices, faces, normals}
     */
    std::tuple<Tensor, Tensor, Tensor> extractMesh(
        const Tensor& density_grid,
        const Tensor& origin,
        const Tensor& spacing,
        const std::vector<int64_t>& dims = {}
    );
    
private:
    Config config_;
};

} // namespace isogs

