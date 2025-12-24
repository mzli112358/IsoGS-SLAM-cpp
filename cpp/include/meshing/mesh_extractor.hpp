#pragma once

#include "core/gaussian_model.hpp"
#include "meshing/marching_cubes.hpp"
#include <string>

namespace isogs {

/**
 * @brief 网格提取器 - 从高斯模型提取Mesh
 */
class MeshExtractor {
public:
    MeshExtractor();
    
    /**
     * @brief 从高斯模型提取网格
     * @param model 高斯模型
     * @return {vertices, faces, normals}
     */
    std::tuple<Tensor, Tensor, Tensor> extractMesh(const GaussianModel& model);
    
    /**
     * @brief 保存网格到PLY文件
     */
    void savePLY(const std::string& path, 
                 const Tensor& vertices,
                 const Tensor& faces,
                 const Tensor& normals);
    
private:
    MarchingCubes marching_cubes_;
};

} // namespace isogs

