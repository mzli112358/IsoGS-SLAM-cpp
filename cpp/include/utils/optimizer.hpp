#pragma once

#include "core/types.hpp"
#include <torch/torch.h>
#include <map>
#include <string>

namespace isogs {

/**
 * @brief CUDA Adam优化器
 * 
 * 手写CUDA Kernel实现Adam更新，避免PyTorch Autograd的开销
 */
class CudaAdamOptimizer {
public:
    struct ParamGroup {
        Tensor params;
        Tensor grad;
        Tensor m;  // 一阶矩估计
        Tensor v;  // 二阶矩估计
        float lr;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;
    };
    
    CudaAdamOptimizer(const std::map<std::string, float>& learning_rates);
    ~CudaAdamOptimizer() = default;
    
    // 添加参数组
    void addParamGroup(const std::string& name, const Tensor& params, float lr);
    
    // 更新参数（调用CUDA Kernel）
    void step();
    
    // 清零梯度
    void zeroGrad();
    
    // 获取参数组（用于访问和更新）
    ParamGroup& getParamGroup(const std::string& name);
    
    // 检查参数组是否存在
    bool hasParamGroup(const std::string& name) const;
    
private:
    std::map<std::string, ParamGroup> param_groups_;
    int step_count_ = 0;
};

} // namespace isogs

