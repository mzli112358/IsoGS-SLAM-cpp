#include "utils/optimizer.hpp"
#include <torch/torch.h>

// CUDA Kernel声明
extern "C" {
void launch_adam_update(
    torch::Tensor params,
    torch::Tensor grads,
    torch::Tensor m,
    torch::Tensor v,
    float lr,
    float beta1,
    float beta2,
    int step_count,
    float eps = 1e-8f
);
}

namespace isogs {

CudaAdamOptimizer::CudaAdamOptimizer(const std::map<std::string, float>& learning_rates)
{
    // 构造函数，learning_rates用于后续添加参数组
    (void)learning_rates;
}

void CudaAdamOptimizer::addParamGroup(const std::string& name, const Tensor& params, float lr)
{
    ParamGroup group;
    group.params = params.clone().requires_grad_(false);
    group.grad = torch::zeros_like(params);
    group.m = torch::zeros_like(params);
    group.v = torch::zeros_like(params);
    group.lr = lr;
    
    param_groups_[name] = std::move(group);
}

void CudaAdamOptimizer::step()
{
    step_count_++;
    
    for (auto& [name, group] : param_groups_) {
        // 调用CUDA Kernel更新参数
        launch_adam_update(
            group.params,
            group.grad,
            group.m,
            group.v,
            group.lr,
            group.beta1,
            group.beta2,
            step_count_,
            group.eps
        );
    }
}

void CudaAdamOptimizer::zeroGrad()
{
    for (auto& [name, group] : param_groups_) {
        group.grad.zero_();
    }
}

CudaAdamOptimizer::ParamGroup& CudaAdamOptimizer::getParamGroup(const std::string& name)
{
    auto it = param_groups_.find(name);
    if (it == param_groups_.end()) {
        throw std::runtime_error("ParamGroup not found: " + name);
    }
    return it->second;
}

bool CudaAdamOptimizer::hasParamGroup(const std::string& name) const
{
    return param_groups_.find(name) != param_groups_.end();
}

} // namespace isogs

