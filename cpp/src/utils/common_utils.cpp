#include "utils/common_utils.hpp"
#include <torch/torch.h>
#include <random>

namespace isogs {

void seedEverything(int seed) {
    // Set random seed
    std::srand(seed);
    std::mt19937 gen(seed);
    
    // Set torch seed
    torch::manual_seed(seed);
    if (torch::cuda::is_available()) {
        torch::cuda::manual_seed(seed);
        torch::cuda::manual_seed_all(seed);
    }
    
    // Set deterministic mode
    torch::globalContext().setDeterministicCuDNN(true);
    torch::globalContext().setBenchmarkCuDNN(false);
}

Tensor paramsToCPU(const Tensor& tensor) {
    return tensor.cpu();
}

Tensor paramsToNumpy(const Tensor& tensor) {
    // 返回CPU tensor，可以通过Pybind11转换为numpy
    return tensor.cpu().contiguous();
}

} // namespace isogs

