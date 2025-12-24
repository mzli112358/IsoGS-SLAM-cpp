#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <vector>
#include <memory>

namespace isogs {

using Tensor = torch::Tensor;
using Device = torch::Device;
using Dtype = torch::ScalarType;

// 基础类型定义
using Index = int64_t;
using Float = float;
using Int = int32_t;

// 向量类型
using Vec3 = torch::Tensor;  // [3]
using Vec4 = torch::Tensor;  // [4]
using Mat3 = torch::Tensor;  // [3, 3]
using Mat4 = torch::Tensor;  // [4, 4]

// 索引集合
using Indices = std::vector<Index>;

// 设备类型 (使用const而非constexpr，因为Device不是literal type)
const Device kCPU = torch::kCPU;
const Device kCUDA = torch::kCUDA;

} // namespace isogs

