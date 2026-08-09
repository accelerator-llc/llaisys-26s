#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);

// 4.9.6 T1：k+v 合并 GEMV（带 bias）。NVIDIA 走 nvidia::linear_kv_fused（C500 decode 手写合并），
// CPU 走两次 linear（设备分派保护，CPU 行为零变化）。
void linear_kv_fused(tensor_t out_k, tensor_t out_v, tensor_t in,
                     tensor_t weight_k, tensor_t weight_v,
                     tensor_t bias_k, tensor_t bias_v);
}
