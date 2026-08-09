#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);

// 4.9.6 T2：融合 residual add + rms_norm。NVIDIA 走 fused kernel，CPU 走 add+rms_norm（兜底）。
// out_norm = rms_norm(x + residual, weight)；residual_out = x + residual（保留供后续 add）。
void rms_norm_add(tensor_t out_norm, tensor_t residual_out,
                  tensor_t x, tensor_t residual, tensor_t weight, float eps);
}
