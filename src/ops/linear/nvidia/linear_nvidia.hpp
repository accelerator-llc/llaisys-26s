#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t val_type, size_t n, size_t in_features, size_t out_features, bool has_bias);

// 4.9.6 T1：k+v 合并 GEMV（带 bias）。C500 decode 小形状走手写合并 kernel；其余走两次 linear。
void linear_kv_fused(std::byte *out_k, std::byte *out_v, const std::byte *in,
                     const std::byte *w_k, const std::byte *w_v,
                     const std::byte *b_k, const std::byte *b_v,
                     llaisysDataType_t val_type, size_t n, size_t in_features,
                     size_t out_k_features, size_t out_v_features);
}
