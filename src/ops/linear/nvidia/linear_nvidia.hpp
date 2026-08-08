#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t val_type, size_t n, size_t in_features, size_t out_features, bool has_bias);

// 优化2：q/k/v 三段 bias 合并一次 kernel（of0/of1/of2 为各段 out_features，可不同）。
void add_bias3(std::byte *y0, std::byte *y1, std::byte *y2,
               const std::byte *b0, const std::byte *b1, const std::byte *b2,
               llaisysDataType_t val_type, size_t of0, size_t of1, size_t of2, size_t n);

// 优化2：批量 GEMM（batch=2，同形状，无 bias）：out_a=in@w_a^T, out_b=in@w_b^T。
void linear_batched2(std::byte *out_a, std::byte *out_b, const std::byte *in,
                     const std::byte *w_a, const std::byte *w_b,
                     llaisysDataType_t val_type, size_t n, size_t in_features, size_t out_features);
}
