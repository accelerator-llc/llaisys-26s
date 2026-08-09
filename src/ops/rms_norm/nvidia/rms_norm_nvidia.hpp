#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t val_type, size_t n, size_t d, float eps);

// 4.9.6 T2：融合 residual add + rms_norm（out_norm=rms_norm(x+residual,weight), residual_out=x+residual）。
void rms_norm_add(std::byte *out_norm, std::byte *residual_out,
                  const std::byte *x, const std::byte *residual, const std::byte *weight,
                  llaisysDataType_t val_type, size_t n, size_t d, float eps);
}
