#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t val_type,
            size_t n, size_t in_features, size_t out_features, bool has_bias);
}
