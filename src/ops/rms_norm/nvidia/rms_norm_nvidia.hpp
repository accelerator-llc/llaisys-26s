#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t val_type, size_t n, size_t d, float eps);
}
