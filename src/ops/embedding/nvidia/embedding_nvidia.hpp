#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t idx_type, llaisysDataType_t val_type,
               size_t seq_len, size_t embd_dim, size_t num_emb);
}
