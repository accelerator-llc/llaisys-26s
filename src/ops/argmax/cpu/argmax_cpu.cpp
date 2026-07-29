#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    // TO_BE_IMPLEMENTED();
    // 对齐 torch.max(vals, dim=-1, keepdim=True) 语义：平局时返回首个最大值的索引。
    // 仅在严格更大时更新，保证首个最大值胜出。
    // 低精度类型提升到 float 比较，与 add 算子的提升风格保持一致。
    float best = llaisys::utils::cast<float>(vals[0]);
    size_t best_idx = 0;
    for (size_t i = 1; i < numel; i++) {
        float cur = llaisys::utils::cast<float>(vals[i]);
        if (cur > best) {
            best = cur;
            best_idx = i;
        }
    }
    *max_idx = static_cast<int64_t>(best_idx);
    *max_val = llaisys::utils::cast<T>(best);
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t idx_type, llaisysDataType_t val_type, size_t numel) {
    ASSERT(idx_type == LLAISYS_DTYPE_I64, "Argmax: index output must be int64.");
    ASSERT(numel > 0, "Argmax: input tensor must have at least one element.");
    int64_t *idx_ptr = reinterpret_cast<int64_t *>(max_idx);

    switch (val_type) {
    case LLAISYS_DTYPE_F32:
        return argmax_<float>(idx_ptr, reinterpret_cast<float *>(max_val),
                              reinterpret_cast<const float *>(vals), numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_<llaisys::bf16_t>(idx_ptr, reinterpret_cast<llaisys::bf16_t *>(max_val),
                                        reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
    case LLAISYS_DTYPE_F16:
        return argmax_<llaisys::fp16_t>(idx_ptr, reinterpret_cast<llaisys::fp16_t *>(max_val),
                                        reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
    }
}
} // namespace llaisys::ops::cpu
