#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/argmax_nvidia.hpp"
#endif

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // TO_BE_IMPLEMENTED();
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    // 暂仅支持 1D 连续输入；max_idx / max_val 各容纳单个值。
    ASSERT(vals->ndim() == 1, "Argmax: vals must be a 1D tensor.");
    ASSERT(vals->isContiguous(), "Argmax: vals must be contiguous.");
    ASSERT(vals->numel() > 0, "Argmax: vals must have at least one element.");
    ASSERT(max_idx->numel() == 1 && max_val->numel() == 1,
           "Argmax: max_idx and max_val must hold a single element.");
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: max_idx must be int64.");
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());

    // always support cpu calculation
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(),
                           max_idx->dtype(), vals->dtype(), vals->numel());
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(),
                           max_idx->dtype(), vals->dtype(), vals->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        // TO_BE_IMPLEMENTED();
        return nvidia::argmax(max_idx->data(), max_val->data(), vals->data(),
                              max_idx->dtype(), vals->dtype(), vals->numel());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
