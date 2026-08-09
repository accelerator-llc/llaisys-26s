#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.hpp"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // TO_BE_IMPLEMENTED();
    CHECK_SAME_DEVICE(out, in, pos_ids);
    ASSERT(in->ndim() == 3, "Rope: input must be a 3D tensor (seq_len, n_heads, head_dim).");
    ASSERT(out->ndim() == 3, "Rope: out must be a 3D tensor.");
    ASSERT(in->isContiguous() && out->isContiguous() && pos_ids->isContiguous(),
           "Rope: all tensors must be contiguous.");
    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];
    ASSERT(out->shape().size() == 3 && out->shape()[0] == seq_len &&
               out->shape()[1] == n_heads && out->shape()[2] == head_dim,
           "Rope: out must have the same shape as input.");
    ASSERT(pos_ids->ndim() == 1 && pos_ids->numel() == seq_len,
           "Rope: pos_ids must be a 1D tensor of length seq_len.");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(head_dim > 0 && head_dim % 2 == 0,
                   "Rope: head_dim must be a positive even number.");  // head_dim 须为正偶数，否则静默返回脏数据
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "Rope: pos_ids must be int64.");
    CHECK_ARGUMENT(theta > 0.0f, "Rope: theta must be positive.");
    // pos_ids 非负属语义校验（负位置不导致越界崩溃，仅算出错误 sin/cos），非内存安全防御；
    // rope 为热点不做 D2H，非负由调用方保证。

    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), in->dtype(),
                         pos_ids->dtype(), seq_len, n_heads, head_dim, theta);
    }

    llaisys::core::context().setDevice(in->deviceType(), in->deviceId());

    switch (in->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), in->dtype(),
                         pos_ids->dtype(), seq_len, n_heads, head_dim, theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        // TO_BE_IMPLEMENTED();
        return nvidia::rope(out->data(), in->data(), pos_ids->data(), in->dtype(),
                            pos_ids->dtype(), seq_len, n_heads, head_dim, theta);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
