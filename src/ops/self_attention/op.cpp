#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // TO_BE_IMPLEMENTED();
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    ASSERT(q->ndim() == 3, "SelfAttention: q must be a 3D tensor (qlen, nh, hd).");
    ASSERT(k->ndim() == 3, "SelfAttention: k must be a 3D tensor (kvlen, nkvh, hd).");
    ASSERT(v->ndim() == 3, "SelfAttention: v must be a 3D tensor (kvlen, nkvh, hd).");
    ASSERT(attn_val->ndim() == 3, "SelfAttention: attn_val must be a 3D tensor (qlen, nh, hd).");
    ASSERT(q->isContiguous() && k->isContiguous() && v->isContiguous() && attn_val->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");
    size_t qlen = q->shape()[0];
    size_t nh = q->shape()[1];
    size_t hd = q->shape()[2];
    size_t kvlen = k->shape()[0];
    size_t nkvh = k->shape()[1];
    ASSERT(k->shape()[2] == hd, "SelfAttention: k must have head_dim matching q.");
    ASSERT(v->shape()[0] == kvlen && v->shape()[1] == nkvh && v->shape()[2] == hd,
           "SelfAttention: v must have the same shape as k.");
    ASSERT(attn_val->shape().size() == 3 && attn_val->shape()[0] == qlen &&
               attn_val->shape()[1] == nh && attn_val->shape()[2] == hd,
           "SelfAttention: attn_val must have shape (qlen, nh, hd).");
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    CHECK_ARGUMENT(hd > 0, "SelfAttention: head_dim must be positive.");
    CHECK_ARGUMENT(kvlen > 0, "SelfAttention: kvlen must be positive.");
    CHECK_ARGUMENT(nkvh > 0 && nh % nkvh == 0,
                   "SelfAttention: nh must be a positive multiple of nkvh (GQA).");

    if (q->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), q->dtype(),
                                   qlen, kvlen, nh, nkvh, hd, scale);
    }

    llaisys::core::context().setDevice(q->deviceType(), q->deviceId());

    switch (q->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), q->dtype(),
                                   qlen, kvlen, nh, nkvh, hd, scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
