#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.hpp"
#endif

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // TO_BE_IMPLEMENTED();
    CHECK_SAME_DEVICE(out, in, weight);
    ASSERT(in->ndim() == 2, "Linear: input must be a 2D tensor.");
    ASSERT(weight->ndim() == 2, "Linear: weight must be a 2D tensor.");
    ASSERT(out->ndim() == 2, "Linear: out must be a 2D tensor.");
    ASSERT(in->isContiguous() && weight->isContiguous() && out->isContiguous(),
           "Linear: input/weight/out must be contiguous.");
    size_t n = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];
    ASSERT(weight->shape()[1] == in_features, "Linear: weight columns must match input feature dim.");
    ASSERT(out->shape()[0] == n && out->shape()[1] == out_features,
           "Linear: out must have shape (n, out_features).");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    // bias 可选：为空 shared_ptr 表示不提供偏置（对齐 PyTorch bias.defined()==false）。
    bool has_bias = static_cast<bool>(bias);
    if (has_bias) {
        CHECK_SAME_DEVICE(out, bias);
        ASSERT(bias->ndim() == 1 && bias->numel() == out_features,
               "Linear: bias must be a 1D tensor of length out_features.");
        CHECK_SAME_DTYPE(bias->dtype(), out->dtype());
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }

    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(),
                           has_bias ? bias->data() : nullptr, out->dtype(),
                           n, in_features, out_features, has_bias);
    }

    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(),
                           has_bias ? bias->data() : nullptr, out->dtype(),
                           n, in_features, out_features, has_bias);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        // TO_BE_IMPLEMENTED();
        return nvidia::linear(out->data(), in->data(), weight->data(),
                              has_bias ? bias->data() : nullptr, out->dtype(),
                              n, in_features, out_features, has_bias);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

// 4.9.6 T1：k+v 合并 GEMV（带 bias）。CPU 走两次 linear（零变化），NVIDIA 走 nvidia::linear_kv_fused。
void linear_kv_fused(tensor_t out_k, tensor_t out_v, tensor_t in,
                     tensor_t weight_k, tensor_t weight_v,
                     tensor_t bias_k, tensor_t bias_v) {
    CHECK_SAME_DEVICE(out_k, out_v, in, weight_k, weight_v);
    ASSERT(in->ndim() == 2, "Linear: input must be a 2D tensor.");
    ASSERT(weight_k->ndim() == 2 && weight_v->ndim() == 2, "Linear: weight must be 2D.");
    ASSERT(out_k->ndim() == 2 && out_v->ndim() == 2, "Linear: out must be 2D.");
    ASSERT(in->isContiguous() && weight_k->isContiguous() && weight_v->isContiguous() &&
               out_k->isContiguous() && out_v->isContiguous(),
           "Linear: tensors must be contiguous.");
    size_t n = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_k_features = weight_k->shape()[0];
    size_t out_v_features = weight_v->shape()[0];
    ASSERT(weight_k->shape()[1] == in_features && weight_v->shape()[1] == in_features,
           "Linear: weight columns must match input feature dim.");
    ASSERT(out_k->shape()[0] == n && out_k->shape()[1] == out_k_features &&
               out_v->shape()[0] == n && out_v->shape()[1] == out_v_features,
           "Linear: out must have shape (n, out_features).");
    CHECK_SAME_DTYPE(out_k->dtype(), in->dtype(), weight_k->dtype());
    bool has_bias_k = static_cast<bool>(bias_k);
    bool has_bias_v = static_cast<bool>(bias_v);
    if (has_bias_k) {
        CHECK_SAME_DEVICE(out_k, bias_k);
        ASSERT(bias_k->ndim() == 1 && bias_k->numel() == out_k_features, "Linear: bias_k shape.");
        CHECK_SAME_DTYPE(bias_k->dtype(), out_k->dtype());
        ASSERT(bias_k->isContiguous(), "Linear: bias_k must be contiguous.");
    }
    if (has_bias_v) {
        CHECK_SAME_DEVICE(out_v, bias_v);
        ASSERT(bias_v->ndim() == 1 && bias_v->numel() == out_v_features, "Linear: bias_v shape.");
        CHECK_SAME_DTYPE(bias_v->dtype(), out_v->dtype());
        ASSERT(bias_v->isContiguous(), "Linear: bias_v must be contiguous.");
    }

    if (weight_k->deviceType() == LLAISYS_DEVICE_CPU) {
        linear(out_k, in, weight_k, bias_k);
        linear(out_v, in, weight_v, bias_v);
        return;
    }
    llaisys::core::context().setDevice(weight_k->deviceType(), weight_k->deviceId());
    switch (weight_k->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        linear(out_k, in, weight_k, bias_k);
        linear(out_v, in, weight_v, bias_v);
        return;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear_kv_fused(out_k->data(), out_v->data(), in->data(),
                                       weight_k->data(), weight_v->data(),
                                       has_bias_k ? bias_k->data() : nullptr,
                                       has_bias_v ? bias_v->data() : nullptr,
                                       in->dtype(), n, in_features, out_k_features, out_v_features);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
