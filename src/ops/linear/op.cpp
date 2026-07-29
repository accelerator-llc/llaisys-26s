#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

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
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
