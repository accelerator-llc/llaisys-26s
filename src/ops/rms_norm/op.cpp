#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.hpp"
#endif
#include "../add/op.hpp"  // 4.9.6 T2：rms_norm_add 的 CPU 兜底走 add+rms_norm

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // TO_BE_IMPLEMENTED();
    CHECK_SAME_DEVICE(out, in, weight);
    ASSERT(in->ndim() == 2, "RmsNorm: input must be a 2D tensor.");
    ASSERT(out->ndim() == 2, "RmsNorm: out must be a 2D tensor.");
    ASSERT(in->isContiguous() && out->isContiguous() && weight->isContiguous(),
           "RmsNorm: all tensors must be contiguous.");
    size_t n = in->shape()[0];
    size_t d = in->shape()[1];
    ASSERT(out->shape()[0] == n && out->shape()[1] == d,
           "RmsNorm: out must have the same shape as input.");
    ASSERT(weight->ndim() == 1 && weight->numel() == d,
           "RmsNorm: weight must be a 1D tensor of length equal to the last dim of input.");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_ARGUMENT(d > 0, "RmsNorm: the last dim of input must be positive.");
    CHECK_ARGUMENT(eps > 0.0f, "RmsNorm: eps must be positive.");  // Fix CR#L24: 拒绝 eps=0，堵住全零行 sqrt(0)->inf->NaN 的注入入口

    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), in->dtype(), n, d, eps);
    }

    llaisys::core::context().setDevice(in->deviceType(), in->deviceId());

    switch (in->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), in->dtype(), n, d, eps);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        // TO_BE_IMPLEMENTED();
        return nvidia::rms_norm(out->data(), in->data(), weight->data(), in->dtype(), n, d, eps);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

// 4.9.6 T2：融合 residual add + rms_norm。CPU 走 add+rms_norm（零变化），NVIDIA 走 fused kernel。
void rms_norm_add(tensor_t out_norm, tensor_t residual_out,
                  tensor_t x, tensor_t residual, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out_norm, residual_out, x, residual, weight);
    ASSERT(x->ndim() == 2, "RmsNorm: input must be a 2D tensor.");
    ASSERT(out_norm->ndim() == 2 && residual_out->ndim() == 2, "RmsNorm: out must be 2D.");
    ASSERT(x->isContiguous() && out_norm->isContiguous() && residual_out->isContiguous() &&
               residual->isContiguous() && weight->isContiguous(),
           "RmsNorm: all tensors must be contiguous.");
    size_t n = x->shape()[0];
    size_t d = x->shape()[1];
    ASSERT(out_norm->shape()[0] == n && out_norm->shape()[1] == d &&
               residual_out->shape()[0] == n && residual_out->shape()[1] == d,
           "RmsNorm: out must have the same shape as input.");
    ASSERT(weight->ndim() == 1 && weight->numel() == d,
           "RmsNorm: weight must be a 1D tensor of length equal to the last dim of input.");
    CHECK_SAME_DTYPE(out_norm->dtype(), x->dtype(), residual->dtype(), weight->dtype());
    CHECK_ARGUMENT(d > 0, "RmsNorm: the last dim of input must be positive.");
    CHECK_ARGUMENT(eps > 0.0f, "RmsNorm: eps must be positive.");

    if (x->deviceType() == LLAISYS_DEVICE_CPU) {
        add(residual_out, x, residual);
        rms_norm(out_norm, residual_out, weight, eps);
        return;
    }
    llaisys::core::context().setDevice(x->deviceType(), x->deviceId());
    switch (x->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        add(residual_out, x, residual);
        rms_norm(out_norm, residual_out, weight, eps);
        return;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm_add(out_norm->data(), residual_out->data(),
                                    x->data(), residual->data(), weight->data(),
                                    x->dtype(), n, d, eps);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
