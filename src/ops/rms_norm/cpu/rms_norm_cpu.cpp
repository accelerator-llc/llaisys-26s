#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t n, size_t d, float eps) {
    // TO_BE_IMPLEMENTED();
    // Y_i = W_i * X_i / sqrt(mean(X^2) + eps)，逐行归一化。
    // 对齐 vLLM CPU rms_norm_impl：平方和在 float 域累加（低精度必须提升），
    // 用 1.0f/std::sqrt(mean_sq + eps) 一次算出 inv_rms（vLLM CPU 亦用 1.0f/sqrtf
    // 而非 rsqrtf，与代码一致），再做 x * inv_rms * w。算法对齐以 test/ops PyTorch 参考为准。
    for (size_t nn = 0; nn < n; nn++) {
        const T *x_row = in + nn * d;
        T *y_row = out + nn * d;
        float var = 0.0f;
        for (size_t i = 0; i < d; i++) {
            float fx = llaisys::utils::cast<float>(x_row[i]);
            var += fx * fx;
        }
        float inv_rms = 1.0f / std::sqrt(var / static_cast<float>(d) + eps);
        for (size_t i = 0; i < d; i++) {
            float fy = llaisys::utils::cast<float>(x_row[i]) * inv_rms * llaisys::utils::cast<float>(weight[i]);
            y_row[i] = llaisys::utils::cast<T>(fy);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t val_type, size_t n, size_t d, float eps) {
    switch (val_type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_<float>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                                reinterpret_cast<const float *>(weight), n, d, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out),
                                           reinterpret_cast<const llaisys::bf16_t *>(in),
                                           reinterpret_cast<const llaisys::bf16_t *>(weight), n, d, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out),
                                          reinterpret_cast<const llaisys::fp16_t *>(in),
                                          reinterpret_cast<const llaisys::fp16_t *>(weight), n, d, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
    }
}
} // namespace llaisys::ops::cpu
