#include "rms_norm_nvidia.hpp"

#include "../../../device/nvidia/nvidia_utils.cuh"
#include "../../../utils.hpp"

#include <cmath>

namespace llaisys::ops::nvidia {
using llaisys::device::nvidia::to_float;
using llaisys::device::nvidia::from_float;
using llaisys::device::nvidia::block_reduce_sum;

constexpr int RMS_BLOCK = 256;

// rms_norm kernel：每 block 处理一行，行内块归约 sum(x^2)。
// 参考 vLLM rms_norm_kernel（每 block 一行 + grid-stride + block reduce + shared inv_rms）。
// 数值对齐 cpu::rms_norm：var 在 float 域累加，inv_rms = 1.0f/sqrtf(var/d + eps)
// （用 1.0f/sqrtf 而非 rsqrtf，与 CPU 一致），y = x * inv_rms * w。
template <typename T>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight, size_t n, size_t d, float eps) {
    size_t row = blockIdx.x;
    if (row >= n) {
        return;
    }
    const T *x_row = in + row * d;
    T *y_row = out + row * d;

    // 阶段1：行内 grid-stride 累加 x^2（float 域）
    float var = 0.0f;
    for (size_t i = threadIdx.x; i < d; i += blockDim.x) {
        float fx = to_float(x_row[i]);
        var += fx * fx;
    }
    __shared__ float s_warp[RMS_BLOCK / 32];
    var = block_reduce_sum<RMS_BLOCK>(var, s_warp);

    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = 1.0f / sqrtf(var / static_cast<float>(d) + eps);
    }
    __syncthreads();

    // 阶段2：y = x * inv_rms * w（float 域，写回 T）
    float inv_rms = s_inv_rms;
    for (size_t i = threadIdx.x; i < d; i += blockDim.x) {
        float fy = to_float(x_row[i]) * inv_rms * to_float(weight[i]);
        y_row[i] = from_float<T>(fy);
    }
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t val_type, size_t n, size_t d, float eps) {
    if (n == 0) {
        return;
    }
    int grid = static_cast<int>(n);
    switch (val_type) {
    case LLAISYS_DTYPE_F32:
        rms_norm_kernel<float><<<grid, RMS_BLOCK>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight), n, d, eps);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_kernel<llaisys::bf16_t><<<grid, RMS_BLOCK>>>(
            reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight), n, d, eps);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_kernel<llaisys::fp16_t><<<grid, RMS_BLOCK>>>(
            reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight), n, d, eps);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 4.9.6 方向 A：rms_norm + residual add 融合（省 1 launch/层）。
// 融合点：add(out=residual_out, x, residual) + rms_norm(out_norm, residual_out, weight)。
// 一个 kernel 同时输出 residual_out = x + residual（保留供后续 add 用）与
// out_norm = rms_norm(residual_out, weight)。数值与原 add+rms_norm 等价（float 域）。
// ============================================================================
template <typename T>
__global__ void fused_add_rmsnorm_kernel(T *out_norm, T *residual_out,
                                         const T *x, const T *residual, const T *weight,
                                         size_t n, size_t d, float eps) {
    size_t row = blockIdx.x;
    if (row >= n) {
        return;
    }
    const T *x_row = x + row * d;
    const T *r_row = residual + row * d;
    T *y_row = out_norm + row * d;
    T *ro_row = residual_out + row * d;

    // 阶段1：residual_out = x + residual，同时累加 var
    float var = 0.0f;
    for (size_t i = threadIdx.x; i < d; i += blockDim.x) {
        float fx = to_float(x_row[i]);
        float fr = to_float(r_row[i]);
        float sum = fx + fr;
        ro_row[i] = from_float<T>(sum);
        var += sum * sum;
    }
    __shared__ float s_warp[RMS_BLOCK / 32];
    var = block_reduce_sum<RMS_BLOCK>(var, s_warp);
    __shared__ float s_inv_rms;
    if (threadIdx.x == 0) {
        s_inv_rms = 1.0f / sqrtf(var / static_cast<float>(d) + eps);
    }
    __syncthreads();
    // 阶段2：out_norm = residual_out * inv_rms * weight
    float inv_rms = s_inv_rms;
    for (size_t i = threadIdx.x; i < d; i += blockDim.x) {
        float fy = to_float(ro_row[i]) * inv_rms * to_float(weight[i]);
        y_row[i] = from_float<T>(fy);
    }
}

// rms_norm_add：融合 residual add + rms_norm。NVIDIA 路径（C500/5090 均执行，T5 实测裁决）。
void rms_norm_add(std::byte *out_norm, std::byte *residual_out,
                  const std::byte *x, const std::byte *residual, const std::byte *weight,
                  llaisysDataType_t val_type, size_t n, size_t d, float eps) {
    if (n == 0) {
        return;
    }
    int grid = static_cast<int>(n);
    switch (val_type) {
    case LLAISYS_DTYPE_F32:
        fused_add_rmsnorm_kernel<float><<<grid, RMS_BLOCK>>>(
            reinterpret_cast<float *>(out_norm), reinterpret_cast<float *>(residual_out),
            reinterpret_cast<const float *>(x), reinterpret_cast<const float *>(residual),
            reinterpret_cast<const float *>(weight), n, d, eps);
        break;
    case LLAISYS_DTYPE_BF16:
        fused_add_rmsnorm_kernel<llaisys::bf16_t><<<grid, RMS_BLOCK>>>(
            reinterpret_cast<llaisys::bf16_t *>(out_norm), reinterpret_cast<llaisys::bf16_t *>(residual_out),
            reinterpret_cast<const llaisys::bf16_t *>(x), reinterpret_cast<const llaisys::bf16_t *>(residual),
            reinterpret_cast<const llaisys::bf16_t *>(weight), n, d, eps);
        break;
    case LLAISYS_DTYPE_F16:
        fused_add_rmsnorm_kernel<llaisys::fp16_t><<<grid, RMS_BLOCK>>>(
            reinterpret_cast<llaisys::fp16_t *>(out_norm), reinterpret_cast<llaisys::fp16_t *>(residual_out),
            reinterpret_cast<const llaisys::fp16_t *>(x), reinterpret_cast<const llaisys::fp16_t *>(residual),
            reinterpret_cast<const llaisys::fp16_t *>(weight), n, d, eps);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
