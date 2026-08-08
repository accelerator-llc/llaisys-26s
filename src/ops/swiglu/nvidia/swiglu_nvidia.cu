#include "swiglu_nvidia.hpp"

#include "../../../device/nvidia/nvidia_utils.cuh"
#include "../../../utils.hpp"

namespace llaisys::ops::nvidia {
using llaisys::device::nvidia::to_float;
using llaisys::device::nvidia::from_float;

// 逐元素 SwiGLU kernel：out = up * SiLU(gate)，SiLU(x) = x / (1 + exp(-x))。
// 与 cpu::swiglu 一致：exp 在 float 域计算，f16/bf16 经 to_float/from_float 提升/写回。
// 极端值与 torch 行为一致：gate 极负 -> exp(-g) 溢出 inf -> silu 趋 0；gate 极正 -> exp(-g) 下溢 0 -> silu 趋 g。
template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float g = to_float(gate[idx]);
        float u = to_float(up[idx]);
        float silu = g / (1.0f + expf(-g));
        out[idx] = from_float<T>(u * silu);
    }
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t val_type, size_t numel) {
    if (numel == 0) {
        return;
    }
    constexpr int block = 256;
    int grid = static_cast<int>((numel + block - 1) / block);
    switch (val_type) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel<float><<<grid, block>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel<llaisys::bf16_t><<<grid, block>>>(
            reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(gate),
            reinterpret_cast<const llaisys::bf16_t *>(up), numel);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_kernel<llaisys::fp16_t><<<grid, block>>>(
            reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(gate),
            reinterpret_cast<const llaisys::fp16_t *>(up), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
