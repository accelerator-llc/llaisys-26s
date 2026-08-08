#include "add_nvidia.hpp"

#include "../../../device/nvidia/nvidia_utils.cuh"
#include "../../../utils.hpp"

namespace llaisys::ops::nvidia {
using llaisys::device::nvidia::to_float;
using llaisys::device::nvidia::from_float;

// 逐元素加法 kernel：c = a + b，低精度在 float 域计算（与 cpu::add 语义一致）。
template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        c[idx] = from_float<T>(to_float(a[idx]) + to_float(b[idx]));
    }
}

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    if (numel == 0) {
        return;
    }
    constexpr int block = 256;
    int grid = static_cast<int>((numel + block - 1) / block);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel<float><<<grid, block>>>(
            reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        add_kernel<llaisys::bf16_t><<<grid, block>>>(
            reinterpret_cast<llaisys::bf16_t *>(c), reinterpret_cast<const llaisys::bf16_t *>(a),
            reinterpret_cast<const llaisys::bf16_t *>(b), numel);
        break;
    case LLAISYS_DTYPE_F16:
        add_kernel<llaisys::fp16_t><<<grid, block>>>(
            reinterpret_cast<llaisys::fp16_t *>(c), reinterpret_cast<const llaisys::fp16_t *>(a),
            reinterpret_cast<const llaisys::fp16_t *>(b), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
