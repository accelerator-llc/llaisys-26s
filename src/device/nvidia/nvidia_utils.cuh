#pragma once

// NVIDIA CUDA 设备通用工具：CUDA API 返回码检查宏、拷贝方向枚举映射、device 端类型转换。
// 供 nvidia_runtime_api.cu 及各算子 nvidia 实现复用。

#include "llaisys.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "../../utils.hpp"

// CUDA API 返回码统一检查：失败时经 ASSERT 抛出 llaisys 异常，与项目 CHECK/ASSERT 风格一致。
#define CUDA_CHECK(call)                                                         \
    do {                                                                                                                                 \
        cudaError_t _llaisys_cuda_err = (call);                                                \
        ASSERT(_llaisys_cuda_err == cudaSuccess,                                           \
               "CUDA error (" #call "): " << cudaGetErrorString(_llaisys_cuda_err));                           \
    } while (0)

// cuBLAS API 返回码统一检查：cublasStatus_t 与 cudaError_t 不同类型，单独宏。
#define CUBLAS_CHECK(call)                                                              \
    do {                                                                                \
        cublasStatus_t _llaisys_cublas_err = (call);                                   \
        ASSERT(_llaisys_cublas_err == CUBLAS_STATUS_SUCCESS,                           \
               "cuBLAS error (" #call "): " << cublasGetStatusString(_llaisys_cublas_err)); \
    } while (0)

namespace llaisys::device::nvidia {

// 将 llaisys 拷贝方向枚举映射为 CUDA cudaMemcpyKind（显式映射，不依赖枚举数值顺序）。
inline cudaMemcpyKind toCudaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    default:
        ASSERT(false, "nvidia: unsupported memcpy kind " << static_cast<int>(kind));
        return cudaMemcpyDefault;
    }
}

// warp 内求和归约（shfl 蝴蝶）。归约后 lane 0 持 warp 和。
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// block 内求和归约：warp shfl + shared cross-warp。归约后 thread 0 持 sum。
// smem 需至少 NUM_WARPS 个 float（NUM_WARPS = BLOCK_SIZE / 32）。
// 参考 PyTorch BlockReduceSum / vLLM rms_norm 的 cub::BlockReduce 思路（自实现避免依赖 cub）。
template <int BLOCK_SIZE>
__device__ inline float block_reduce_sum(float val, float *smem) {
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;
    val = warp_reduce_sum(val);
    if (lane == 0) {
        smem[wid] = val;
    }
    __syncthreads();
    val = (tid < NUM_WARPS) ? smem[tid] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

// warp 内求最大值归约（shfl + fmaxf）。
__device__ inline float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// block 内求最大值归约：warp shfl + shared cross-warp，identity=-INFINITY。
// 参考 vLLM PagedAttention 的 qk_max 归约（warp shfl -> shared -> cross-warp -> broadcast）。
template <int BLOCK_SIZE>
__device__ inline float block_reduce_max(float val, float *smem) {
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;
    val = warp_reduce_max(val);
    if (lane == 0) {
        smem[wid] = val;
    }
    __syncthreads();
    val = (tid < NUM_WARPS) ? smem[tid] : -INFINITY;
    if (wid == 0) {
        val = warp_reduce_max(val);
    }
    return val;
}

// device 端 T <-> float 转换：bf16/fp16 经 CUDA intrinsic，数值与 host utils::cast 一致。
// 供各算子 nvidia kernel 复用，保证低精度在 float 域计算（与 cpu 算子语义对齐）。
template <typename T>
__device__ inline float to_float(T val);

template <>
__device__ inline float to_float<float>(float val) { return val; }

template <>
__device__ inline float to_float<llaisys::bf16_t>(llaisys::bf16_t val) {
    return __bfloat162float(__ushort_as_bfloat16(val._v));
}

template <>
__device__ inline float to_float<llaisys::fp16_t>(llaisys::fp16_t val) {
    return __half2float(__ushort_as_half(val._v));
}

template <typename T>
__device__ inline T from_float(float val);

template <>
__device__ inline float from_float<float>(float val) { return val; }

template <>
__device__ inline llaisys::bf16_t from_float<llaisys::bf16_t>(float val) {
    llaisys::bf16_t r;
    r._v = __bfloat16_as_ushort(__float2bfloat16(val));
    return r;
}

template <>
__device__ inline llaisys::fp16_t from_float<llaisys::fp16_t>(float val) {
    llaisys::fp16_t r;
    r._v = __half_as_ushort(__float2half(val));
    return r;
}

} // namespace llaisys::device::nvidia
