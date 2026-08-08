#include "argmax_nvidia.hpp"

#include "../../../device/nvidia/nvidia_utils.cuh"
#include "../../../utils.hpp"

#include <cmath>

namespace llaisys::ops::nvidia {
using llaisys::device::nvidia::to_float;
using llaisys::device::nvidia::from_float;

// 单 block 归约线程数（32 的倍数）。argmax 规模（测试 <=4096，模型 logits voc~1.5e5）
// 低于 vLLM topK 的多 block 阈值（2e5），单 block 是 vLLM/PyTorch 的标准做法。
constexpr int ARGMAX_BLOCK = 512;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = ARGMAX_BLOCK / WARP_SIZE;

// 比较函数：返回 b 是否应替换 a 作为当前最优（argmax）。
// 对齐 cpu::argmax 与 torch.max 语义：
// - NaN 传播：任一为 NaN 时 NaN 胜出；两者皆 NaN 取 idx 小（首个 NaN）。
// - 平局首个：值相等时取 idx 小（仅在严格更大或 NaN 时才替换）。
__device__ inline bool argmax_better(float bv, size_t bi, float av, size_t ai) {
    bool a_nan = isnan(av), b_nan = isnan(bv);
    if (a_nan && b_nan) {
        return bi < ai;
    }
    if (b_nan) {
        return true;
    }
    if (a_nan) {
        return false;
    }
    return bv > av || (bv == av && bi < ai);
}

// warp 内归约 (val, idx)：参考 PyTorch WarpReduceMax / vLLM paged_attention 的 red_smem 归约思路。
// 用 __shfl_down_sync 做蝴蝶归约，归约后 lane 0 持有该 warp 的最优 (val, idx)。
__device__ inline void warp_reduce_argmax(float &val, size_t &idx) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(0xffffffff, val, offset);
        size_t oi = __shfl_down_sync(0xffffffff, idx, offset);
        if (argmax_better(ov, oi, val, idx)) {
            val = ov;
            idx = oi;
        }
    }
}

// 1D argmax 归约 kernel：单 block + grid-stride + warp shfl 归约 + cross-warp 归约。
// 每 thread grid-stride 维护局部 (max_val, max_idx)；warp 归约后各 warp lane 0 写共享内存；
// warp 0 再归约得全局最优；thread 0 写 max_val/max_idx。
// identity 用 -INFINITY 确保任意有限值（含 -FLT_MAX）胜出；低精度经 to_float 提升 float 比较。
template <typename T>
__global__ void argmax_kernel(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    __shared__ float sval[NUM_WARPS];
    __shared__ size_t sidx[NUM_WARPS];

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;

    float my_val = -INFINITY;
    size_t my_idx = 0;
    for (size_t i = tid; i < numel; i += blockDim.x) {
        float v = to_float(vals[i]);
        if (argmax_better(v, i, my_val, my_idx)) {
            my_val = v;
            my_idx = i;
        }
    }

    warp_reduce_argmax(my_val, my_idx);
    if (lane == 0) {
        sval[wid] = my_val;
        sidx[wid] = my_idx;
    }
    __syncthreads();

    if (wid == 0) {
        my_val = (lane < NUM_WARPS) ? sval[lane] : -INFINITY;
        my_idx = (lane < NUM_WARPS) ? sidx[lane] : 0;
        warp_reduce_argmax(my_val, my_idx);
        if (lane == 0) {
            *max_idx = static_cast<int64_t>(my_idx);
            *max_val = from_float<T>(my_val);
        }
    }
}

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t idx_type, llaisysDataType_t val_type, size_t numel) {
    CHECK_ARGUMENT(idx_type == LLAISYS_DTYPE_I64, "Argmax: index output must be int64.");
    CHECK_ARGUMENT(numel > 0, "Argmax: input tensor must have at least one element.");
    int64_t *idx_ptr = reinterpret_cast<int64_t *>(max_idx);
    switch (val_type) {
    case LLAISYS_DTYPE_F32:
        argmax_kernel<float><<<1, ARGMAX_BLOCK>>>(
            idx_ptr, reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(vals), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_kernel<llaisys::bf16_t><<<1, ARGMAX_BLOCK>>>(
            idx_ptr, reinterpret_cast<llaisys::bf16_t *>(max_val), reinterpret_cast<const llaisys::bf16_t *>(vals), numel);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_kernel<llaisys::fp16_t><<<1, ARGMAX_BLOCK>>>(
            idx_ptr, reinterpret_cast<llaisys::fp16_t *>(max_val), reinterpret_cast<const llaisys::fp16_t *>(vals), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
