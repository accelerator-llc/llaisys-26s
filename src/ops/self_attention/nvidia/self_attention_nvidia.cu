#include "self_attention_nvidia.hpp"

#include "../../../device/nvidia/nvidia_utils.cuh"
#include "../../../utils.hpp"

#include <cmath>

namespace llaisys::ops::nvidia {
using llaisys::device::nvidia::to_float;
using llaisys::device::nvidia::from_float;
using llaisys::device::nvidia::block_reduce_sum;
using llaisys::device::nvidia::block_reduce_max;

constexpr int SA_BLOCK = 256;

// 因果自注意力（含 GQA）朴素实现：q@k^T -> ×scale -> 因果掩码 -> softmax -> @v。
// 每 block 处理一个 (i, h)（qlen, nh），block 内三阶段：
//   1) score[j] = (Q·K^T)*scale，因果掩码（j > i+(kvlen-qlen) 置 -inf），d 顺序累加（与 cpu 一致）
//   2) softmax：block reduce max/sum（减最大值防 exp 溢出，与 cpu 一致），scores[j] 变权重
//   3) out[d] = sum_j w[j]*v[j,d]，j 顺序累加（与 cpu 一致）
// GQA: q head h 用 kv head h/g（g=nh/nkvh，对齐 torch repeat_interleave）。
// softmax 归约参考 vLLM PagedAttention（warp shfl -> shared -> cross-warp -> broadcast）。
// 低精度全程 float 域；expf 对齐 cpu std::exp(float)；不加 1e-6（op.cpp 已拒 kvlen<qlen，不全掩码，对齐 cpu）。
template <typename T>
__global__ void self_attention_kernel(T *attn_val, const T *q, const T *k, const T *v,
                                      size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, float scale) {
    size_t i = blockIdx.x;
    size_t h = blockIdx.y;
    if (i >= qlen || h >= nh) {
        return;
    }
    size_t g = nh / nkvh;
    size_t kvh = h / g;  // GQA: q head h 用 kv head h/g
    int64_t causal_offset = static_cast<int64_t>(kvlen) - static_cast<int64_t>(qlen);
    const float neg_inf = -INFINITY;

    const T *q_vec = q + (i * nh + h) * hd;
    extern __shared__ float scores[];  // scores[kvlen]

    // 阶段1: score[j] = (Q·K^T)*scale + 因果掩码
    for (size_t j = threadIdx.x; j < kvlen; j += blockDim.x) {
        if (static_cast<int64_t>(j) > static_cast<int64_t>(i) + causal_offset) {
            scores[j] = neg_inf;
        } else {
            const T *k_vec = k + (j * nkvh + kvh) * hd;
            float dot = 0.0f;
            for (size_t d = 0; d < hd; d++) {
                dot += to_float(q_vec[d]) * to_float(k_vec[d]);
            }
            scores[j] = dot * scale;
        }
    }
    __syncthreads();

    // 阶段2a: row_max（block reduce，减最大值防 exp 溢出）
    float my_max = neg_inf;
    for (size_t j = threadIdx.x; j < kvlen; j += blockDim.x) {
        my_max = fmaxf(my_max, scores[j]);
    }
    __shared__ float s_warp[SA_BLOCK / 32];
    my_max = block_reduce_max<SA_BLOCK>(my_max, s_warp);
    __shared__ float s_max;
    if (threadIdx.x == 0) {
        s_max = my_max;
    }
    __syncthreads();
    float row_max = s_max;

    // 阶段2b: exp(scores-max) + sum
    float my_sum = 0.0f;
    for (size_t j = threadIdx.x; j < kvlen; j += blockDim.x) {
        float e = expf(scores[j] - row_max);
        scores[j] = e;
        my_sum += e;
    }
    my_sum = block_reduce_sum<SA_BLOCK>(my_sum, s_warp);
    __shared__ float s_sum;
    if (threadIdx.x == 0) {
        s_sum = my_sum;
    }
    __syncthreads();
    float inv_sum = 1.0f / s_sum;

    // scores[j] *= inv_sum（变权重）
    for (size_t j = threadIdx.x; j < kvlen; j += blockDim.x) {
        scores[j] *= inv_sum;
    }
    __syncthreads();

    // 阶段3: out[d] = sum_j w[j]*v[j,d]（j 顺序累加，与 cpu 一致）
    for (size_t d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (size_t j = 0; j < kvlen; j++) {
            const T *v_vec = v + (j * nkvh + kvh) * hd;
            acc += scores[j] * to_float(v_vec[d]);
        }
        attn_val[(i * nh + h) * hd + d] = from_float<T>(acc);
    }
}

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t val_type,
                    size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, float scale) {
    if (qlen == 0 || nh == 0) {
        return;
    }
    dim3 grid(static_cast<unsigned int>(qlen), static_cast<unsigned int>(nh));
    int block = SA_BLOCK;
    size_t smem_bytes = kvlen * sizeof(float);
    switch (val_type) {
    case LLAISYS_DTYPE_F32:
        self_attention_kernel<float><<<grid, block, smem_bytes>>>(
            reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
            qlen, kvlen, nh, nkvh, hd, scale);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_kernel<llaisys::bf16_t><<<grid, block, smem_bytes>>>(
            reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q),
            reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v),
            qlen, kvlen, nh, nkvh, hd, scale);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_kernel<llaisys::fp16_t><<<grid, block, smem_bytes>>>(
            reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q),
            reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v),
            qlen, kvlen, nh, nkvh, hd, scale);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
