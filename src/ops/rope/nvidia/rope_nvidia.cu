#include "rope_nvidia.hpp"

#include "../../../device/nvidia/nvidia_utils.cuh"
#include "../../../utils.hpp"

#include <cmath>

namespace llaisys::ops::nvidia {
using llaisys::device::nvidia::to_float;
using llaisys::device::nvidia::from_float;

constexpr int ROPE_BLOCK = 256;

// rope kernel：每 block 处理一个位置 pos，shared 预计算 cos/sin[half]（所有 head 复用），
// grid-stride 旋转 (head, i) 对。GPT-NeoX 半分旋转，先读 a/b 到局部再写双半，支持就地（out==in）。
// pos_ids 从显存读取；cos/sin 用 powf/cosf/sinf（float 域，与 cpu::rope 的 std::pow/cos/sin 一致）。
template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids,
                            size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    size_t p = blockIdx.x;
    if (p >= seq_len) {
        return;
    }
    size_t half = head_dim / 2;
    extern __shared__ float smem[];
    float *s_cos = smem;
    float *s_sin = smem + half;

    // 阶段1：预计算 cos/sin[half]（仅依赖 pos 与 i，与 head 无关，所有 head 复用）
    float pos = static_cast<float>(pos_ids[p]);
    for (size_t i = threadIdx.x; i < half; i += blockDim.x) {
        float exponent = 2.0f * static_cast<float>(i) / static_cast<float>(head_dim);
        float freq = pos / powf(theta, exponent);
        s_cos[i] = cosf(freq);
        s_sin[i] = sinf(freq);
    }
    __syncthreads();

    // 阶段2：旋转 (head, i)，grid-stride。先读 a,b 到局部再写，天然支持 out==in 别名。
    size_t total = n_heads * half;
    for (size_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
        size_t h = idx / half;
        size_t i = idx - h * half;
        const T *x_row = in + (p * n_heads + h) * head_dim;
        T *y_row = out + (p * n_heads + h) * head_dim;
        float a = to_float(x_row[i]);
        float b = to_float(x_row[i + half]);
        float c = s_cos[i];
        float s = s_sin[i];
        y_row[i] = from_float<T>(a * c - b * s);
        y_row[i + half] = from_float<T>(b * c + a * s);
    }
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t val_type, llaisysDataType_t pos_type,
          size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    ASSERT(pos_type == LLAISYS_DTYPE_I64, "Rope: pos_ids must be int64.");
    if (seq_len == 0) {
        return;
    }
    const int64_t *pos_ptr = reinterpret_cast<const int64_t *>(pos_ids);
    int grid = static_cast<int>(seq_len);
    size_t half = head_dim / 2;
    size_t smem_bytes = half * 2 * sizeof(float);
    switch (val_type) {
    case LLAISYS_DTYPE_F32:
        rope_kernel<float><<<grid, ROPE_BLOCK, smem_bytes>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), pos_ptr,
            seq_len, n_heads, head_dim, theta);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_kernel<llaisys::bf16_t><<<grid, ROPE_BLOCK, smem_bytes>>>(
            reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), pos_ptr,
            seq_len, n_heads, head_dim, theta);
        break;
    case LLAISYS_DTYPE_F16:
        rope_kernel<llaisys::fp16_t><<<grid, ROPE_BLOCK, smem_bytes>>>(
            reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), pos_ptr,
            seq_len, n_heads, head_dim, theta);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
