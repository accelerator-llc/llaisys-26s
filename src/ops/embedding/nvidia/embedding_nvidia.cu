#include "embedding_nvidia.hpp"

#include "../../../device/nvidia/nvidia_utils.cuh"
#include "../../../utils.hpp"

#include <vector>

namespace llaisys::ops::nvidia {

// embedding 查表 kernel：out[i, :] = weight[index[i], :]。
// 每 block 处理一行，block 内 thread 以 grid-stride 拷贝行内元素，warp 内连续
// thread 访问连续列（合并访问）；同类型 POD 逐元素赋值，与 cpu::embedding 一致。
// index 合法性已在 host 层经 D2H 校验（见 embedding()），kernel 内不再 guard。
template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight,
                                 size_t seq_len, size_t embd_dim) {
    size_t i = blockIdx.x;
    if (i >= seq_len) {
        return;
    }
    size_t row = static_cast<size_t>(index[i]);
    const T *src = weight + row * embd_dim;
    T *dst = out + i * embd_dim;
    for (size_t j = threadIdx.x; j < embd_dim; j += blockDim.x) {
        dst[j] = src[j];
    }
}

void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t idx_type, llaisysDataType_t val_type,
               size_t seq_len, size_t embd_dim, size_t num_emb) {
    CHECK_ARGUMENT(idx_type == LLAISYS_DTYPE_I64, "Embedding: index must be int64.");
    if (seq_len == 0) {
        return;
    }
    const int64_t *idx_ptr = reinterpret_cast<const int64_t *>(index);

    // 越界校验：D2H index，host 循环 CHECK_ARGUMENT（与 cpu::embedding 前置校验一致）。
    // CUDA kernel 无法抛异常，故在 host 层校验；embedding 非热点，index 通常很小（seq_len 个 int64），
    // D2H 开销可接受。
    std::vector<int64_t> host_idx(seq_len);
    CUDA_CHECK(cudaMemcpy(host_idx.data(), idx_ptr, seq_len * sizeof(int64_t), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < seq_len; i++) {
        CHECK_ARGUMENT(host_idx[i] >= 0 && static_cast<size_t>(host_idx[i]) < num_emb,
                       "Embedding: index out of range.");
    }

    constexpr int block = 256;
    int grid = static_cast<int>(seq_len);
    switch (val_type) {
    case LLAISYS_DTYPE_F32:
        embedding_kernel<float><<<grid, block>>>(
            reinterpret_cast<float *>(out), idx_ptr, reinterpret_cast<const float *>(weight),
            seq_len, embd_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_kernel<llaisys::bf16_t><<<grid, block>>>(
            reinterpret_cast<llaisys::bf16_t *>(out), idx_ptr, reinterpret_cast<const llaisys::bf16_t *>(weight),
            seq_len, embd_dim);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_kernel<llaisys::fp16_t><<<grid, block>>>(
            reinterpret_cast<llaisys::fp16_t *>(out), idx_ptr, reinterpret_cast<const llaisys::fp16_t *>(weight),
            seq_len, embd_dim);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
