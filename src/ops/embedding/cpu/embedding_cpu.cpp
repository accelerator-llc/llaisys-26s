#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

template <typename IdxT, typename T>
void embedding_(T *out, const IdxT *index, const T *weight,
                size_t seq_len, size_t embd_dim, size_t num_emb) {
    // TO_BE_IMPLEMENTED();
    // 经典 embedding 查表：out[i, :] = weight[index[i], :]，按行连续 gather。
    // 对同类型 POD 逐元素赋值，与 add 算子逐元素风格保持一致。
    // 越界校验独立成前置循环，与计算分离（对齐 PyTorch gather kernel 的逐元素
    // bounds check 职责，留出干净的计算循环以利向量化和寄存器优化）。
    // Fix CR#L14-L17: index 合法性属输入校验，ASSERT 改 CHECK_ARGUMENT（与 op.cpp 层统一）。
    for (size_t i = 0; i < seq_len; i++) {
        CHECK_ARGUMENT(index[i] >= 0 && static_cast<size_t>(index[i]) < num_emb,
                       "Embedding: index out of range.");
    }
    for (size_t i = 0; i < seq_len; i++) {
        const T *src = weight + static_cast<size_t>(index[i]) * embd_dim;
        T *dst = out + i * embd_dim;
        for (size_t j = 0; j < embd_dim; j++) {
            dst[j] = src[j];
        }
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t idx_type, llaisysDataType_t val_type,
               size_t seq_len, size_t embd_dim, size_t num_emb) {
    CHECK_ARGUMENT(idx_type == LLAISYS_DTYPE_I64, "Embedding: index must be int64.");  // Fix CR#L31: 输入校验统一用 CHECK_ARGUMENT

    const int64_t *idx_ptr = reinterpret_cast<const int64_t *>(index);

    switch (val_type) {
    case LLAISYS_DTYPE_F32:
        return embedding_<int64_t, float>(reinterpret_cast<float *>(out), idx_ptr,
                                           reinterpret_cast<const float *>(weight),
                                           seq_len, embd_dim, num_emb);
    case LLAISYS_DTYPE_BF16:
        return embedding_<int64_t, llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out), idx_ptr,
                                                    reinterpret_cast<const llaisys::bf16_t *>(weight),
                                                    seq_len, embd_dim, num_emb);
    case LLAISYS_DTYPE_F16:
        return embedding_<int64_t, llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out), idx_ptr,
                                                   reinterpret_cast<const llaisys::fp16_t *>(weight),
                                                   seq_len, embd_dim, num_emb);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
    }
}
} // namespace llaisys::ops::cpu
