#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // TO_BE_IMPLEMENTED();
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(weight->ndim() == 2, "Embedding: weight must be a 2D tensor (num_embeddings, embedding_dim).");
    ASSERT(index->ndim() == 1, "Embedding: index must be a 1D tensor.");
    ASSERT(out->ndim() == 2, "Embedding: out must be a 2D tensor.");
    ASSERT(weight->isContiguous() && index->isContiguous() && out->isContiguous(),
           "Embedding: all tensors must be contiguous.");
    size_t num_emb = weight->shape()[0];
    size_t embd_dim = weight->shape()[1];
    size_t seq_len = index->numel();
    ASSERT(out->shape().size() == 2 && out->shape()[0] == seq_len && out->shape()[1] == embd_dim,
           "Embedding: out must have shape (seq_len, embedding_dim).");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index must be int64.");

    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              index->dtype(), weight->dtype(), seq_len, embd_dim, num_emb);
    }

    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(),
                              index->dtype(), weight->dtype(), seq_len, embd_dim, num_emb);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
