#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, float scale) {
    // TO_BE_IMPLEMENTED();
    // 因果自注意力（含 GQA）：对每个 q head h（用 kv head h/g，g=nh/nkvh，对齐 torch
    // 的 repeat_interleave 复用 kv），计算 score=(Q·K^T)*scale，施因果掩码
    // （j > i+(kvlen-qlen) 置 -inf，等价 torch 的 tril(diagonal=S-L) 下三角掩码），
    // softmax over kvlen，最后与 V 加权求和，输出 (qlen,nh,hd)。
    // 数值：全程 float 域计算（f16/bf16 输入 cast 到 float，对齐 vLLM/llama.cpp CPU
    // attention 在 float 域累加、亦与 torch 中 half/bf16 的 acc_type=float 内部提升一致）；
    // softmax 减行最大值防 exp 溢出（对齐 torch.softmax 的 max-subtraction），掩码位
    // exp(-inf)=0 自动归零；最终 cast 回 T。
    size_t g = nh / nkvh; // GQA 分组数：每个 kv head 被 g 个 q head 共享
    int64_t causal_offset = static_cast<int64_t>(kvlen) - static_cast<int64_t>(qlen);
    const float neg_inf = -std::numeric_limits<float>::infinity();

    std::vector<float> scores(kvlen); // 复用：先存 Q·K score，softmax 后原地变权重
    std::vector<float> out_acc(hd);

    for (size_t h = 0; h < nh; h++) {
        size_t kvh = h / g; // 对齐 repeat_interleave：q head h 用 kv head h/g
        for (size_t i = 0; i < qlen; i++) {
            const T *q_vec = q + (i * nh + h) * hd;
            // Pass 1: Q·K^T * scale + 因果掩码
            for (size_t j = 0; j < kvlen; j++) {
                if (static_cast<int64_t>(j) > static_cast<int64_t>(i) + causal_offset) {
                    scores[j] = neg_inf;
                } else {
                    const T *k_vec = k + (j * nkvh + kvh) * hd;
                    float dot = 0.0f;
                    for (size_t d = 0; d < hd; d++) {
                        dot += llaisys::utils::cast<float>(q_vec[d]) * llaisys::utils::cast<float>(k_vec[d]);
                    }
                    scores[j] = dot * scale;
                }
            }
            // Pass 2: 数值稳定 softmax（减行最大值；掩码位 exp(-inf)=0 自动归零）
            float row_max = scores[0];
            for (size_t j = 1; j < kvlen; j++) {
                if (scores[j] > row_max) row_max = scores[j];
            }
            float exp_sum = 0.0f;
            for (size_t j = 0; j < kvlen; j++) {
                float e = std::exp(scores[j] - row_max);
                scores[j] = e;
                exp_sum += e;
            }
            float inv_sum = 1.0f / exp_sum;
            // Pass 3: 加权求和 V -> out
            for (size_t d = 0; d < hd; d++) out_acc[d] = 0.0f;
            for (size_t j = 0; j < kvlen; j++) {
                const T *v_vec = v + (j * nkvh + kvh) * hd;
                float w = scores[j] * inv_sum;
                for (size_t d = 0; d < hd; d++) {
                    out_acc[d] += w * llaisys::utils::cast<float>(v_vec[d]);
                }
            }
            T *out_vec = attn_val + (i * nh + h) * hd;
            for (size_t d = 0; d < hd; d++) out_vec[d] = llaisys::utils::cast<T>(out_acc[d]);
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t val_type,
                    size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, float scale) {
    switch (val_type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_<float>(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
                                      reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
                                      qlen, kvlen, nh, nkvh, hd, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                                                reinterpret_cast<const llaisys::bf16_t *>(q),
                                                reinterpret_cast<const llaisys::bf16_t *>(k),
                                                reinterpret_cast<const llaisys::bf16_t *>(v),
                                                qlen, kvlen, nh, nkvh, hd, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                                                reinterpret_cast<const llaisys::fp16_t *>(q),
                                                reinterpret_cast<const llaisys::fp16_t *>(k),
                                                reinterpret_cast<const llaisys::fp16_t *>(v),
                                                qlen, kvlen, nh, nkvh, hd, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
    }
}
} // namespace llaisys::ops::cpu
