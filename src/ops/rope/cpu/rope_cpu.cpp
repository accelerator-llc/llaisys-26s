#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids,
           size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    // TO_BE_IMPLEMENTED();
    // GPT-NeoX / Llama 风格旋转位置编码：将 head_dim 拆成前后两半 [x_a, x_b]，
    // 对第 i 对 (x_a[i]=x[i], x_b[i]=x[i+d/2]) 以角频率 freq_i = pos / theta^(2i/d)
    // 做旋转：y_a = x_a*cos - x_b*sin,  y_b = x_b*cos + x_a*sin。
    // 半分格式与旋转符号对齐 torch_rope，并对齐 llama.cpp GPT-NeoX 风格 RoPE 的
    // 半分旋转（dst0=x0*cos-x1*sin, dst1=x0*sin+x1*cos，各 backend 保留 is_neox 路径）。
    // Fix CR#L16: 注释收敛符号名（旧称 ggml_compute_forward_rope_f32 已模板化为
    // ggml_compute_forward_rope），算法对齐以 test/ops 的 PyTorch 参考为准。
    size_t half = head_dim / 2;

    // theta^(2i/d) 每个频率只算一次（与位置无关），对齐 torch 的 theta**(2i/d)；
    // 用 std::pow 在 float 域计算（llama.cpp 同样走 powf 的 float 路径）。
    std::vector<float> base(half);
    for (size_t i = 0; i < half; i++) {
        float exponent = 2.0f * static_cast<float>(i) / static_cast<float>(head_dim);
        base[i] = std::pow(theta, exponent);
    }

    // sin/cos 仅依赖 (pos, i)，按位置构建后向所有 head 复用（对齐 HF transformers /
    // llama.cpp 预计算 cos/sin cache 的做法），主计算循环按 head 连续遍历以利缓存。
    std::vector<float> cos_row(half);
    std::vector<float> sin_row(half);
    for (size_t p = 0; p < seq_len; p++) {
        float pos = static_cast<float>(pos_ids[p]);
        for (size_t i = 0; i < half; i++) {
            float freq = pos / base[i]; // 对齐 torch: positions / theta**(2i/d)
            cos_row[i] = std::cos(freq);
            sin_row[i] = std::sin(freq);
        }
        for (size_t h = 0; h < n_heads; h++) {
            const T *x_row = in + (p * n_heads + h) * head_dim;
            T *y_row = out + (p * n_heads + h) * head_dim;
            // f16/bf16 在 float 域完成旋转再 cast 回 T（对齐 llama.cpp f16 RoPE 提升
            // 到 float 计算的做法，亦与 torch 中 f16*f32->f32 的类型提升语义一致）。
            // 先读 a,b 到局部再写双半，天然支持 out 与 in 别名（in-place）。
            for (size_t i = 0; i < half; i++) {
                float a = llaisys::utils::cast<float>(x_row[i]);
                float b = llaisys::utils::cast<float>(x_row[i + half]);
                float c = cos_row[i];
                float s = sin_row[i];
                y_row[i] = llaisys::utils::cast<T>(a * c - b * s);
                y_row[i + half] = llaisys::utils::cast<T>(b * c + a * s);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids,
          llaisysDataType_t val_type, llaisysDataType_t pos_type,
          size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    ASSERT(pos_type == LLAISYS_DTYPE_I64, "Rope: pos_ids must be int64.");
    const int64_t *pos_ptr = reinterpret_cast<const int64_t *>(pos_ids);

    switch (val_type) {
    case LLAISYS_DTYPE_F32:
        return rope_<float>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                            pos_ptr, seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out),
                                      reinterpret_cast<const llaisys::bf16_t *>(in),
                                      pos_ptr, seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out),
                                      reinterpret_cast<const llaisys::fp16_t *>(in),
                                      pos_ptr, seq_len, n_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
    }
}
} // namespace llaisys::ops::cpu
