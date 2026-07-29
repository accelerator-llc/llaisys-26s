#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    // TO_BE_IMPLEMENTED();
    // 逐元素 SwiGLU: out = up * SiLU(gate), SiLU(x) = x / (1 + exp(-x)) = x*sigmoid(x)。
    // 对齐 torch_swiglu: exp 在 float 域计算（gate.float()），f16/bf16 输入 cast 到 float
    // 完成 silu 与乘法再 cast 回 T（对齐 vLLM/llama.cpp 低精度激活在 float 域计算的做法，
    // 亦与 torch 中 exp 经 .float() 提升的语义一致）。极端值：gate 极负时 exp(-gate)
    // 溢出为 inf -> gate/inf = 0（silu 正确趋零）；gate 极正时 exp(-gate) 下溢为 0 ->
    // silu = gate（正确趋同 gate），与 torch 行为一致。
    for (size_t i = 0; i < numel; i++) {
        float g = llaisys::utils::cast<float>(gate[i]);
        float u = llaisys::utils::cast<float>(up[i]);
        float silu = g / (1.0f + std::exp(-g));
        out[i] = llaisys::utils::cast<T>(u * silu);
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t val_type, size_t numel) {
    switch (val_type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_<float>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate),
                              reinterpret_cast<const float *>(up), numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out),
                                        reinterpret_cast<const llaisys::bf16_t *>(gate),
                                        reinterpret_cast<const llaisys::bf16_t *>(up), numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out),
                                        reinterpret_cast<const llaisys::fp16_t *>(gate),
                                        reinterpret_cast<const llaisys::fp16_t *>(up), numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
    }
}
} // namespace llaisys::ops::cpu
