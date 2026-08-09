// Qwen2 大语言模型 C++ 实现。
// Qwen2Model 类：持有 meta、权重张量与 KV-Cache，提供 create/destroy/reset/infer；
// 前向复用 llaisys::ops 算子（embedding/linear/rms_norm/rope/self_attention/
// swiglu/argmax/add）。

#pragma once

#include <cstdint>
#include <vector>

#include "../../tensor/tensor.hpp"
#include "llaisys/models/qwen2.h"

namespace llaisys::models {

// Qwen2 因果语言模型（DeepSeek-R1-Distill-Qwen-1.5B 架构）。
// 前向：token embedding -> N×Transformer 层（RMSNorm + GQA self-attn + RoPE + 残差，
// RMSNorm + SwiGLU MLP + 残差）-> final RMSNorm -> lm_head -> argmax。
// KV-Cache 跨 infer 调用保持，reset() 清零；权重经 weights() 暴露后由上层填充。
class Qwen2Model {
public:
    // 按 meta 预分配全部权重张量与 per-layer KV-Cache buffer。
    // 支持 CPU 与 NVIDIA 单设备（device_id 取自 device_ids[0]）。
    Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int device_id);
    ~Qwen2Model();

    Qwen2Model(const Qwen2Model &) = delete;
    Qwen2Model &operator=(const Qwen2Model &) = delete;

    // 暴露权重结构体，供上层（C API -> Python tensorLoad）填充各权重数据。
    LlaisysQwen2Weights *weights() { return &_weights; }
    const LlaisysQwen2Meta &meta() const { return _meta; }

    // 清空 KV-Cache（cur_len 归零），使 generate 可重复调用。
    void reset();

    // 对 token_ids[0..ntoken) 做一次前向（prefill 或 decode 增量），更新 KV-Cache，
    // 返回最后一个位置的 argmax next token。
    int64_t infer(const int64_t *token_ids, size_t ntoken);

private:
    // 单层 Transformer 前向：hidden(in/out, (ntoken,hs)) 经 attn+mlp 与两次残差。
    void forward_layer(size_t layer, tensor_t &hidden, const tensor_t &pos_ids, size_t ntoken);

    // 分配/释放 _weights 中所有 llaisysTensor_t 与 per-layer 数组。
    void allocate_weights();
    void release_weights();

    // 张量池：临时 buffer 预分配复用（容量只增不减），消除热循环 cudaMalloc/cudaFree。
    // forward_layer slot 跨层共享（层间顺序执行，同流有序安全）；infer slot 跨步复用。
    // 对齐 PyTorch CUDACachingAllocator / CUDA stream-ordered allocator 的同流复用语义。
    struct TensorPool {
        // forward_layer 临时 slot
        tensor_t normed, q, k, v, attn_val, o, hidden_attn, normed2, gate, up, act, down, hidden_mlp;
        // infer 临时 slot
        tensor_t tokens, pos_ids, hidden, final_normed, logits, max_idx, max_val;
    } _pool;
    // 取池中 buffer 的 ntoken 切片；容量不足时扩容（只增不减）。shape[0]=ntoken。
    tensor_t scratch(tensor_t &pool, llaisysDataType_t dt, const std::vector<size_t> &shape);

    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device;
    int _device_id;

    LlaisysQwen2Weights _weights;   // 字段为 llaisysTensor_t，由 allocate/release 管理
    std::vector<tensor_t> _k_cache; // per-layer (maxseq, nkvh, dh)
    std::vector<tensor_t> _v_cache;
    size_t _cur_len = 0;
};

} // namespace llaisys::models
