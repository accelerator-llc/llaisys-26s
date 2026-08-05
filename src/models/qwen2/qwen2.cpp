// Qwen2 大语言模型 C++ 实现（作业3）。
// 前向：embedding -> 逐层 Transformer 块（attn_norm + GQA self-attn + RoPE + 残差，
// mlp_norm + SwiGLU MLP + 残差）-> final norm -> lm_head -> argmax。
// KV-Cache 逐 token 增量更新。

#include "qwen2.hpp"

#include <cmath>

#include "../../llaisys/llaisys_tensor.hpp"  // LlaisysTensor 完整定义
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../utils.hpp"

namespace llaisys::models {

using llaisys::Tensor;
using llaisys::tensor_t;
using llaisys::ops::add;
using llaisys::ops::argmax;
using llaisys::ops::embedding;
using llaisys::ops::linear;
using llaisys::ops::rms_norm;
using llaisys::ops::rope;
using llaisys::ops::self_attention;
using llaisys::ops::swiglu;

// ============================================================================
// 构造 / 析构
// ============================================================================

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int device_id)
    : _meta(meta), _device(device), _device_id(device_id), _weights{}, _cur_len(0) {
    // V1 仅支持 CPU 单设备；多设备留待后续作业。
    CHECK_ARGUMENT(device == LLAISYS_DEVICE_CPU, "Qwen2Model: V1 only supports CPU device.");
    CHECK_ARGUMENT(_meta.nlayer > 0, "Qwen2Model: nlayer must be positive.");
    CHECK_ARGUMENT(_meta.nh > 0 && _meta.nkvh > 0 && _meta.dh > 0,
                   "Qwen2Model: head config (nh/nkvh/dh) must be positive.");
    CHECK_ARGUMENT(_meta.nh % _meta.nkvh == 0,
                   "Qwen2Model: nh must be a positive multiple of nkvh (GQA).");
    CHECK_ARGUMENT(_meta.hs == _meta.nh * _meta.dh,
                   "Qwen2Model: hs must equal nh * dh.");
    CHECK_ARGUMENT(_meta.maxseq > 0, "Qwen2Model: maxseq must be positive.");

    allocate_weights();

    // KV-Cache：每层一对 (maxseq, nkvh, dh) 的 k/v buffer，跨 infer 调用复用。
    _k_cache.resize(_meta.nlayer);
    _v_cache.resize(_meta.nlayer);
    for (size_t i = 0; i < _meta.nlayer; i++) {
        _k_cache[i] = Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device, _device_id);
        _v_cache[i] = Tensor::create({_meta.maxseq, _meta.nkvh, _meta.dh}, _meta.dtype, _device, _device_id);
    }
}

Qwen2Model::~Qwen2Model() {
    release_weights();
    // _k_cache / _v_cache 为 vector<tensor_t>，shared_ptr 自动释放。
}

// ============================================================================
// 权重张量分配 / 释放
// ============================================================================

void Qwen2Model::allocate_weights() {
    // Fix CR#L51(低危5): 中途异常时释放已分配权重，避免泄漏（构造函数异常析构不执行）。
    try {
    llaisysDataType_t dt = _meta.dtype;

    // 单个权重张量：new LlaisysTensor{tensor_t}，由 release_weights 以 delete 释放。
    auto mk = [&](const std::vector<size_t> &shape) -> llaisysTensor_t {
        return new LlaisysTensor{Tensor::create(shape, dt, _device, _device_id)};
    };
    // per-layer 权重数组：nlayer 个同 shape 张量，连续存放于 new[] 数组。
    auto mk_arr = [&](const std::vector<size_t> &shape) -> llaisysTensor_t * {
        llaisysTensor_t *arr = new llaisysTensor_t[_meta.nlayer];
        for (size_t i = 0; i < _meta.nlayer; i++) {
            arr[i] = mk(shape);
        }
        return arr;
    };

    const size_t hs = _meta.hs, nh = _meta.nh, nkvh = _meta.nkvh, dh = _meta.dh, di = _meta.di, voc = _meta.voc;
    // 权重 shape 与 safetensors 中 Qwen2 权重一致（weight 均为 (out_features, in_features)，
    // 与 linear 算子 xW^T 布局匹配，无需转置）。
    _weights.in_embed    = mk({voc, hs});     // model.embed_tokens.weight
    _weights.out_embed   = mk({voc, hs});     // lm_head.weight
    _weights.out_norm_w  = mk({hs});          // model.norm.weight
    _weights.attn_norm_w = mk_arr({hs});                  // input_layernorm.weight
    _weights.attn_q_w    = mk_arr({nh * dh, hs});         // self_attn.q_proj.weight
    _weights.attn_q_b    = mk_arr({nh * dh});             // self_attn.q_proj.bias
    _weights.attn_k_w    = mk_arr({nkvh * dh, hs});       // self_attn.k_proj.weight
    _weights.attn_k_b    = mk_arr({nkvh * dh});           // self_attn.k_proj.bias
    _weights.attn_v_w    = mk_arr({nkvh * dh, hs});       // self_attn.v_proj.weight
    _weights.attn_v_b    = mk_arr({nkvh * dh});           // self_attn.v_proj.bias
    _weights.attn_o_w    = mk_arr({hs, nh * dh});         // self_attn.o_proj.weight（无 bias）
    _weights.mlp_norm_w  = mk_arr({hs});                  // post_attention_layernorm.weight
    _weights.mlp_gate_w  = mk_arr({di, hs});              // mlp.gate_proj.weight（无 bias）
    _weights.mlp_up_w    = mk_arr({di, hs});              // mlp.up_proj.weight（无 bias）
    _weights.mlp_down_w  = mk_arr({hs, di});              // mlp.down_proj.weight（无 bias）
    } catch (...) {
        release_weights();  // 释放已分配部分（nullptr 安全）
        throw;  // 重新抛出，交由 C API 层 try/catch 兜底
    }
}

void Qwen2Model::release_weights() {
    // delete nullptr 安全；数组先逐元素 delete 再 delete[]。
    delete _weights.in_embed;
    delete _weights.out_embed;
    delete _weights.out_norm_w;
    auto del_arr = [&](llaisysTensor_t *arr) {
        if (arr) {
            for (size_t i = 0; i < _meta.nlayer; i++) {
                delete arr[i];
            }
            delete[] arr;
        }
    };
    del_arr(_weights.attn_norm_w);
    del_arr(_weights.attn_q_w);
    del_arr(_weights.attn_q_b);
    del_arr(_weights.attn_k_w);
    del_arr(_weights.attn_k_b);
    del_arr(_weights.attn_v_w);
    del_arr(_weights.attn_v_b);
    del_arr(_weights.attn_o_w);
    del_arr(_weights.mlp_norm_w);
    del_arr(_weights.mlp_gate_w);
    del_arr(_weights.mlp_up_w);
    del_arr(_weights.mlp_down_w);
}

// ============================================================================
// KV-Cache 重置
// ============================================================================

void Qwen2Model::reset() {
    _cur_len = 0;
}

// ============================================================================
// 单层 Transformer 前向
// ============================================================================

void Qwen2Model::forward_layer(size_t layer, tensor_t &hidden, const tensor_t &pos_ids, size_t ntoken) {
    const size_t hs = _meta.hs, nh = _meta.nh, nkvh = _meta.nkvh, dh = _meta.dh, di = _meta.di;
    // attention 缩放：1/sqrt(head_dim)，对齐 Qwen2/HF 标准缩放。
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

    // 取本层权重句柄（LlaisysTensor -> 内部 tensor_t）。
    tensor_t w_attn_norm = _weights.attn_norm_w[layer]->tensor;
    tensor_t w_q = _weights.attn_q_w[layer]->tensor;
    tensor_t b_q = _weights.attn_q_b[layer]->tensor;
    tensor_t w_k = _weights.attn_k_w[layer]->tensor;
    tensor_t b_k = _weights.attn_k_b[layer]->tensor;
    tensor_t w_v = _weights.attn_v_w[layer]->tensor;
    tensor_t b_v = _weights.attn_v_b[layer]->tensor;
    tensor_t w_o = _weights.attn_o_w[layer]->tensor;
    tensor_t w_mlp_norm = _weights.mlp_norm_w[layer]->tensor;
    tensor_t w_gate = _weights.mlp_gate_w[layer]->tensor;
    tensor_t w_up = _weights.mlp_up_w[layer]->tensor;
    tensor_t w_down = _weights.mlp_down_w[layer]->tensor;

    // ---- Self-Attention ----
    // input_layernorm：hidden (ntoken, hs) -> normed (ntoken, hs)
    auto normed = Tensor::create({ntoken, hs}, _meta.dtype, _device, _device_id);
    rms_norm(normed, hidden, w_attn_norm, _meta.epsilon);

    // q/k/v 投影（Qwen2 的 q/k/v 带 bias）-> (ntoken, nh*dh) / (ntoken, nkvh*dh)
    auto q = Tensor::create({ntoken, nh * dh}, _meta.dtype, _device, _device_id);
    auto k = Tensor::create({ntoken, nkvh * dh}, _meta.dtype, _device, _device_id);
    auto v = Tensor::create({ntoken, nkvh * dh}, _meta.dtype, _device, _device_id);
    linear(q, normed, w_q, b_q);
    linear(k, normed, w_k, b_k);
    linear(v, normed, w_v, b_v);

    // reshape 2D -> 3D：(seq, nhead, head_dim)；contiguous 张量 reshape 不发生拷贝。
    auto q3 = q->reshape({ntoken, nh, dh});
    auto k3 = k->reshape({ntoken, nkvh, dh});
    auto v3 = v->reshape({ntoken, nkvh, dh});

    // RoPE 仅作用于 q、k（v 不旋转）；算子支持 out==in 就地旋转。
    rope(q3, q3, pos_ids, _meta.theta);
    rope(k3, k3, pos_ids, _meta.theta);

    // KV-Cache 追加：把当前 k3/v3 拷入 cache[cur_len : cur_len+ntoken]。
    const size_t kv_off = _cur_len;
    auto k_cache_view = _k_cache[layer]->slice(0, kv_off, kv_off + ntoken);
    auto v_cache_view = _v_cache[layer]->slice(0, kv_off, kv_off + ntoken);
    k_cache_view->load(k3->data());
    v_cache_view->load(v3->data());

    // attention 使用全部历史 k/v [0 : cur_len+ntoken]（slice 得到 contiguous view）。
    const size_t kvlen = _cur_len + ntoken;
    auto k_all = _k_cache[layer]->slice(0, 0, kvlen);
    auto v_all = _v_cache[layer]->slice(0, 0, kvlen);

    // GQA 因果自注意力 -> attn_val (ntoken, nh, dh)
    auto attn_val = Tensor::create({ntoken, nh, dh}, _meta.dtype, _device, _device_id);
    self_attention(attn_val, q3, k_all, v_all, scale);

    // reshape 3D -> 2D 后 o_proj（无 bias）-> o (ntoken, hs)
    auto attn_val_2d = attn_val->reshape({ntoken, nh * dh});
    auto o = Tensor::create({ntoken, hs}, _meta.dtype, _device, _device_id);
    linear(o, attn_val_2d, w_o, nullptr);

    // 残差：hidden = hidden + o
    auto hidden_attn = Tensor::create({ntoken, hs}, _meta.dtype, _device, _device_id);
    add(hidden_attn, hidden, o);
    hidden = hidden_attn;

    // ---- MLP ----
    // post_attention_layernorm
    auto normed2 = Tensor::create({ntoken, hs}, _meta.dtype, _device, _device_id);
    rms_norm(normed2, hidden, w_mlp_norm, _meta.epsilon);

    // gate / up 投影（无 bias）-> (ntoken, di)
    auto gate = Tensor::create({ntoken, di}, _meta.dtype, _device, _device_id);
    auto up = Tensor::create({ntoken, di}, _meta.dtype, _device, _device_id);
    linear(gate, normed2, w_gate, nullptr);
    linear(up, normed2, w_up, nullptr);

    // SwiGLU：silu(gate) * up -> act (ntoken, di)
    auto act = Tensor::create({ntoken, di}, _meta.dtype, _device, _device_id);
    swiglu(act, gate, up);

    // down 投影（无 bias）-> down (ntoken, hs)
    auto down = Tensor::create({ntoken, hs}, _meta.dtype, _device, _device_id);
    linear(down, act, w_down, nullptr);

    // 残差：hidden = hidden + down
    auto hidden_mlp = Tensor::create({ntoken, hs}, _meta.dtype, _device, _device_id);
    add(hidden_mlp, hidden, down);
    hidden = hidden_mlp;
}

// ============================================================================
// 推理：前向 + argmax
// ============================================================================

int64_t Qwen2Model::infer(const int64_t *token_ids, size_t ntoken) {
    CHECK_ARGUMENT(ntoken > 0, "Qwen2Model: ntoken must be positive.");
    CHECK_ARGUMENT(token_ids != nullptr, "Qwen2Model: token_ids must not be null.");
    CHECK_ARGUMENT(_cur_len + ntoken <= _meta.maxseq,
                   "Qwen2Model: KV-Cache overflow (cur_len + ntoken > maxseq).");

    const size_t hs = _meta.hs, voc = _meta.voc;

    // 1. token_ids -> tensor (ntoken,) int64
    auto tokens = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
    tokens->load(token_ids);

    // 2. token embedding -> hidden (ntoken, hs)
    auto hidden = Tensor::create({ntoken, hs}, _meta.dtype, _device, _device_id);
    embedding(hidden, tokens, _weights.in_embed->tensor);

    // 3. 位置 id (ntoken,) int64 = [cur_len, cur_len+1, ..., cur_len+ntoken-1]
    auto pos_ids = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
    int64_t *pos_ptr = reinterpret_cast<int64_t *>(pos_ids->data());
    for (size_t i = 0; i < ntoken; i++) {
        pos_ptr[i] = static_cast<int64_t>(_cur_len + i);
    }

    // 4. 逐层前向
    for (size_t i = 0; i < _meta.nlayer; i++) {
        forward_layer(i, hidden, pos_ids, ntoken);
    }

    // 5. final norm -> lm_head -> logits (ntoken, voc)
    auto final_normed = Tensor::create({ntoken, hs}, _meta.dtype, _device, _device_id);
    rms_norm(final_normed, hidden, _weights.out_norm_w->tensor, _meta.epsilon);
    auto logits = Tensor::create({ntoken, voc}, _meta.dtype, _device, _device_id);
    linear(logits, final_normed, _weights.out_embed->tensor, nullptr);

    // 6. 取最后一个 token 的 logits 做 argmax（贪婪解码）
    auto last = logits->slice(0, ntoken - 1, ntoken);  // (1, voc)
    auto last_1d = last->reshape({voc});                // (voc,)，argmax 要求 1D
    auto max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device, _device_id);
    auto max_val = Tensor::create({1}, _meta.dtype, _device, _device_id);
    argmax(max_idx, max_val, last_1d);

    _cur_len += ntoken;

    return *reinterpret_cast<const int64_t *>(max_idx->data());
}

} // namespace llaisys::models
