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
    // V1 支持 CPU 与 NVIDIA 单设备；多设备留待后续作业。
    // 设备白名单按编译期宏裁剪：未开启 ENABLE_NVIDIA_API 时仅允许 CPU；
    // 第二平台（4.9）在此追加分支即可扩展，不泄漏平台专属逻辑到调用方。
    bool device_supported = (device == LLAISYS_DEVICE_CPU);
#ifdef ENABLE_NVIDIA_API
    device_supported = device_supported || (device == LLAISYS_DEVICE_NVIDIA);
#endif
    CHECK_ARGUMENT(device_supported, "Qwen2Model: unsupported device type.");
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
        // Fix CR#L87(任务3b): 值初始化为 nullptr，mk() 中途抛异常时释放已分配元素与数组外壳。
        llaisysTensor_t *arr = new llaisysTensor_t[_meta.nlayer]();
        try {
            for (size_t i = 0; i < _meta.nlayer; i++) {
                arr[i] = mk(shape);
            }
        } catch (...) {
            for (size_t i = 0; i < _meta.nlayer; i++) {
                delete arr[i];  // nullptr 安全
            }
            delete[] arr;
            throw;
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
// 张量池：取 buffer 的 ntoken 切片，容量不足时扩容（只增不减）
// ============================================================================

tensor_t Qwen2Model::scratch(tensor_t &pool, llaisysDataType_t dt,
                             const std::vector<size_t> &shape) {
    // shape[0]=ntoken（随 prefill/decode 变化），其余为静态维度；
    // pool 按 (cap, rest...) 预分配，cap >= ntoken 时直接 slice(0,0,ntoken) 复用。
    // 同流内复用安全：stream ordering 保证前 kernel 完成后后 kernel 才执行
    // （对齐 PyTorch CUDACachingAllocator 同流复用语义）。
    size_t rest = 1;
    for (size_t i = 1; i < shape.size(); i++) {
        rest *= shape[i];
    }
    size_t need = shape[0];
    size_t have = pool ? (rest > 0 ? pool->numel() / rest : 0) : 0;
    if (have < need) {
        pool = Tensor::create(shape, dt, _device, _device_id);
    }
    return pool->slice(0, 0, need);
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
    auto normed = scratch(_pool.normed, _meta.dtype, {ntoken, hs});
    rms_norm(normed, hidden, w_attn_norm, _meta.epsilon);

    // q/k/v 投影（Qwen2 的 q/k/v 带 bias）-> (ntoken, nh*dh) / (ntoken, nkvh*dh)
    auto q = scratch(_pool.q, _meta.dtype, {ntoken, nh * dh});
    auto k = scratch(_pool.k, _meta.dtype, {ntoken, nkvh * dh});
    auto v = scratch(_pool.v, _meta.dtype, {ntoken, nkvh * dh});
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
    // Tensor::load 假定 src 为 host 指针，不能用于设备间拷贝；
    // 按 dst 设备类型选 H2H(CPU)/D2D(设备)，setDevice 到数据所在设备做拷贝
    // （与 Tensor::debug 的设备侧 memcpy 模式一致，src/dst 同设备）。
    auto copy_tensor_data = [](const tensor_t &dst, const tensor_t &src) {
        llaisysMemcpyKind_t kind = (dst->deviceType() == LLAISYS_DEVICE_CPU)
                                       ? LLAISYS_MEMCPY_H2H
                                       : LLAISYS_MEMCPY_D2D;
        core::context().setDevice(dst->deviceType(), dst->deviceId());
        core::context().runtime().api()->memcpy_sync(
            dst->data(), src->data(), src->numel() * src->elementSize(), kind);
    };
    const size_t kv_off = _cur_len;
    auto k_cache_view = _k_cache[layer]->slice(0, kv_off, kv_off + ntoken);
    auto v_cache_view = _v_cache[layer]->slice(0, kv_off, kv_off + ntoken);
    copy_tensor_data(k_cache_view, k3);
    copy_tensor_data(v_cache_view, v3);

    // attention 使用全部历史 k/v [0 : cur_len+ntoken]（slice 得到 contiguous view）。
    const size_t kvlen = _cur_len + ntoken;
    auto k_all = _k_cache[layer]->slice(0, 0, kvlen);
    auto v_all = _v_cache[layer]->slice(0, 0, kvlen);

    // GQA 因果自注意力 -> attn_val (ntoken, nh, dh)
    auto attn_val = scratch(_pool.attn_val, _meta.dtype, {ntoken, nh, dh});
    self_attention(attn_val, q3, k_all, v_all, scale);

    // reshape 3D -> 2D 后 o_proj（无 bias）-> o (ntoken, hs)
    auto attn_val_2d = attn_val->reshape({ntoken, nh * dh});
    auto o = scratch(_pool.o, _meta.dtype, {ntoken, hs});
    linear(o, attn_val_2d, w_o, nullptr);

    // 残差：hidden = hidden + o
    auto hidden_attn = scratch(_pool.hidden_attn, _meta.dtype, {ntoken, hs});
    add(hidden_attn, hidden, o);
    hidden = hidden_attn;

    // ---- MLP ----
    // post_attention_layernorm
    auto normed2 = scratch(_pool.normed2, _meta.dtype, {ntoken, hs});
    rms_norm(normed2, hidden, w_mlp_norm, _meta.epsilon);

    // gate / up 投影（无 bias）-> (ntoken, di)
    auto gate = scratch(_pool.gate, _meta.dtype, {ntoken, di});
    auto up = scratch(_pool.up, _meta.dtype, {ntoken, di});
    linear(gate, normed2, w_gate, nullptr);
    linear(up, normed2, w_up, nullptr);

    // SwiGLU：silu(gate) * up -> act (ntoken, di)
    auto act = scratch(_pool.act, _meta.dtype, {ntoken, di});
    swiglu(act, gate, up);

    // down 投影（无 bias）-> down (ntoken, hs)
    auto down = scratch(_pool.down, _meta.dtype, {ntoken, hs});
    linear(down, act, w_down, nullptr);

    // 残差：hidden = hidden + down
    auto hidden_mlp = scratch(_pool.hidden_mlp, _meta.dtype, {ntoken, hs});
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
    auto tokens = scratch(_pool.tokens, LLAISYS_DTYPE_I64, {ntoken});
    tokens->load(token_ids);

    // 2. token embedding -> hidden (ntoken, hs)
    auto hidden = scratch(_pool.hidden, _meta.dtype, {ntoken, hs});
    embedding(hidden, tokens, _weights.in_embed->tensor);

    // 3. 位置 id (ntoken,) int64 = [cur_len, cur_len+1, ..., cur_len+ntoken-1]
    //    host 侧填充后经 load 按设备类型 H2H(CPU)/H2D(NVIDIA) 拷入张量；
    //    不可直接写 data()：设备张量返回显存指针，CPU 不可解引用。
    auto pos_ids = scratch(_pool.pos_ids, LLAISYS_DTYPE_I64, {ntoken});
    std::vector<int64_t> pos_host(ntoken);
    for (size_t i = 0; i < ntoken; i++) {
        pos_host[i] = static_cast<int64_t>(_cur_len + i);
    }
    pos_ids->load(pos_host.data());

    // 4. 逐层前向
    for (size_t i = 0; i < _meta.nlayer; i++) {
        forward_layer(i, hidden, pos_ids, ntoken);
    }

    // 5. final norm -> lm_head 仅计算最后一行（V3: lm_head 仅计算末行）。
    //    工业标准做法（llama.cpp 默认 logits_all=false，vLLM 同），linear 按行独立，
    //    末行累加顺序与全量计算 bit-exact 一致，argmax 不变；省 prefill lm_head 计算。
    auto final_normed = scratch(_pool.final_normed, _meta.dtype, {ntoken, hs});
    rms_norm(final_normed, hidden, _weights.out_norm_w->tensor, _meta.epsilon);
    auto last_hidden = final_normed->slice(0, ntoken - 1, ntoken);  // (1, hs)
    auto logits = scratch(_pool.logits, _meta.dtype, {1, voc});
    linear(logits, last_hidden, _weights.out_embed->tensor, nullptr);

    // 6. argmax（logits 已是 (1, voc)，reshape 为 1D）
    auto last_1d = logits->reshape({voc});  // (voc,)，argmax 要求 1D
    auto max_idx = scratch(_pool.max_idx, LLAISYS_DTYPE_I64, {1});
    auto max_val = scratch(_pool.max_val, _meta.dtype, {1});
    argmax(max_idx, max_val, last_1d);

    _cur_len += ntoken;

    // argmax 结果位于显存时需 D2H 取回（CPU 不可解引用显存指针）；CPU 路径直接读。
    // 模仿 Tensor::debug 的 D2H 模式：setDevice 到设备侧，用设备 runtime 拷贝。
    // 仅 8 字节控制信号过桥，hidden/KV-cache/logits 全程驻留显存（数据全闭环）。
    int64_t next_token = 0;
    if (max_idx->deviceType() == LLAISYS_DEVICE_CPU) {
        next_token = *reinterpret_cast<const int64_t *>(max_idx->data());
    } else {
        core::context().setDevice(max_idx->deviceType(), max_idx->deviceId());
        core::context().runtime().api()->memcpy_sync(
            &next_token, max_idx->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
    }
    return next_token;
}

} // namespace llaisys::models
