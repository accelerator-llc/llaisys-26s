from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
from ..libllaisys import DataType
from ..libllaisys import LlaisysQwen2Meta

from pathlib import Path
import ctypes
import json
import mmap
import struct
import safetensors


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        # TODO: Implement model constructor

        model_path = Path(model_path)

        # 读取 config.json 构造模型超参（与 DeepSeek-R1-Distill-Qwen-1.5B 对齐）。
        config = json.loads((model_path / "config.json").read_text())
        hs = config["hidden_size"]
        nh = config["num_attention_heads"]
        meta = LlaisysQwen2Meta()
        meta.dtype = DataType.BF16
        meta.nlayer = config["num_hidden_layers"]
        meta.hs = hs
        meta.nh = nh
        meta.nkvh = config["num_key_value_heads"]
        meta.dh = hs // nh  # head_dim = hidden_size / num_attention_heads
        meta.di = config["intermediate_size"]
        meta.voc = config["vocab_size"]
        meta.epsilon = config["rms_norm_eps"]
        meta.theta = config["rope_theta"]
        meta.end_token = config["eos_token_id"]
        # KV-Cache 容量：取 sliding_window（4096），兼顾内存与常见 prompt 长度。
        meta.maxseq = config.get("sliding_window", 4096)

        # 创建模型（V1 仅 CPU 单设备，device_ids[0]=0）。
        device_ids = (ctypes.c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta), device, device_ids, 1)
        self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model).contents
        self._end_token = int(meta.end_token)

        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="numpy", device="cpu")
            # numpy backend 无法 get_tensor bf16（raise TypeError），
            # 为了不修改桩代码，手动解析 safetensors 文件取每个权重的原始 bf16 字节再 tensorLoad。
            raw_map = self._read_safetensors(file)
            for name_ in data_.keys():
                ## TODO: load the model weights
                self._load_weight(name_, raw_map[name_])

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        # TODO: Implement generate function

        if max_new_tokens is None:
            max_new_tokens = 128

        # V1 仅实现 argmax 贪婪解码；HF model.generate 未设 do_sample=True 时亦走贪婪，
        # 故 top_k/top_p/temperature 参数接收但不影响结果。
        LIB_LLAISYS.llaisysQwen2ModelReset(self._model)

        inputs = list(inputs)
        # prefill：整段 prompt 一次前向，返回首个生成 token。
        token_ids = (ctypes.c_int64 * len(inputs))(*inputs)
        next_token = int(LIB_LLAISYS.llaisysQwen2ModelInfer(
            self._model, token_ids, len(inputs)))
        outputs = inputs + [next_token]

        # decode：逐 token 增量前向；命中 end_token 或达 max_new_tokens 停止。
        for _ in range(max_new_tokens - 1):
            if outputs[-1] == self._end_token:
                break
            token_ids = (ctypes.c_int64 * 1)(outputs[-1])
            next_token = int(LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model, token_ids, 1))
            outputs.append(next_token)

        return outputs

    def __del__(self):
        if getattr(self, "_model", None):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    # ------------------------------------------------------------------
    # 权重加载辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _read_safetensors(file):
        """手动解析 safetensors 文件，返回 {name: 原始字节}。

        safetensors 格式：[8B header_len(u64 LE)][header JSON][data...]。
        header 中每个 tensor 含 data_offsets [start, end]（相对数据区起始）。
        用 mmap 按需读取，避免一次性载入整个文件。
        """
        with open(file, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
            data_start = 8 + header_len
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            raw_map = {}
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                start, end = info["data_offsets"]
                raw_map[name] = mm[data_start + start : data_start + end]
            return raw_map

    def _load_weight(self, name, raw):
        """按 safetensors 权重名匹配到模型权重句柄并 tensorLoad。

        权重命名与 qwen2.h 的 LlaisysQwen2Weights 字段一一对应：
        model.embed_tokens.weight -> in_embed, lm_head.weight -> out_embed,
        model.norm.weight -> out_norm_w, model.layers.{i}.* -> per-layer 数组。
        """
        w = self._weights
        lib = LIB_LLAISYS
        if name == "model.embed_tokens.weight":
            lib.tensorLoad(w.in_embed, raw)
        elif name == "model.norm.weight":
            lib.tensorLoad(w.out_norm_w, raw)
        elif name == "lm_head.weight":
            lib.tensorLoad(w.out_embed, raw)
        elif name.startswith("model.layers."):
            parts = name.split(".")
            layer = int(parts[2])
            suffix = ".".join(parts[3:])
            if suffix == "input_layernorm.weight":
                lib.tensorLoad(w.attn_norm_w[layer], raw)
            elif suffix == "self_attn.q_proj.weight":
                lib.tensorLoad(w.attn_q_w[layer], raw)
            elif suffix == "self_attn.q_proj.bias":
                lib.tensorLoad(w.attn_q_b[layer], raw)
            elif suffix == "self_attn.k_proj.weight":
                lib.tensorLoad(w.attn_k_w[layer], raw)
            elif suffix == "self_attn.k_proj.bias":
                lib.tensorLoad(w.attn_k_b[layer], raw)
            elif suffix == "self_attn.v_proj.weight":
                lib.tensorLoad(w.attn_v_w[layer], raw)
            elif suffix == "self_attn.v_proj.bias":
                lib.tensorLoad(w.attn_v_b[layer], raw)
            elif suffix == "self_attn.o_proj.weight":
                lib.tensorLoad(w.attn_o_w[layer], raw)
            elif suffix == "post_attention_layernorm.weight":
                lib.tensorLoad(w.mlp_norm_w[layer], raw)
            elif suffix == "mlp.gate_proj.weight":
                lib.tensorLoad(w.mlp_gate_w[layer], raw)
            elif suffix == "mlp.up_proj.weight":
                lib.tensorLoad(w.mlp_up_w[layer], raw)
            elif suffix == "mlp.down_proj.weight":
                lib.tensorLoad(w.mlp_down_w[layer], raw)
            # 其余 per-layer 权重（若有）跳过
        # 其余非 model.layers 权重（若有）跳过
