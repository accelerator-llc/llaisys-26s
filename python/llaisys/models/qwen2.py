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


# dtype -> 字节数，用于加载前校验原始字节与张量容量一致。
# 含 F8（8-bit float，1 字节）；未知 dtype 在 _tensor_capacity_bytes 抛错。
_DSIZE = {
    DataType.BYTE: 1, DataType.BOOL: 1, DataType.I8: 1, DataType.I16: 2,
    DataType.I32: 4, DataType.I64: 8, DataType.U8: 1, DataType.U16: 2,
    DataType.U32: 4, DataType.U64: 8, DataType.F8: 1, DataType.F16: 2,
    DataType.F32: 4, DataType.F64: 8, DataType.BF16: 2,
}


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
        # KV-Cache 容量取 min(max_position_embeddings, 4096)，
        # 4096 为内存预算折中（28 层×2×4096×256×2B≈117MB）；与 sliding_window 无关，
        # prompt+生成超过此值会触发 C++ 侧溢出校验（返回错误而非越界）。
        meta.maxseq = min(config.get("max_position_embeddings", 4096), 4096)

        # 创建模型（仅 CPU 单设备，device_ids[0]=0）。
        device_ids = (ctypes.c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta), device, device_ids, 1)
        # Create 失败（非法参数或 C++ 异常）返回 nullptr。
        if not self._model:
            raise RuntimeError("llaisysQwen2ModelCreate failed (invalid meta or C++ exception)")
        self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model).contents
        self._end_token = int(meta.end_token)

        # 逐文件加载权重；统计已加载数与 safetensors 张量总数比对（实际 339 个：
        # 3 个全局 + 12 类 per-layer × 28 层，含 84 个 bias）。
        loaded_tensors = 0
        total_tensors = 0
        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="numpy", device="cpu")
            # numpy backend 无法 get_tensor bf16（raise TypeError），
            # 手动解析 safetensors 文件取每个权重的原始 bf16 字节再 tensorLoad。
            raw_map = self._read_safetensors(file)
            total_tensors += len(raw_map)
            for name_ in data_.keys():
                ## TODO: load the model weights
                if not self._load_weight(name_, raw_map[name_]):
                    raise ValueError(f"未知权重名，无法映射: {name_}")
                loaded_tensors += 1
        # 校验已加载张量数与 safetensors 总数一致，漏载即报错。
        if loaded_tensors != total_tensors:
            raise ValueError(f"loaded {loaded_tensors} != total {total_tensors}")

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        # TODO: Implement generate function

        # 空 inputs 前置校验，避免 ntoken=0 触发 C++ 异常路径。
        if not inputs:
            raise ValueError("inputs must not be empty")
        # 建模 HF 语义：贪婪模式下 top_k/top_p/temperature 存在但不生效（HF generate
        # 默认 do_sample=False 即贪婪）；本实现仅支持贪婪 argmax，非贪婪参数忽略。
        if max_new_tokens is None:
            max_new_tokens = 128

        LIB_LLAISYS.llaisysQwen2ModelReset(self._model)

        inputs = list(inputs)
        # prefill：整段 prompt 一次前向，返回首个生成 token。
        # Infer 返回 -1 表 C++ 异常，转 Python 异常。
        token_ids = (ctypes.c_int64 * len(inputs))(*inputs)
        next_token = int(LIB_LLAISYS.llaisysQwen2ModelInfer(
            self._model, token_ids, len(inputs)))
        if next_token < 0:
            raise RuntimeError("prefill Infer failed (C++ exception)")
        outputs = inputs + [next_token]

        # decode：逐 token 增量前向；命中 end_token 或达 max_new_tokens 停止。
        for _ in range(max_new_tokens - 1):
            if outputs[-1] == self._end_token:
                break
            token_ids = (ctypes.c_int64 * 1)(outputs[-1])
            next_token = int(LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model, token_ids, 1))
            if next_token < 0:
                raise RuntimeError("decode Infer failed (C++ exception)")
            outputs.append(next_token)

        return outputs

    def __del__(self):
        if getattr(self, "_model", None):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None
            self._weights = None  # 置空避免 use-after-free。

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
            # header_len 合理性校验，防止损坏文件产生异常切片。
            if header_len == 0 or header_len > (1 << 30):
                raise ValueError(f"invalid safetensors header_len: {header_len}")
            header = json.loads(f.read(header_len))
            data_start = 8 + header_len
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            file_size = len(mm)
            raw_map = {}
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                start, end = info["data_offsets"]
                # offsets 越界校验。
                if start < 0 or end < start or data_start + end > file_size:
                    raise ValueError(f"weight '{name}' offsets out of range: [{start},{end})")
                raw_map[name] = mm[data_start + start : data_start + end]
            return raw_map

    def _match_weight_handle(self, name):
        """按 safetensors 权重名匹配到模型权重句柄，未知返回 None。"""
        w = self._weights
        if name == "model.embed_tokens.weight":
            return w.in_embed
        if name == "model.norm.weight":
            return w.out_norm_w
        if name == "lm_head.weight":
            return w.out_embed
        if name.startswith("model.layers."):
            parts = name.split(".")
            layer = int(parts[2])
            suffix = ".".join(parts[3:])
            mapping = {
                "input_layernorm.weight": w.attn_norm_w,
                "self_attn.q_proj.weight": w.attn_q_w,
                "self_attn.q_proj.bias": w.attn_q_b,
                "self_attn.k_proj.weight": w.attn_k_w,
                "self_attn.k_proj.bias": w.attn_k_b,
                "self_attn.v_proj.weight": w.attn_v_w,
                "self_attn.v_proj.bias": w.attn_v_b,
                "self_attn.o_proj.weight": w.attn_o_w,
                "post_attention_layernorm.weight": w.mlp_norm_w,
                "mlp.gate_proj.weight": w.mlp_gate_w,
                "mlp.up_proj.weight": w.mlp_up_w,
                "mlp.down_proj.weight": w.mlp_down_w,
            }
            arr = mapping.get(suffix)
            if arr is None:
                return None
            return arr[layer]
        return None

    @staticmethod
    def _tensor_capacity_bytes(handle):
        """计算张量容量字节数 = numel × dtype 字节数。"""
        ndim = LIB_LLAISYS.tensorGetNdim(handle)
        shape = (ctypes.c_size_t * max(ndim, 1))()
        LIB_LLAISYS.tensorGetShape(handle, shape)
        dtype = LIB_LLAISYS.tensorGetDataType(handle)
        dsize = _DSIZE.get(dtype)
        if dsize is None:
            raise ValueError(f"unsupported dtype for capacity check: {dtype}")
        numel = 1
        for i in range(ndim):
            numel *= shape[i]
        return numel * dsize

    def _load_weight(self, name, raw):
        """按权重名匹配句柄并 tensorLoad；未知权重返回 False，字节长度不符抛异常。"""
        handle = self._match_weight_handle(name)
        if handle is None:
            return False
        # 校验原始字节长度与张量容量一致，避免映射错配静默加载或越界。
        expected = self._tensor_capacity_bytes(handle)
        if len(raw) != expected:
            raise ValueError(
                f"weight '{name}' bytes {len(raw)} != tensor capacity {expected}")
        LIB_LLAISYS.tensorLoad(handle, raw)
        return True
