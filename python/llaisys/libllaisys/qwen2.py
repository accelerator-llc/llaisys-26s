from ctypes import POINTER, Structure, c_int, c_int64, c_size_t, c_float
from .llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from .tensor import llaisysTensor_t


# 不透明模型句柄（对应 C 侧 struct LlaisysQwen2Model; 前置声明）。
class LlaisysQwen2Model(Structure):
    pass


# 模型超参（对应 C 侧 struct LlaisysQwen2Meta）。
# 字段顺序/类型须与 qwen2.h 完全一致，ctypes 按自然对齐自动填充 padding。
class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),   # int (enum)
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]


# 权重句柄集合（对应 C 侧 struct LlaisysQwen2Weights）。
# 单个权重为 llaisysTensor_t(c_void_p)；per-layer 权重为指向 c_void_p 数组的指针。
class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", POINTER(llaisysTensor_t)),
        ("attn_q_w", POINTER(llaisysTensor_t)),
        ("attn_q_b", POINTER(llaisysTensor_t)),
        ("attn_k_w", POINTER(llaisysTensor_t)),
        ("attn_k_b", POINTER(llaisysTensor_t)),
        ("attn_v_w", POINTER(llaisysTensor_t)),
        ("attn_v_b", POINTER(llaisysTensor_t)),
        ("attn_o_w", POINTER(llaisysTensor_t)),
        ("mlp_norm_w", POINTER(llaisysTensor_t)),
        ("mlp_gate_w", POINTER(llaisysTensor_t)),
        ("mlp_up_w", POINTER(llaisysTensor_t)),
        ("mlp_down_w", POINTER(llaisysTensor_t)),
    ]


def load_qwen2(lib):
    # LlaisysQwen2Model *llaisysQwen2ModelCreate(meta, device, device_ids, ndevice)
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),  # meta
        llaisysDeviceType_t,         # device
        POINTER(c_int),              # device_ids
        c_int,                       # ndevice
    ]
    lib.llaisysQwen2ModelCreate.restype = POINTER(LlaisysQwen2Model)

    # void llaisysQwen2ModelDestroy(model)
    lib.llaisysQwen2ModelDestroy.argtypes = [POINTER(LlaisysQwen2Model)]
    lib.llaisysQwen2ModelDestroy.restype = None

    # LlaisysQwen2Weights *llaisysQwen2ModelWeights(model)
    lib.llaisysQwen2ModelWeights.argtypes = [POINTER(LlaisysQwen2Model)]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)

    # int64_t llaisysQwen2ModelInfer(model, token_ids, ntoken)
    lib.llaisysQwen2ModelInfer.argtypes = [
        POINTER(LlaisysQwen2Model),
        POINTER(c_int64),  # token_ids
        c_size_t,           # ntoken
    ]
    lib.llaisysQwen2ModelInfer.restype = c_int64

    # void llaisysQwen2ModelReset(model)
    lib.llaisysQwen2ModelReset.argtypes = [POINTER(LlaisysQwen2Model)]
    lib.llaisysQwen2ModelReset.restype = None
