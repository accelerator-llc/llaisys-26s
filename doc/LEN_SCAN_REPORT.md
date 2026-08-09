# 长度扫描实验报告：per-token vs 生成长度 + 逐 token 一致性验证

> 2026-08-10。目的：测不同生成长度下 llaisys 与 PyTorch(HF) 的性能曲线与一致性边界。实验脚本为独立程序（不修改仓库代码），存放于 `scripts/bench/`。

## 1. 实验目的

1. 建立 llaisys 推理的 **per-token 耗时 vs 生成长度** 曲线（验证 attention O(kvlen) 的预期增长）
2. 建立 **加速比（HF/llaisys）vs 生成长度** 曲线（长序列下优势是否缩小）
3. **逐 token 一致性验证**（强制生成下 llaisys 与 HF 是否逐位一致）

## 2. 实验设计

### 2.1 平台与参数

- 平台：5090（NVIDIA，CUDA 12.8）、C500（沐曦，MACA 3.2.1.10）
- 模型：DeepSeek-R1-Distill-Qwen-1.5B（bf16）
- 长度：64 / 256 / 1024 / 2048（生成 tokens）
- prompt："Who are you?"（chat template 展开；5090 编码 9 token，C500 编码 10 token——transformers 版本差异）
- 每长度 3 次（llaisys/HF 交替测）

### 2.2 方法

- **强制生成不截断**（对应工业标准 ignore_eos，与 vLLM `SamplingParams(ignore_eos=True)` 语义一致）：
  - llaisys：绕过模型包装的 generate（其内部遇 EOS 停止，EOS 判断在 Python 层），直接调用 C 接口循环（Reset → prefill → decode 到目标长度，遇 EOS token 继续生成）
  - HF：`model.generate(..., max_new_tokens=N, eos_token_id=None)`（经 transformers 源码链验证等效忽略 EOS）
- **验证**：len_ok（实际生成数 == 目标长度）+ token_identical（llaisys 与 HF 生成序列逐位对比，报第一个不一致位置）
- 计时：generate 全流程（含 prefill + decode），不含模型加载；两边均有隐式同步点（无异步计时虚低）
- per-token 为端到端混合口径（含 prefill）——与工业 TPOT（排除首 token）的差异见局限 1

### 2.3 对照实验（分叉归因）

HF 三个 attention 实现（eager / sdpa / flash_attention_2）同一模型同长度（2048）生成，两两对比 token 序列——验证"attention 实现差异 → 分叉"是否普遍现象。

## 3. 原始数据

### 3.1 性能（3 次均值；run1 冷启动剔除——64 长度 run1 明显偏高，实测证实）

**5090**：

| 长度 | llaisys (s) | HF (s) | 加速比 | llaisys per-token (ms) |
|---|---|---|---|---|
| 64 | 0.217 | 0.856 | 3.9x | 3.4 |
| 256 | 0.911 | 3.46 | 3.8x | 3.6 |
| 1024 | 4.446 | 13.79 | 3.1x | 4.3 |
| 2048 | 11.157 | 27.56 | 2.5x | 5.4 |

**C500**：

| 长度 | llaisys (s) | HF (s) | 加速比 | llaisys per-token (ms) |
|---|---|---|---|---|
| 64 | 0.474 | 1.70 | 3.6x | 7.4 |
| 256 | 2.149 | 6.87 | 3.2x | 8.4 |
| 1024 | 13.29 | 27.57 | 2.1x | 13.0 |
| 2048 | 38.28 | 55.33 | 1.4x | 18.7 |

### 3.2 token 一致性（逐位对比）

| 平台 | 64 | 256 | 1024 | 2048 |
|---|---|---|---|---|
| 5090 | ✅ 3/3 一致 | ✅ 3/3 一致 | run1 一致；run2/3 在 **206** 分叉 | 3/3 在 **1213** 分叉 |
| C500 | ✅ 3/3 一致 | ✅ 3/3 一致 | ✅ 3/3 一致 | run1/2 在 **206**、run3 在 **1129** 分叉 |

分叉示例：206 处 llaisys=151643(EOS) vs HF=220；1213 处 llaisys=13 vs HF=382。

### 3.3 对照实验（HF 内部 attention 实现，5090，2048 长度）

| 对比 | 结果 |
|---|---|
| eager vs sdpa | 分叉于 206（151643 vs 220） |
| eager vs flash_attention_2 | 分叉于 459（24163 vs 151646） |
| sdpa vs flash_attention_2 | 分叉于 206（220 vs 151643） |

### 3.4 logits 差异量级（5090）

- **实现间 logits 差异（eager vs sdpa，同一输入序列 forward）**：max_abs=3.62、mean_abs=0.48——**O(1) 量级**（远大于舍入差）
- **分叉位置（206）EOS 竞争**：eager 在该步 argmax=EOS（151643）、top1-top2=2.875；sdpa 的 EOS 分数同为 top1（17.0）
- **实验缺陷（如实记录）**：对同一输入序列做**全量 forward 复算**（无 KV cache）得到的 logits，与 **generate 增量路径**（带 KV cache）的实际行为不完全一致——sdpa 复算 argmax=EOS 但实际 generate 输出 220——**logits 复算路径与生成路径不等价，logits 与生成行为的对应关系未可靠建立**

## 4. 结论

### 4.1 性能曲线

- **per-token 随长度近似线性上升**：5090 3.4→5.4ms（1.6x）、C500 7.4→18.7ms（2.5x）——与 self_attention O(kvlen) 的预期一致（混合口径下上升同时含 prefill 的 O(kvlen) 贡献，两者均随 kvlen 增长）；C500 增长更陡（attention 占比更高）
- **加速比随长度下降**：5090 3.9x→2.5x、C500 3.6x→1.4x——长序列下与 PyTorch（优化 attention）的差距缩小，**长序列 attention 是后续优化空间**

### 4.2 token 分叉

- **证据**：分叉是概率性的（同平台同长度 run 间位置漂移：206/1129/1213）；分叉位置是 logits 竞争边界（206 处 EOS 为 top1 竞争者；1213 处非 EOS——分叉不限于 EOS）；**HF 内部不同 attention 实现之间同样分叉（206/459）**；前 205 个 token 逐位一致是真实的
- **推断（标注证据强度）**：分叉机制 = 两个数值不完全一致的实现（llaisys vs HF，或 HF 内部 eager vs flash）在长序列 logits 竞争边界处，实现间数值差异触发 argmax 翻转。实测佐证：实现间 logits 差异达 **O(1) 量级**；EOS 在分叉位置是 top1 竞争者。**但**：logits 复算路径与 generate 路径不等价（见 3.4），"logits 差异直接导致翻转"的对应关系**未可靠实证**——该机制为合理推断而非直接证据
- **非实现缺陷**：证据充分（HF 内部实现互不一致，说明分叉是长序列普遍现象，非 llaisys 独有）
- **影响**：真实推理（EOS 停止）中长对话可能提前 EOS 停止（206 处分叉选 EOS）；"逐 token 一致"声明限定短序列（81 tokens，作业验收口径）成立，长序列为已知边界
- **不修的理由**：要消除需 logits bit 级一致（不可能——HF 自己 eager/sdpa/flash 互不一致；cuBLAS/mcblas 累加顺序本质不同）；实际影响有限（至多分叉位置附近文本微差，非错误输出）

### 4.3 与工业界做法对照（vLLM 官方 benchmark）

- 工业标准：合成随机 token prompt + `ignore_eos=True` 强制生成到 output_len + TTFT/TPOT/ITL 指标（TPOT 排除首 token）+ warmup 与多次迭代取均值/百分位——**本实验的"忽略 EOS 强制生成"与工业标准一致**；差异：本实验用真实 prompt（非随机合成）、per-token 为混合口径（含 prefill，非工业 TPOT 的 decode-only）、无正式 warmup（以剔除 run1 冷启动替代）、3 次取均值（工业 30 次取百分位）
- 工业界不将 token 一致性作为性能测试内容（本实验额外验证了它，发现了长序列分叉边界）

## 5. 局限

1. per-token 为混合口径（含 prefill）——与工业 TPOT（排除首 token）不一致，可补测 TTFT 拆分
2. prompt 为真实文本（非工业随机合成）；prompt 长度两平台差 1 token（transformers 版本差异）
3. 分叉位置观测样本有限（每长度 3 次）——概率性结论基于现有样本
4. 未测 512/4096 等中间/更长长度（曲线线性已确认，边际信息有限）
5. 计时为单进程顺序执行（llaisys 与 HF 交替），未做 ABBA 反转（同轮内顺序固定）；无正式 warmup（以剔除 run1 冷启动替代）
6. **logits 复算缺陷**：3.4 的 logits 对比用全量 forward（无 KV cache），与 generate 增量路径不等价——logits 与生成行为的对应关系未可靠建立（若要可靠测量，需用与 generate 等价的增量路径取 logits，或对 llaisys 内部加观测点——后者需改代码，超出本实验范围）

## 6. 实验复现方法

### 6.1 环境要求

- 平台：NVIDIA GPU（CUDA 12.x）或沐曦 C500（MACA + cu-bridge 环境，需 `scripts/c500/env.sh`）
- Python：含 torch、transformers、safetensors；仓库 `python/` 在 PYTHONPATH（llaisys 包）
- 模型：DeepSeek-R1-Distill-Qwen-1.5B（本地路径）

### 6.2 复现步骤

```bash
# 1. 长度扫描（perf_len.py）：强制生成 N tokens + 逐 token 一致验证
python scripts/bench/perf_len.py --model <模型路径> --device nvidia \
    --lengths 64,256,1024,2048 --runs 3

# 2. HF attention 实现对比（attn_impl_compare.py）：eager/sdpa/flash 两两对比
python scripts/bench/attn_impl_compare.py --model <模型路径> --length 2048

# 3. logits 差异量级（hf_logits_diff.py / hf_logits_diff2.py）
python scripts/bench/hf_logits_diff.py --model <模型路径> --divergence 206
python scripts/bench/hf_logits_diff2.py --model <模型路径> --length 1024 --runs 6
```

### 6.3 输出解读

- `perf_len.py`：每长度每 run 输出 `llaisys=<耗时> HF=<耗时> len_ok=<是否生成满目标长度> token_identical=<逐位一致>`；不一致时输出首个分叉位置与两侧 token
- `attn_impl_compare.py`：三实现耗时 + 两两对比（identical / MISMATCH at 位置）
- `hf_logits_diff.py`：分叉前一步的 logits 差异量级（max/mean abs）+ 两实现 top-2 竞争度 + argmax
- `hf_logits_diff2.py`：多 run 生成找分叉，分叉前一步 logits 分析（注意：logits 为全量 forward 复算，与 generate 增量路径不等价——见 5.6）

### 6.4 已知注意事项

- 长度扫描中 llaisys 侧强制生成依赖"EOS 判断在 Python generate 层"这一实现事实（`python/llaisys/models/qwen2.py`），绕过 generate 直调 C 接口即绕过 EOS 停止
- 5090 与 C500 的 transformers 版本不同（5.14.1 vs 4.57.1），prompt 编码差 1 token，不影响 per-token 对比口径（生成数均为目标长度）
- 原始 run 输出留存于实验时的各平台任务日志；本报告数据表为汇总
