# LLAISYS 作业报告：作业 1-4 完整实现

## 概述

本报告为 LLAISYS 推理引擎**作业 0-4 的整体报告**：各作业的做法、关键技术决策、值得记录的问题（影响技术方向的错误与教训）、与 PyTorch 的差异设计、性能数据与复现流程。作业 0（入门：环境组件安装、构建、首次运行、测试模型下载）已完成，其内容已融入第七节复现流程。代码最终状态为分支 homework（作业 1-4 各阶段提交 + 第二平台适配 + 性能优化），在 **CPU、NVIDIA（RTX 5090 / 4090D / 4060）与国产 GPU（沐曦 MetaX C500）** 上全部通过测试，推理逐 token 与 PyTorch 一致。

## 一、作业 1：Tensor

**做法**：实现 Tensor 的全部桩函数，内存模型为"元数据（dtype/shape/strides）+ 共享 storage + 字节偏移"：
- `load`：主机/设备四方向拷贝（H2H/H2D/D2H/D2D），目标张量须连续（入口断言校验），memcpy 直拷
- `isContiguous`：从末维向前验证 strides 与形状的连续性关系（零维/单元素维处理正确）
- `view`：连续张量零拷贝形状重解释；非连续时校验合并维度是否可连续化，不可 view 抛错（含溢出检查）
- `permute`：仅重排元数据，零拷贝
- `slice`：字节偏移视图（偏移按字节、strides 按元素——两套单位的正确混用是本作业的关键正确性点）

**Challenging features（顺带完成）**：`contiguous`（非连续物理重排）、`reshape`（连续张量零拷贝形状重解释，与 `view` 的区别在于自动插入 `contiguous` 兜底）、`to`（跨设备拷贝，修复了 D2H 场景设备选择错误的缺陷）。`reshape` 的零拷贝能力在作业 3 的注意力头分组中直接复用（见第三节）。

## 二、作业 2：CPU 算子

**做法**：7 个算子（argmax/embedding/linear/rms_norm/rope/self_attention/swiglu）+ 既有 add，统一三层结构（op.cpp 校验 + 按设备分发 / cpu 模板实现 / float 域中转累加），全部支持 F32/F16/BF16 三精度，低精度一律提升到 float 域计算后截断写回（半精度直接累加会吃精度）。

**数值语义与 PyTorch 逐项对齐**（对齐是实现正确性的关键）：
- argmax：NaN 传播、平局取首个
- linear：`Y = XW^T (+b)`，权重不转置（按 W 行取输出特征）；bias 可选
- rms_norm：`W·X/√(mean(X²)+eps)`，float 域
- rope：`φ=p/θ^(2j/d)`，a/b 对半旋转，支持就地
- self_attention：因果掩码（`j > i+(kvlen-qlen)`，等价 `tril(diagonal=S-L)`——生成场景 qlen=1 时能看全部历史）+ GQA 分组（`h/groups`，等价 PyTorch 的 `repeat_interleave` 连续复制）+ softmax 减最大值（防 exp 溢出）

8 算子 × 3 dtype × 大/小尺寸测试全过。

**CPU 性能优化**：linear 实现**持久线程池并行**（常驻 worker 线程，按输出特征分块并行，避免每次调用创建/join 线程的开销——参照 vLLM CPU/llama.cpp 的常驻 worker 池思路），并行阈值按总计算量（n×out×in）判断，decode 小矩阵不并行（防并行负收益）；release 构建启用 `-march=native -flto`，使 linear 内层循环可被 SIMD 向量化。

## 三、作业 3：模型推理

**做法**：实现 Qwen2 的完整推理链路：embedding → 28 层（rms_norm → q/k/v 投影 → RoPE → KV-cache → 因果自注意力 → o_proj → 残差 → MLP：gate/up 投影 + SwiGLU + down）→ final_norm → lm_head → argmax 采样。KV-cache 按层预分配、增量写入（每步只计算新 token 的 K/V，历史复用）；权重经 safetensors 手动解析加载（不依赖 PyTorch 推理）：safetensors 的 numpy backend 无法读取 bf16 张量（环境 numpy 无 bfloat16 dtype，get_tensor 抛 TypeError），故手动解析文件取每个权重的原始 bf16 字节后直接加载；C API（extern "C" 边界 try/catch 兜底）+ ctypes 绑定 + Python 包装。

**与 PyTorch 的关键差异：张量布局约定**。PyTorch/HF 的注意力实现中，q/k/v 需要 `view + transpose` 把头维从连续布局挪到中间（物理重排），attention 后再次转置恢复。我们约定 **head 维始终合并进最后一维的连续存储**：linear 输出 `(n, nh*dh)` 后经 `reshape` 零拷贝解释为 `(n, nh, dh)`（同一块内存，无任何拷贝），self_attention 算子接口直接接受该形状并在 kernel 内按头分组；输出同理 reshape 回 2D 喂 o_proj。这一约定避免了 PyTorch 式 transpose 类物理重排（省拷贝、省算子调用）。

顺带说明老师预留的 `rearrange` 算子（README 未要求实现）在本项目中未开发的原因，分两层：
1. **直接原因**：推理过程从不产生非连续张量——布局约定消灭了 transpose 类需求，模型层只使用 `create`/`slice`/`reshape`（`slice` 仅对连续张量第 0 维切，产生连续子视图），整个链路无需要连续化的对象；
2. **算子层面**：即便未来出现非连续张量，连续化也由 Tensor 的挑战特性 `contiguous()` 承担（rearrange 的官方语义即"同 shape 不同 strides 的张量间拷贝"）——故该算子在本项目中无使用场景。

## 四、作业 4：CUDA 集成与第二平台

### 4.1 NVIDIA 路径

- **Runtime API**：12 个函数（设备查询/流/内存/四方向拷贝），以函数指针表适配器模式接入上层。过程中修正了老师桩代码的一处签名错误（`memcpyAsync` 缺流参数，与权威头文件声明及 CPU 实现不一致——按"桩代码有误不照搬"原则以权威声明为准修复）。
- **CUDA 算子**：8 个算子的 NVIDIA 实现。linear 使用 cuBLAS（`cublasGemmEx`），通过 op/leading-dim 适配行主序（零转置）；self_attention 为每 (i,h) 一块的融合 kernel（点积 + 因果掩码 + softmax + 加权求和一气呵成）。
- **模型推理**：权重与 KV-cache 全程驻留显存，每步仅 8 字节控制信号过桥（D2H 取 argmax 结果）。

### 4.2 第二平台：沐曦 C500（cu-bridge 路线）

按作业要求"将同一套实现适配到第二平台"，选择沐曦 C500。**关键事实**：cu-bridge 兼容层在**编译期**将 CUDA 符号翻译为沐曦原生符号（官方定位为"用 CUDA 语言开发、编译期生成 MXMACA 目标文件 + API mapping"，见 gitee metax-maca/cu-bridge 官方文档；具体映射形态 `cublasCreate/cublasGemmEx` → `mcblasCreate/mcblasGemmEx`（由 libmcblas 提供）、`cudaMalloc/cudaMemcpy` → `wcudaMalloc/wcudaMemcpy`（由 libruntime_cu 提供）经编译产物符号级（nm）实证）。基于该机制，适配完全落在环境层（仓库零改动）：nvcc→cucc 翻译 wrapper + 链接软链（`-lcublas` 等解析到沐曦替代库）+ 环境变量导出，以 `scripts/c500/` 脚本化交付。

### 4.3 性能优化（模型层结构性优化为主）

- **张量池（TensorPool）**：decode 热循环的临时张量经池复用（容量只增不减），消除热循环 cudaMalloc/cudaFree——5090 实测分配次数降 98.7%（30,483→395）；基线 128 步推理 0.41s 中，分配/释放的 API 时间合计约 573ms（含同步开销），是 wall-clock 最大单项开销。
- **rope 直写 KV-cache**：k 旋转结果直接写入 cache 视图（rope 支持 out≠in），省 rope→scratch + D2D memcpy 往返（56 次/步）；v 不旋转，linear 输出直写 cache。
- **数据闭环**：权重与 KV-cache 全程驻留显存，每步仅 8 字节控制信号过桥（D2H 取 argmax 结果）。
- **效果**：5090 端到端 0.41s→0.28s；以上为平台无关的模型层优化，C500 随同一套代码免费受益。
- **kernel 级优化的端到端裁决**：C500 上曾实施手写 GEMV 与融合 kernel（微基准显示收益），端到端 4 状态同刻裁决无收益后全部删除（详见第五节）——优化决策以端到端实测为准。

## 五、值得记录的问题（影响技术方向的错误与教训）

1. **"优化生效"必须端到端实测验证——融合优化的假阳性**：实现了手写 GEMV（uint4 向量化 + smem 复用 + bias 融合）、k+v 合并、rms_norm+add 融合三个优化，端到端测得 0.60→0.53s。收官全量审查发现：两个融合算子的开关宏定义在仅 `.cu` 可见的头文件中，模型层（`.cpp`）中条件编译恒为假——**融合代码编译进库但从未执行**（死代码）。随后用 4 状态 × 3 次的编译开关矩阵**同刻**实测裁决：三个优化在端到端均无收益（差异 0%~1.8%，<2% 噪声），全部删除；同刻 ABBA 交替对比（20 次）证实删除后与优化前基线（1ad818b）性能等价（均值差 0.6%）。历史记录的 0.60→0.53s 差异在 C500 跨时段环境波动量级内（0.05s 量级），不能归因于任何单项优化；性能基线（~0.6s）包含平台无关的张量池与 rope 直写（N 卡优化时期的模型层成果，随代码带入 C500，其在 C500 上的单独贡献未做对照实验）。**微基准显示单次手写 GEMV 快 11%，但端到端无收益——微基准收益不等于端到端收益，必须以端到端实测为最终裁决**。
2. **性能测量工具的坑**：C500 无 nsys/ncu；沐曦自带 mcTracer 的耗时数据经校准发现被 tracer 放大 ~43 倍（launch 实测 6.46μs vs trace 278μs），仅事件计数可用——**性能分析必须实测校准，工具数据不可直接采信**。据此改用独立微基准（launch 单价、GEMV 带宽、K 扫描固定开销）建立平台画像。
3. **平台画像决定优化方向**：C500 实测 kernel 固定开销 ~8μs（K 扫描）、launch 单价 6.46μs、mcblas 小形状 GEMV 效率分层（约 524 GB/s vs 大形状 1089 GB/s）——据此评估手写/融合方向；但端到端裁决推翻了微基准预期（见问题 1），证明**优化决策的最终裁判是目标平台端到端实测**。
4. **编译期符号翻译链的发现**（4.2 节）：第二平台适配的可行性判断基于 nm 符号级实证（cublas→mcblas、cuda→wcuda），而非厂商宣传——据此设计了仓库零改动的环境层适配方案。
5. **uint4 对齐的潜在未定义行为**：手写 GEMV 的向量化读取要求 16B 对齐，分派条件初版未检查（K 非 8 倍数时未对齐读为 UB）——代码审查发现后于分派条件补充对齐检查（当前模型形状均为 8 倍数，未实际触发；该修复随后随优化删除一并移除——优化经端到端裁决已全部删除，当前代码无手写 GEMV）。
6. **linear 用 cuBLAS 而非手写的决策（C500）**：C500 上微基准曾显示手写小形状 GEMV 更快（11-19%），一度计划按形状分派（小形状手写 + 大形状用 mcblas）；端到端 4 状态同刻裁决手写无收益（0%）——微基准收益在端到端不成立，全部删除恢复纯 cuBLAS。
7. **内存/显存泄露检测（行为级）**：本地 CPU 6 轮推理 RSS 零增长；5090/C500 同实例多轮推理显存恒定（张量池复用生效），进程退出后残留显存（519/190 MiB）经延迟观察 90 秒内归零（驱动/兼容层延迟回收，非泄露）——无内存/显存泄露。

## 六、Challenging features

Tensor 的 `contiguous` / `reshape` / `to` 三个挑战特性均已实现并在作业 3/4 中实际使用（`reshape` 支撑注意力头分组的零拷贝布局；`to` 支撑 CPU/NVIDIA 权重加载）。

## 七、复现流程

### 结构

```
include/llaisys/        C API 头文件
src/tensor/             Tensor（含 contiguous/reshape/to）
src/ops/<name>/{op.cpp, cpu/, nvidia/}   算子调度 + 双平台实现
src/device/{cpu,nvidia}/                 运行时 API（函数指针表适配器）
src/models/qwen2/       Qwen2 模型（KV-cache、张量池、rope 直写）
src/llaisys/            C API 包装
python/llaisys/         ctypes 绑定 + Python 模型层（safetensors 解析）
scripts/c500/           沐曦 C500 环境适配脚本（机器级 + 会话级 + 说明）
xmake/nvidia.lua        CUDA 构建配置（--nv-gpu 开关）
test/                   测试（runtime/tensor/infer + 8 算子）
```

### 环境版本（以下版本验证通过；其他版本可能需适配）

| 平台 | 关键版本 |
|---|---|
| 本地 CPU / 4060 | Ubuntu 24.04 / CUDA 13.3 / Python 3.12（.venv）/ transformers 4.57.6 |
| 5090 | CUDA 12.8 / Python 3.12 / transformers 5.14.1 |
| 4090D | CUDA 12.8（NGC 镜像）/ Python 3.12 / transformers 5.14.1 |
| 沐曦 C500 | MACA 3.2.1.10（驱动 3.8.30，含 cu-bridge）/ Python 3.10.10（/opt/conda）/ torch 2.6.0+metax3.2.1.3 / transformers 4.57.1 |

### 环境准备（作业 0）

```bash
# 0. 获取代码（分支 homework 为最终版本）
git clone https://github.com/accelerator-llc/llaisys-26s.git && cd llaisys-26s && git checkout homework
git log --oneline -1   # 校验：HEAD 应为作业1-4 最终版提交
# 1. 安装 xmake（https://xmake.io）与 C++ 编译器（gcc/clang）
curl -fsSL https://xmake.io/shget.text | bash   # 国内: XMAKE_GET_SOURCE=gitee
source ~/.xmake/profile                          # 装完需手动 source 生效
# 2. Python 依赖：pip install torch transformers safetensors
#    （GPU 路径另需 CUDA 12.x/13.x 工具链：nvcc + cuBLAS）
# 3. 下载模型（DeepSeek-R1-Distill-Qwen-1.5B，bf16）：
#    HuggingFace: snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
#    国内网络可用 modelscope: modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

### 构建与测试

```bash
# CPU / NVIDIA 通用
export XMAKE_ROOT=y   # root 用户必需（xmake 默认拒绝 root）
xmake f --nv-gpu=[y|n] -cv && xmake && xmake install
export PYTHONPATH=$(pwd)/python
python test/test_runtime.py --device [cpu|nvidia]
python test/test_tensor.py
for f in add swiglu embedding argmax rms_norm rope linear self_attention; do
  python test/ops/$f.py --device [cpu|nvidia]
done
python test/test_infer.py --model <模型目录> --test --device [cpu|nvidia]
```

预期输出：各测试输出 `Test passed`；`test_infer` 最后输出 `Test passed!`（含 llaisys 与 HF 两段 Time elapsed，逐 token 一致）。

### 沐曦 C500（环境层适配，仓库代码零改动）

```bash
# 机器级（root，一次）：nvcc->cucc wrapper + 链接软链
bash scripts/c500/setup-machine.sh
# 会话级（每次）：MACA/CUCC/LIBRARY/LD 环境变量
source scripts/c500/env.sh
export PATH=/opt/conda/bin:$PATH   # conda 环境（torch+metax 在此）
# 之后同一套 xmake/测试命令（--device nvidia，cu-bridge 兼容层）
python test/test_infer.py --model /data/DeepSeek-R1-Distill-Qwen-1.5B --test --device nvidia
```

前提：MACA SDK 3.2.1+ 与 cu-bridge 装于 `/opt/maca`（含 mxcc、libmcblas、libruntime_cu）；Python 环境含 torch+metax、transformers、safetensors。详细说明见 `scripts/c500/README.md`。

## 八、性能数据（test_infer：prompt 10 token（transformers 4.57 系）或 9 token（5.14 系）+ 81 生成，EOS 提前停；生成部分各平台均为 81 token）

模型：DeepSeek-R1-Distill-Qwen-1.5B（bf16）；prompt 经 chat template 展开；贪心解码。加速比 = PyTorch(HF transformers) 端到端 generate 耗时 / llaisys 端到端 generate 耗时；per-token 按 81 生成 token 计（端到端混合口径，含 prefill）。

| 平台 | 环境 | llaisys (s) | PyTorch (s) | 加速比 | llaisys per-token (ms) |
|---|---|---|---|---|---|
| 本地 CPU | i9-13980HX 24C32T ｜ 本地 ｜ Ubuntu 24.04 | 11.23 | 5.60 | 0.50x | 138.6 |
| RTX 4060 | RTX 4060 Laptop 8G ｜ 本地 ｜ Ubuntu 24.04 / CUDA 13.3 | 1.22 | 1.56 | 1.28x | 15.1 |
| RTX 5090 | RTX 5090 32G ｜ 云端 ｜ CUDA 12.8 | 0.28 | 1.76 | 6.30x | 3.5 |
| RTX 4090D | RTX 4090D 32G ｜ 云端 ｜ NGC 镜像 CUDA 12.8 | 0.39 | 2.04 | 5.23x | 4.8 |
| 沐曦 C500 | MetaX C500 64G ｜ 云端 ｜ MACA 3.2.1.10（cu-bridge） | 0.59 | 2.80 | 4.72x | 7.3 |

数据说明：每平台连续 3 次独立进程测量取均值（逐次原始值见回归报告），计时不含模型加载与 tokenizer 初始化；GPU 平台 llaisys 均快于 PyTorch（1.28x~6.30x）；CPU 平台 llaisys 慢于 PyTorch（CPU 非优化重点）。各平台 transformers 版本不同（4.57/5.14 系）导致 prompt 编码差 1 token（10 vs 9），生成部分均为 81 token，per-token 口径一致。

### 长度扫描（per-token vs 生成长度，强制生成不截断，5090 + C500）

| 平台 | 64 | 256 | 1024 | 2048 |
|---|---|---|---|---|
| 5090 llaisys per-token (ms) | 3.4 | 3.6 | 4.3 | 5.4 |
| 5090 加速比 (HF/llaisys) | 3.9x | 3.8x | 3.1x | 2.5x |
| C500 llaisys per-token (ms) | 7.4 | 8.4 | 13.0 | 18.7 |
| C500 加速比 (HF/llaisys) | 3.6x | 3.2x | 2.1x | 1.4x |

per-token 随长度近似线性上升（符合 self_attention O(kvlen)），加速比随长度下降（长序列下与 PyTorch 优化 attention 的差距缩小）——长序列 attention 是后续优化空间。

**长序列 token 一致性边界**：验收口径（test_infer `--test`，81 tokens，四平台）逐 token 一致完整成立；强制生成扩展测试（1024+ 长序列）下 llaisys 与 HF 出现概率性分叉（分叉位置漂移：206/1129/1213，1024 以下一致）。经分析属浮点数值差异的普遍现象（对照实验：HF 的 eager/sdpa/flash 三种实现互相之间亦分叉），非实现缺陷，未做额外修复。详见 [LEN_SCAN_REPORT.md](LEN_SCAN_REPORT.md)。

## 九、平台支持状态

| 平台 | 构建 | Runtime | 8 算子 | 推理（逐 token 一致） | 性能 |
|---|---|---|---|---|---|
| Linux CPU | ✅（--nv-gpu=n） | ✅ | ✅ | ✅ | 基线（朴素实现） |
| NVIDIA（4060/5090/4090D） | ✅（--nv-gpu=y，CUDA 12.x） | ✅ | ✅ | ✅ | 1.28x~6.30x vs PyTorch |
| 沐曦 C500 | ✅（cu-bridge 环境层） | ✅ | ✅ | ✅ | 4.72x vs PyTorch |

三平台（CPU/NVIDIA/C500）全部通过：test_runtime、test_tensor、8 算子 × 三 dtype × 大小尺寸、test_infer 逐 token 与 PyTorch 一致。4090D 为补充验证平台（第二款 NVIDIA 卡，扩大卡型覆盖）。

## 十、后续优化方向

1. **CUDA Graph**：decode 形状固定符合录制-重放条件（vLLM 标准优化，launch 占比高时收益显著）；C500 capture API 已实测可用，完整验证与实测裁决待做。
2. **长序列 attention**：长度扫描实测（强制生成 64/256/1024/2048，5090 + C500）：per-token 随长度近似线性上升（5090 3.4→5.4ms、C500 7.4→18.7ms，符合 attention O(kvlen)），加速比随长度下降（5090 3.9x→2.5x、C500 3.6x→1.4x）——长序列下与 PyTorch 优化 attention 的差距缩小，attention 分块/访存优化有明确空间。
3. **C500 batched GEMV（gate+up）**：微基准实测省 5.8μs/组，端到端收益有限未实施。
4. **5090 手写 GEMV 端到端验证**：最优手写 GEMV 实现在 5090 微基准显示明显提升（快 38-55%），但端到端未测——后续做端到端 A/B 验证，若成立为潜在优化空间。

## 十一、结论

作业 1-4 全部完成：Tensor 含三个挑战特性；CPU/CUDA 双平台算子数值语义与 PyTorch 对齐；Qwen2 推理在 NVIDIA 与沐曦 C500 双平台逐 token 一致（验收口径 81 tokens）；性能上 GPU 平台显著快于 PyTorch（最高 6.30x）。过程中通过"先做后测、数据裁决"的方法论（端到端实测推翻微基准预期、全量审查发现死代码优化）保证了优化决策的真实性。
