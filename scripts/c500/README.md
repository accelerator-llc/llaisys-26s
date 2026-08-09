# C500（沐曦 MetaX C500）平台适配

LLAISYS 仓库代码**零改动**，通过环境层适配即可在 C500 上编译运行。原理：cu-bridge 将 CUDA 源码翻译为 MXMACA，cuBLAS/cudart 符号经软链映射到 MACA 替代库（libmcblas/libruntime_cu）。

## 前提

- MACA SDK 3.2.1+ 装于 `/opt/maca`（含 mxcc、libmcblas、libruntime_cu）
- cu-bridge 装于 `/opt/maca/tools/cu-bridge`（含 bin/cucc、include/ cuda 头文件）
- conda python（如 `/opt/conda/bin/python`）含 torch+metax、transformers、safetensors 等

## 1. 机器级配置（root，一次执行）

```bash
bash scripts/c500/setup-machine.sh
```

创建：
- `/usr/local/bin/nvcc` wrapper（exec cucc，让 xmake `add_rules("cuda")` 找到的"nvcc"即 cucc）
- `/opt/maca/links/` 软链：`libcublas.so->libmcblas.so`、`libcudart_static.so->libruntime_cu.so`、`libcudadevrt.so->libruntime_cu.so`

## 2. 会话级环境（每次构建/测试前 source）

```bash
source scripts/c500/env.sh
```

导出：`MACA_PATH`、`CUCC_PATH`、`XMAKE_ROOT`、`PATH`（含 nvcc wrapper + xmake）、`LIBRARY_PATH`（链接期）、`LD_LIBRARY_PATH`（运行期）。非交互 SSH 不加载 `.bashrc`，必须显式 source。

## 3. 构建与测试

```bash
xmake f --nv-gpu=y && xmake && xmake install
PYTHONPATH=python /opt/conda/bin/python test/test_runtime.py --device nvidia
# 8 个算子
for op in add swiglu embedding argmax rms_norm rope linear self_attention; do
  PYTHONPATH=python /opt/conda/bin/python test/ops/$op.py --device nvidia
done
PYTHONPATH=python /opt/conda/bin/python test/test_infer.py --model <model_path> --test --device nvidia
```

## 4. 已知坑

- **非交互 SSH 环境变量缺失**：`MACA_PATH`/`CUCC_PATH` 等不会从 `.bashrc` 继承，必须 `source env.sh` 或命令内显式 `export`。
- **xmake 探测 cuda devices 警告**：`find_cudadevices: No such file or directory`（cucc 不提供该工具），仅警告不影响编译。
- **waveSize 差异**：沐曦 GPU waveSize=64（官方 FAQ），项目 kernel 用 `__shfl_down_sync(0xffffffff,...)` 与 `WARP_SIZE=32` 硬编码，实测零改动通过全部测试（cu-bridge/maca 兼容处理）；长序列/大输入若有归约异常，需按官方建议适配 mask 至 64 位。
- **密码/敏感信息**不写入仓库。

## 5. 参考

- cu-bridge 使用指南：https://gitee.com/metax-maca/cu-bridge/blob/master/docs/02_User_Manual/README.md
- 沐曦开发者社区：https://developer.metax-tech.com/doc/index
