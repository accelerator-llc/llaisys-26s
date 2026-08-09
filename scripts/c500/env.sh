#!/bin/bash
# C500 会话级环境变量（每次构建/测试前 source）。
# 非交互 SSH 不加载 .bashrc，以下变量必须显式导出（实测缺失任一均导致失败）。
#
# 用法：source scripts/c500/env.sh

export MACA_PATH=/opt/maca              # triton/metax 后端 maca_home_dirs() 依赖，缺失 import 崩溃
export CUCC_PATH=/opt/maca/tools/cu-bridge  # cucc 内部路径拼接依赖
export XMAKE_ROOT=y                     # root 用户下 xmake 必须
export PATH=/usr/local/bin:/root/.local/bin:$PATH  # /usr/local/bin(nvcc wrapper) + /root/.local/bin(xmake)
export LIBRARY_PATH=/opt/maca/links:/opt/maca/lib  # 链接期找 -lcublas 等（links 软链 -> mcblas/runtime_cu）
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/links:/opt/maca/mxgpu_llvm/lib  # 运行期加载 libmcblas/libruntime_cu
