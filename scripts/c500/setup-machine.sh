#!/bin/bash
# C500（沐曦 MetaX C500）机器级环境配置（root，一次执行）。
#
# 作用：创建 cu-bridge nvcc wrapper + /opt/maca/links 软链，使 xmake add_rules("cuda")
# 找到的"nvcc"即 cucc（CUDA->MXMACA 翻译编译器），链接时 -lcublas/-lcudart_static
# 解析到 MACA 替代库（libmcblas/libruntime_cu）。
#
# 前提：MACA SDK 已装于 /opt/maca，cu-bridge 已装于 /opt/maca/tools/cu-bridge。
# 参考：cu-bridge 使用指南 https://gitee.com/metax-maca/cu-bridge
set -e

MACA_LIB=/opt/maca/lib
CUCC=/opt/maca/tools/cu-bridge

# 1. nvcc -> cucc wrapper：xmake cuda 规则调用"nvcc"即走 cucc 翻译
cat > /usr/local/bin/nvcc <<EOF
#!/bin/bash
# C500: nvcc -> cucc (cu-bridge) 翻译 wrapper
export CUCC_PATH=${CUCC}
export CUDA_PATH=\${CUCC_PATH}
exec \${CUCC_PATH}/bin/cucc "\$@"
EOF
chmod +x /usr/local/bin/nvcc

# 2. /opt/maca/links 软链：cuBLAS/cudart 符号 -> MACA 替代库
#    项目链接 -lcublas -lcudart_static -lcudadevrt，C500 无这些库，需软链到 mcblas/runtime_cu。
mkdir -p /opt/maca/links
ln -sf ${MACA_LIB}/libmcblas.so    /opt/maca/links/libcublas.so
ln -sf ${MACA_LIB}/libruntime_cu.so /opt/maca/links/libcudart_static.so
ln -sf ${MACA_LIB}/libruntime_cu.so /opt/maca/links/libcudadevrt.so

echo "C500 机器级配置完成："
echo "  - nvcc wrapper -> ${CUCC}/bin/cucc"
echo "  - /opt/maca/links: libcublas->libmcblas, libcudart_static/libcudadevrt->libruntime_cu"
echo "会话级环境变量请 source scripts/c500/env.sh"
