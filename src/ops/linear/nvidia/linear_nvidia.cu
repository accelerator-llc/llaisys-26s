#include "linear_nvidia.hpp"

#include "../../../device/nvidia/nvidia_utils.cuh"
#include "../../../utils.hpp"

#include <cublas_v2.h>

namespace llaisys::ops::nvidia {
using llaisys::device::nvidia::to_float;
using llaisys::device::nvidia::from_float;

// cuBLAS handle（单 device，懒创建；C++11 保证初始化线程安全）。
// V1 单 device；多 device 需 per-device handle（4.9 第二平台另议）。
// 4.6 linear 用 cuBLAS，与 llama.cpp/vLLM/PyTorch 对 F32/F16/BF16 矩阵乘的做法一致，
// cuBLAS 自动选 GEMM（m 大）与 GEMV（m=1 decode）路径。
static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = [] {
        cublasHandle_t h = nullptr;
        CUBLAS_CHECK(cublasCreate(&h));
        return h;
    }();
    return handle;
}

// bias 加法 kernel：Y[nn, k] += bias[k]，低精度 float 域加（cuBLAS gemm 不含 bias）。
template <typename T>
__global__ void add_bias_kernel(T *y, const T *bias, size_t n, size_t out_features) {
    size_t nn = blockIdx.x;
    if (nn >= n) {
        return;
    }
    T *y_row = y + nn * out_features;
    for (size_t k = threadIdx.x; k < out_features; k += blockDim.x) {
        y_row[k] = from_float<T>(to_float(y_row[k]) + to_float(bias[k]));
    }
}

// 4.9.6 T1 手写 GEMV kernel 前向声明（定义在文件末尾，linear() 内部 C500 分派需前置可见）。
template <typename T>
__global__ void gemv_fused_kernel(const T *x, const T *W, const T *bias, T *y, int M, int K);

static cudaDataType_t to_cuda_dtype(llaisysDataType_t type) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return CUDA_R_32F;
    case LLAISYS_DTYPE_F16:
        return CUDA_R_16F;
    case LLAISYS_DTYPE_BF16:
        return CUDA_R_16BF;
    default:
        ASSERT(false, "Linear: unsupported dtype for cuBLAS " << static_cast<int>(type));
        return CUDA_R_32F;
    }
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t val_type, size_t n, size_t in_features, size_t out_features, bool has_bias) {
    if (n == 0 || out_features == 0) {
        return;
    }
#if LLASYS_FUSE_LINEAR_KV
    // 4.9.6 T1：C500 decode 小形状走手写 GEMV + bias 融合（省 add_bias kernel + 手写快 11-19%）。
    // NVIDIA / prefill / 大形状走下方 cuBLAS 路径。is_c500() 一次探测缓存（T4）。
    if (llaisys::device::nvidia::is_c500() && n == 1 && val_type == LLAISYS_DTYPE_BF16 &&
        out_features <= 1536 && in_features <= 1536) {
        constexpr int GEMV_BLOCK = 256;
        int grid = (static_cast<int>(out_features) + 7) / 8;
        size_t smem = in_features * sizeof(llaisys::bf16_t);
        gemv_fused_kernel<llaisys::bf16_t><<<grid, GEMV_BLOCK, smem>>>(
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            has_bias ? reinterpret_cast<const llaisys::bf16_t *>(bias) : nullptr,
            reinterpret_cast<llaisys::bf16_t *>(out),
            static_cast<int>(out_features), static_cast<int>(in_features));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
#endif
    // Y = X @ W^T (+ b)。行主序 X(n,in), W(out,in), Y(n,out)，W 不转置（W[k] 行 = 第 k 输出特征）。
    // cuBLAS 列主序视角：Y^T(out,n) = W(out,in) * X^T(in,n)。
    //   cublasGemmEx: C(M=out, N=n, K=in) = op(A)(M,K) * op(B)(K,N)
    //   A=W, op_A=T（W 行主序 (out,in) 的内存 = W^T 列主序 (in,out)，op T -> 逻辑 (out,in)），lda=in
    //   B=X, op_B=N（X 行主序 (n,in) 的内存 = X^T 列主序 (in,n)），ldb=in
    //   C=Y, ldc=out（Y 行主序 (n,out) 的内存 = Y^T 列主序 (out,n)）
    //   不做任何多余转置，仅靠 op/leading-dim 适配行主序。
    // computeType=CUBLAS_COMPUTE_32F：低精度在 float 域累加，与 cpu::linear 一致。
    cublasHandle_t handle = get_cublas_handle();
    float alpha = 1.0f, beta = 0.0f;
    cudaDataType_t dtype = to_cuda_dtype(val_type);
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            static_cast<int>(out_features), static_cast<int>(n), static_cast<int>(in_features),
                            &alpha, weight, dtype, static_cast<int>(in_features),
                            in, dtype, static_cast<int>(in_features),
                            &beta, out, dtype, static_cast<int>(out_features),
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    if (has_bias) {
        constexpr int block = 256;
        int grid = static_cast<int>(n);
        switch (val_type) {
        case LLAISYS_DTYPE_F32:
            add_bias_kernel<float><<<grid, block>>>(
                reinterpret_cast<float *>(out), reinterpret_cast<const float *>(bias), n, out_features);
            break;
        case LLAISYS_DTYPE_BF16:
            add_bias_kernel<llaisys::bf16_t><<<grid, block>>>(
                reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(bias), n, out_features);
            break;
        case LLAISYS_DTYPE_F16:
            add_bias_kernel<llaisys::fp16_t><<<grid, block>>>(
                reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(bias), n, out_features);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(val_type);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

// ============================================================================
// 4.9.6 方向 E：手写 GEMV kernel（decode m=1，bf16，float 域累加 + bias 融合）
// 参考 gemv_v3c 实测（增量验证报告）：uint4 向量化读 16B=8 bf16、x 进 smem 复用、
// 每 warp 一行 + 32-lane shfl 归约、尾部 acc+bias[row] 一次舍入（两次舍入->一次）。
// C500 warpSize=64 但 cucc 保持 32-lane 子组 shuffle 语义（CR 实测 __shfl_down_sync 安全）。
// 仅 C500 decode 小形状（M,K<=1536）走此路径；NVIDIA/prefill/大形状走 cuBLAS。
// ============================================================================

template <typename T>
__global__ void gemv_fused_kernel(const T *x, const T *W, const T *bias,
                                  T *y, int M, int K) {
    int warp_in_block = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * 8 + warp_in_block;
    if (row >= M) {
        return;
    }
    extern __shared__ unsigned char smem_raw[];
    T *sx = reinterpret_cast<T *>(smem_raw);
    for (int k = threadIdx.x; k < K; k += 256) {
        sx[k] = x[k];
    }
    __syncthreads();
    const T *w_row = W + (size_t)row * K;
    float acc = 0.0f;
    const uint4 *w4 = reinterpret_cast<const uint4 *>(w_row);
    const uint4 *x4 = reinterpret_cast<const uint4 *>(sx);
    int n8 = K / 8;
    for (int i = lane; i < n8; i += 32) {
        uint4 wv = w4[i], xv = x4[i];
        const T *wp = reinterpret_cast<const T *>(&wv);
        const T *xp = reinterpret_cast<const T *>(&xv);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            acc += to_float(wp[j]) * to_float(xp[j]);
        }
    }
    // 尾部（K 非 8 倍数；本项目形状均可整除，防御性处理）
    for (int i = n8 * 8 + lane; i < K; i += 32) {
        acc += to_float(w_row[i]) * to_float(sx[i]);
    }
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, off);
    }
    if (lane == 0) {
        float v = acc + (bias ? to_float(bias[row]) : 0.0f);
        y[row] = from_float<T>(v);
    }
}

// k+v 合并 GEMV kernel：一次 launch 处理 k/v 两段（grid 覆盖 M_k+M_v 行）。
// kernel 内按行段取 W/bias/y 指针（row < M_k -> k 段；row >= M_k -> v 段，seg_row=row-M_k）。
template <typename T>
__global__ void gemv_kv_fused_kernel(const T *x, const T *W_k, const T *W_v,
                                     const T *bias_k, const T *bias_v,
                                     T *y_k, T *y_v, int M_k, int M_v, int K) {
    int warp_in_block = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * 8 + warp_in_block;
    int total = M_k + M_v;
    if (row >= total) {
        return;
    }
    bool is_k = (row < M_k);
    const T *W = is_k ? W_k : W_v;
    const T *bias = is_k ? bias_k : bias_v;
    T *y = is_k ? y_k : y_v;
    int seg_row = is_k ? row : (row - M_k);
    extern __shared__ unsigned char smem_raw[];
    T *sx = reinterpret_cast<T *>(smem_raw);
    for (int k = threadIdx.x; k < K; k += 256) {
        sx[k] = x[k];
    }
    __syncthreads();
    const T *w_row = W + (size_t)seg_row * K;
    float acc = 0.0f;
    const uint4 *w4 = reinterpret_cast<const uint4 *>(w_row);
    const uint4 *x4 = reinterpret_cast<const uint4 *>(sx);
    int n8 = K / 8;
    for (int i = lane; i < n8; i += 32) {
        uint4 wv = w4[i], xv = x4[i];
        const T *wp = reinterpret_cast<const T *>(&wv);
        const T *xp = reinterpret_cast<const T *>(&xv);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            acc += to_float(wp[j]) * to_float(xp[j]);
        }
    }
    for (int i = n8 * 8 + lane; i < K; i += 32) {
        acc += to_float(w_row[i]) * to_float(sx[i]);
    }
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, off);
    }
    if (lane == 0) {
        float v = acc + (bias ? to_float(bias[seg_row]) : 0.0f);
        y[seg_row] = from_float<T>(v);
    }
}

// linear_kv_fused：一次调用算 k+v 两个 GEMV（带 bias）。
// C500 decode 小形状走 gemv_kv_fused（省 1 launch + 合并后 kernel 更高效）；
// 其余走两次 linear（cuBLAS）。qwen2.cpp NVIDIA 路径调用，CPU 走原两次 linear。
void linear_kv_fused(std::byte *out_k, std::byte *out_v, const std::byte *in,
                     const std::byte *w_k, const std::byte *w_v,
                     const std::byte *b_k, const std::byte *b_v,
                     llaisysDataType_t val_type, size_t n, size_t in_features,
                     size_t out_k_features, size_t out_v_features) {
#if LLASYS_FUSE_LINEAR_KV
    if (llaisys::device::nvidia::is_c500() && n == 1 && val_type == LLAISYS_DTYPE_BF16 &&
        out_k_features <= 1536 && out_v_features <= 1536 && in_features <= 1536) {
        constexpr int GEMV_BLOCK = 256;
        int total = static_cast<int>(out_k_features + out_v_features);
        int grid = (total + 7) / 8;
        size_t smem = in_features * sizeof(llaisys::bf16_t);
        gemv_kv_fused_kernel<llaisys::bf16_t><<<grid, GEMV_BLOCK, smem>>>(
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(w_k),
            reinterpret_cast<const llaisys::bf16_t *>(w_v),
            b_k ? reinterpret_cast<const llaisys::bf16_t *>(b_k) : nullptr,
            b_v ? reinterpret_cast<const llaisys::bf16_t *>(b_v) : nullptr,
            reinterpret_cast<llaisys::bf16_t *>(out_k),
            reinterpret_cast<llaisys::bf16_t *>(out_v),
            static_cast<int>(out_k_features), static_cast<int>(out_v_features),
            static_cast<int>(in_features));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
#endif
    // 非 C500 / prefill / 大形状：走两次 cuBLAS linear（NVIDIA 与 C500 prefill 共用）
    linear(out_k, in, w_k, b_k, val_type, n, in_features, out_k_features, b_k != nullptr);
    linear(out_v, in, w_v, b_v, val_type, n, in_features, out_v_features, b_v != nullptr);
}

} // namespace llaisys::ops::nvidia
